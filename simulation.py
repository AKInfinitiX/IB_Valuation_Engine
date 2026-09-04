import numpy as np
from data_pipeline import fetch_data
from portfolio import PortfolioModel

class MonteCarloEngine:
    def __init__(self, portfolio_model):
        self.model = portfolio_model
        self.L = self.model.get_cholesky_decomposition()
        self.garch_params = self.model.fit_garch_models()
        self.num_assets = len(self.model.mean_returns)
        self.tickers = self.model.tickers

    def simulate_paths(self, days=252, simulations=10000):
        dt = 1.0
        mu = self.model.mean_returns
        
        # Array to store simulated daily log returns: (Days x Simulations x Assets)
        log_returns_sim = np.zeros((days, simulations, self.num_assets))
        
        # Broadcast initial variance across all 10,000 paths: (Simulations x Assets)
        initial_vols = np.array([self.garch_params[t]['last_vol'] for t in self.tickers])
        current_vars = np.tile(initial_vols ** 2, (simulations, 1))
        
        # Extract parameter vectors: (Assets,)
        omega = np.array([self.garch_params[t]['omega'] for t in self.tickers])
        alpha = np.array([self.garch_params[t]['alpha'] for t in self.tickers])
        beta = np.array([self.garch_params[t]['beta'] for t in self.tickers])
        
        # Pre-generate independent standard normal shocks: (Days x Simulations x Assets)
        Z = np.random.standard_normal((days, simulations, self.num_assets))
        
        # Correlate shocks across assets using the Cholesky matrix
        correlated_shocks = Z @ self.L.T
        
        # Step-by-step path-dependent GARCH(1,1) simulation
        for d in range(days):
            current_vols = np.sqrt(np.maximum(current_vars, 1e-8))
            
            # Geometric Brownian Motion increment with dynamic conditional variance
            drift = (mu - 0.5 * current_vars) * dt
            diffusion = current_vols * correlated_shocks[d]
            r_t = drift + diffusion
            log_returns_sim[d] = r_t
            
            # GARCH variance recurrence: sigma_{t+1}^2 = omega + alpha * r_t^2 + beta * sigma_t^2
            current_vars = omega + alpha * (r_t ** 2) + beta * current_vars
            
        # Accumulate returns from Day 0 (base value = 1.0)
        cumulative_log_returns = np.vstack([
            np.zeros((1, simulations, self.num_assets)),
            np.cumsum(log_returns_sim, axis=0)
        ])
        
        price_paths = np.exp(cumulative_log_returns)
        print(f"GARCH Monte Carlo simulation complete. Output shape: {price_paths.shape}")
        return price_paths

if __name__ == "__main__":
    portfolio_tickers = ['AAPL', 'SPY', 'GLD', 'TLT']
    _, returns = fetch_data(portfolio_tickers, "2016-01-01", "2026-01-01")
    
    portfolio = PortfolioModel(returns)
    engine = MonteCarloEngine(portfolio)
    paths = engine.simulate_paths(days=252, simulations=10000)