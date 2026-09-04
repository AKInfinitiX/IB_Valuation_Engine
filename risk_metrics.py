import numpy as np
from data_pipeline import fetch_data
from portfolio import PortfolioModel
from simulation import MonteCarloEngine

class RiskMetricsEngine:
    def __init__(self, price_paths, initial_portfolio_value=1_000_000, weights=None):
        self.paths = price_paths  # Shape: (253, 10000, 4)
        self.portfolio_value = initial_portfolio_value
        
        num_assets = price_paths.shape[2]
        if weights is None:
            self.weights = np.ones(num_assets) / num_assets
        else:
            self.weights = np.array(weights)

    def calculate_portfolio_terminal_values(self):
        # Normalized terminal asset values at day 252
        terminal_relatives = self.paths[-1, :, :]  # Shape: (10000, 4)
        
        # Portfolio value across all 10,000 paths
        terminal_portfolio_values = self.portfolio_value * (terminal_relatives @ self.weights)
        pnl = terminal_portfolio_values - self.portfolio_value
        return terminal_portfolio_values, pnl

    def calculate_var_cvar(self, alpha=0.95):
        _, pnl = self.calculate_portfolio_terminal_values()
        
        # Express losses as positive quantities: Loss = -PnL
        losses = -pnl
        
        # Monte Carlo VaR: 95th or 99th percentile of loss distribution
        var_mc = np.percentile(losses, alpha * 100)
        
        # Monte Carlo CVaR (Expected Shortfall): Average loss beyond VaR
        cvar_mc = losses[losses >= var_mc].mean()
        
        return var_mc, cvar_mc

if __name__ == "__main__":
    portfolio_tickers = ['AAPL', 'SPY', 'GLD', 'TLT']
    _, returns = fetch_data(portfolio_tickers, "2016-01-01", "2026-01-01")
    
    portfolio = PortfolioModel(returns)
    engine = MonteCarloEngine(portfolio)
    paths = engine.simulate_paths(days=252, simulations=10000)
    
    risk_engine = RiskMetricsEngine(paths, initial_portfolio_value=1_000_000)
    
    var_95, cvar_95 = risk_engine.calculate_var_cvar(alpha=0.95)
    var_99, cvar_99 = risk_engine.calculate_var_cvar(alpha=0.99)
    
    print("\n" + "="*45)
    print("      QUANTITATIVE RISK ENGINE METRICS       ")
    print("="*45)
    print(f"Initial Portfolio Value: $1,000,000")
    print(f"95% 1-Year VaR         : ${var_95:,.2f}")
    print(f"95% 1-Year CVaR (ES)   : ${cvar_95:,.2f}")
    print(f"99% 1-Year VaR         : ${var_99:,.2f}")
    print(f"99% 1-Year CVaR (ES)   : ${cvar_99:,.2f}")
    print("="*45)