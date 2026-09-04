import numpy as np
import pandas as pd
from arch import arch_model
from data_pipeline import fetch_data

class PortfolioModel:
    def __init__(self, log_returns):
        self.returns = log_returns
        self.tickers = list(log_returns.columns)
        self.mean_returns = log_returns.mean().values
        self.cov_matrix = log_returns.cov().values
        self.corr_matrix = log_returns.corr().values
        self.garch_params = {}

    def get_cholesky_decomposition(self):
        """
        Decomposes the correlation matrix (R = L * L^T) to separate 
        asset correlation from individual asset volatilities.
        """
        try:
            return np.linalg.cholesky(self.corr_matrix)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
            eigvals = np.maximum(eigvals, 1e-8)
            clean_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return np.linalg.cholesky(clean_corr)

    def fit_garch_models(self):
        """
        Fits a GARCH(1,1) model for each asset:
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        print("Fitting GARCH(1,1) models to historical returns...")
        for ticker in self.tickers:
            series = self.returns[ticker] * 100.0  # Rescale for optimizer convergence
            am = arch_model(series, vol='Garch', p=1, q=1, rescale=False)
            res = am.fit(disp='off')
            
            # Extract parameters and rescale back
            omega = res.params['omega'] / 10000.0
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            current_vol = np.sqrt(res.conditional_volatility.iloc[-1]**2 / 10000.0)
            
            self.garch_params[ticker] = {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'last_vol': current_vol
            }
            print(f"  -> {ticker}: alpha={alpha:.4f}, beta={beta:.4f} (Persistence={alpha+beta:.4f})")
            
        return self.garch_params

if __name__ == "__main__":
    portfolio_tickers = ['AAPL', 'SPY', 'GLD', 'TLT']
    _, returns = fetch_data(portfolio_tickers, "2016-01-01", "2026-01-01")
    
    model = PortfolioModel(returns)
    L = model.get_cholesky_decomposition()
    garch_params = model.fit_garch_models()