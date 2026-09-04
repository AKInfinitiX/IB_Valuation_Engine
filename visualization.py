import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import fetch_data
from portfolio import PortfolioModel
from simulation import MonteCarloEngine
from risk_metrics import RiskMetricsEngine

def plot_risk_profile(risk_engine, num_sample_paths=100):
    paths = risk_engine.paths
    terminal_vals, pnl = risk_engine.calculate_portfolio_terminal_values()
    var_95, cvar_95 = risk_engine.calculate_var_cvar(alpha=0.95)
    var_99, cvar_99 = risk_engine.calculate_var_cvar(alpha=0.99)
    
    # Portfolio paths over time: (253, 10000)
    portfolio_paths = risk_engine.portfolio_value * (paths @ risk_engine.weights)
    
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Monte Carlo Path Trajectories
    plt.subplot(1, 2, 1)
    plt.plot(portfolio_paths[:, :num_sample_paths], color='steelblue', alpha=0.15)
    plt.plot(np.median(portfolio_paths, axis=1), color='navy', linewidth=2, label='Median Path')
    plt.axhline(risk_engine.portfolio_value, color='black', linestyle='--', label='Initial Value ($1M)')
    plt.title('Monte Carlo Simulation: 100 Correlated Paths (1-Year)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Terminal PnL & Risk Cutoffs
    plt.subplot(1, 2, 2)
    # Losses are -PnL
    losses = -pnl
    plt.hist(losses, bins=60, color='lightgray', edgecolor='black', alpha=0.7, density=True)
    plt.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: ${var_95:,.0f}')
    plt.axvline(cvar_95, color='darkorange', linewidth=2, label=f'95% CVaR: ${cvar_95:,.0f}')
    plt.axvline(var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR: ${var_99:,.0f}')
    plt.axvline(cvar_99, color='darkred', linewidth=2, label=f'99% CVaR: ${cvar_99:,.0f}')
    
    plt.title('Portfolio 1-Year Loss Distribution & Tail Risk')
    plt.xlabel('Loss ($)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("risk_engine_results.png", dpi=300)
    print("Plot generated and saved as 'risk_engine_results.png'. Displaying window...")
    plt.show()

if __name__ == "__main__":
    portfolio_tickers = ['AAPL', 'SPY', 'GLD', 'TLT']
    _, returns = fetch_data(portfolio_tickers, "2016-01-01", "2026-01-01")
    
    portfolio = PortfolioModel(returns)
    engine = MonteCarloEngine(portfolio)
    paths = engine.simulate_paths(days=252, simulations=10000)
    
    risk_engine = RiskMetricsEngine(paths, initial_portfolio_value=1_000_000)
    plot_risk_profile(risk_engine)