import numpy as np
import pandas as pd
from analysis import (
    load_cleaned_data,
    calculate_return,
    calculate_mean_returns,
    calculate_covariance_matrix
)

# Carica dati
prices = load_cleaned_data()
returns = calculate_return(prices, method='simple')

# Calcola statistiche
mean_returns = calculate_mean_returns(returns, annualize=True)
cov_matrix = calculate_covariance_matrix(returns, annualize=True)

print("=== MEAN RETURNS ===")
print(f"Tipo: {type(mean_returns)}")
print(f"Valori:\n{mean_returns}")

print("\n=== COV MATRIX ===")
print(f"Tipo: {type(cov_matrix)}")
print(f"Shape: {cov_matrix.shape}")
print(f"Valori:\n{cov_matrix}")

# Prova un calcolo manuale
mean_returns_arr = mean_returns.values
cov_matrix_arr = cov_matrix.values

print("\n=== ARRAY VALUES ===")
print(f"mean_returns_arr: {mean_returns_arr}")
print(f"cov_matrix_arr shape: {cov_matrix_arr.shape}")

# Test con pesi casuali
weights = np.array([0.25, 0.25, 0.25, 0.25])
port_return = np.dot(weights, mean_returns_arr)
port_var = np.dot(weights.T, np.dot(cov_matrix_arr, weights))
port_vol = np.sqrt(port_var)

print(f"\n=== TEST PORTAFOGLIO EQUIPONDERATO ===")
print(f"Return: {port_return}")
print(f"Volatility: {port_vol}")



# Debug 2 

print("=" *50)
print("=" *50)

import numpy as np
from analysis import (
    load_cleaned_data,
    calculate_return,
    calculate_mean_returns,
    calculate_covariance_matrix,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe
)
from simulation import generate_random_weights

# Carica dati
prices = load_cleaned_data()
returns = calculate_return(prices, method='simple')

# Calcola statistiche
mean_returns = calculate_mean_returns(returns, annualize=True).values
cov_matrix = calculate_covariance_matrix(returns, annualize=True).values

# Genera pesi
n_portfolios = 5
n_assets = 4
all_weights = generate_random_weights(n_assets, n_portfolios, seed=42)

print("=== PESI GENERATI ===")
print(all_weights)
print(f"Shape: {all_weights.shape}")

# Test loop manuale
print("\n=== TEST LOOP ===")
for i in range(n_portfolios):
    w = all_weights[i]
    print(f"\nPortafoglio {i}:")
    print(f"  Pesi: {w}")
    print(f"  Somma pesi: {w.sum()}")
    
    ret = calculate_portfolio_return(w, mean_returns)
    vol = calculate_portfolio_volatility(w, cov_matrix)
    sharpe = calculate_portfolio_sharpe(ret, vol, 0.02)
    
    print(f"  Return: {ret}")
    print(f"  Volatility: {vol}")
    print(f"  Sharpe: {sharpe}")