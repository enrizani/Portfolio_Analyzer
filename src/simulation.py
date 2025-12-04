"""
Modulo per la simulazione Monte Carlo di portafofli.
Genera portafogli casuali e identifica la frontiera efficiente.
"""

import pandas as pd 
import numpy as np 
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass
import json 
from datetime import datetime


# Import del modulo analysis
from analysis import (
    load_cleaned_data,
    calculate_return,
    calculate_mean_returns,
    calculate_covariance_matrix,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe,
    DEFAULT_RISK_FREE_RATE
)

# Percorsi di default 
RESULTS_PATH = Path("data/results")

# =============================================================================
# SEZIONE 1: STRUTTURA DATI PER I RISULTATI
# =============================================================================

@dataclass
class SimulationResults:
    """Contenitore per i sultati della simulazione Monte Carlo."""

    returns: np.array       # Rendimenti di ogni portafoglio
    volatilities: np.array  # Volatilita' di ogni portafoglio
    sharpe_ratios: np.array # Sharpe Ratio di ogni portafoglio
    weights: np.array       # Pesi di ogni portafoglio (n_portfolios x n_assets)
    n_portfolios: int       # Numero dei portafogli simulati
    n_assets: int           # Numero di titoli
    asset_names: List[str]  # Nomi dei titoli

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte i risultati in un DataFrame.

        Returns:
            DataFrame con tutte le simulazioni
        """

        df = pd.DataFrame({
            'Return': self.returns,
            'Volatility': self.volatilities,
            'Sharpe_Ratio': self.sharpe_ratios
        })

        # Aggiungi colonne per i pesi di ogni asset 
        for i, name in enumerate(self.asset_names):
            df[f'Weight_{name}'] = self.weights[:, i]

        return df 
    

# =============================================================================
# SEZIONE 2: GENERAZIONE PESI CASUALI
# =============================================================================

def generate_random_weights(n_assets: int, n_portfolios: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Genera pesi casuali per i portafogli usando la distribuzione Dirichlet,
    arrotondati a due decimali e rinormalizzati.
    """

    if seed is not None:
        np.random.seed(seed)

    # Dirichlet con alpha=1 genera una distribuzione uniforme sui pesi
    weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)

    # Arrotonda a due decimali
    weights = np.round(weights, 4)

    # Rinormalizza per garantire somma = 1
    weights = weights / weights.sum(axis=1, keepdims=True)

    return weights


def generate_constrained_weights(n_assets: int, n_portfolios: int,
                                 min_weight: float = 0.0, max_weight: float = 1.0,
                                 seed: Optional[int] = None) -> np.ndarray:
    """
    Genera pesi casuali con vincoli su min/max per ogni asset,
    arrotondati a due decimali e rinormalizzati.
    """

    if seed is not None:
        np.random.seed(seed)

    weights_list = []

    while len(weights_list) < n_portfolios:
        # Genera pesi casuali
        w = np.random.random(n_assets)
        w = w / w.sum()  # Normalizza a 1

        # Arrotonda
        w = np.round(w, 4)

        # Rinormalizza (dopo l'arrotondamento la somma non è più 1)
        w = w / w.sum()

        # Verifica vincoli
        if np.all(w >= min_weight) and np.all(w <= max_weight):
            weights_list.append(w)

    return np.array(weights_list)


# =============================================================================
# SEZIONE 3: SIMULAZIONE MONTE CARLO
# =============================================================================


def simulate_portfolios(returns: pd.DataFrame, n_portfolios: int=10000, risk_free_rate: float = DEFAULT_RISK_FREE_RATE, seed: Optional[int] = None) -> SimulationResults:
    """
    Esegue la simulazione Monte Carlo di portafogli casuali.
    
    Args:
        returns: DataFrame con rendimenti giornalieri
        n_portfolios: Numero di portafogli da simulare
        risk_free_rate: Tasso risk-free annualizzato
        seed: Seed per riproducibilità
    
    Returns:
        SimulationResults con tutti i risultati
    """

    n_assets = len(returns.columns)
    asset_names = returns.columns.tolist()

    # Calcola statistiche necessarie (annualizzate)
    mean_returns = calculate_mean_returns(returns, annualize=True).values
    cov_matrix = calculate_covariance_matrix(returns, annualize=True).values 

    # Genera pesi casuali
    print(f"Generazione {n_portfolios:,} portafogli casuali ...")
    all_weights = generate_random_weights(n_assets, n_portfolios, seed)

    # Array per i risultati
    port_returns = []
    port_volatilities = []
    port_sharpe = []

    # Calcola metriche per ogni portafoglio
    print("Calcolo metriche...")
    for i in range(n_portfolios):
        w = all_weights[i]
        
        # Rendimento del portafoglio
        ret = calculate_portfolio_return(w, mean_returns)
        port_returns.append(ret)
        
        # Volatilità del portafoglio
        vol = calculate_portfolio_volatility(w, cov_matrix)
        port_volatilities.append(vol)
        
        # Sharpe Ratio
        sharpe = calculate_portfolio_sharpe(ret, vol, risk_free_rate)
        port_sharpe.append(sharpe)
    
    # Converti liste in array numpy
    port_returns = np.array(port_returns)
    port_volatilities = np.array(port_volatilities)
    port_sharpe = np.array(port_sharpe)

    print(f"Simulazione completata!")

    return SimulationResults(
        returns = port_returns,
        volatilities = port_volatilities,
        sharpe_ratios = port_sharpe,
        weights = all_weights,
        n_portfolios = n_portfolios,
        n_assets = n_assets,
        asset_names = asset_names
    )


# =============================================================================
# SEZIONE 4: IDENTIFICAZIONE PORTAFOGLI OTTIMALI
# =============================================================================


def find_optimal_portfolios(results: SimulationResults) -> dict:
    """
    Identifica i portafogli ottimali dalla simulazione.
    
    Trova:
    - Max Sharpe Ratio: miglior rendimento aggiustato per il rischio
    - Min Volatility: portafoglio meno rischioso
    - Max Return: portafoglio con rendimento più alto
    
    Args:
        results: Risultati della simulazione
    
    Returns:
        Dizionario con i portafogli ottimali
    """

    # Indici dei portafogli ottimali 
    max_sharpe_idx = np.argmax(results.sharpe_ratios)
    min_vol_idx = np.argmin(results.volatilities)
    max_return_idx = np.argmax(results.returns)

    def extract_portfolio(idx: int, name: str) -> dict:
        """Estrae le informazioni di un portafoglio"""
        weights_dict = {
            asset: round(weight, 4)
            for asset, weight in zip(results.asset_names, results.weights[idx])
        }

        return {
            'name': name,
            'index': int(idx),
            'return': float(results.returns[idx]),
            'volatility': float(results.volatilities[idx]),
            'sharpe_ratio': float(results.sharpe_ratios[idx]),
            'weights': weights_dict
        }
    
    optimal = {
        'max_sharpe': extract_portfolio(max_sharpe_idx, 'Maximum Sharpe Ratio'),
        'min_volatility': extract_portfolio(min_vol_idx, 'Minimum Volatility'),
        'max_return': extract_portfolio(max_return_idx, 'Maximum Return')
    }

    return optimal 


# =============================================================================
# SEZIONE 5: FRONTIERA EFFICIENTE
# =============================================================================


def extract_efficient_frontier(results: SimulationResults, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estrae i punti della frontiera efficiente dai risultati simulati.
    
    Per ogni livello di volatilità, seleziona il portafoglio con
    il rendimento più alto.
    
    Args:
        results: Risultati della simulazione
        n_points: Numero di punti da estrarre sulla frontiera
    
    Returns:
        Tuple: (volatilità, rendimenti, sharpe_ratios) sulla frontiera
    """

    # Dividi il range di volatilita' in bucket 
    vol_min = results.volatilities.min()
    vol_max = results.volatilities.max()
    vol_bins = np.linspace(vol_min, vol_max, n_points + 1)

    frontier_vols = []
    frontier_rets = []
    frontier_sharpe = []

    for i in range(len(vol_bins) - 1):
        # Trova portafogli nel bucket di volatilita'
        mask = ((results.volatilities >= vol_bins[i]) & (results.volatilities < vol_bins[i + 1]))

        if mask.any():
            # Prendi quello con rendimento massimo nel bucket
            bucket_indices = np.where(mask)[0]
            bucket_returns = results.returns[mask]
            best_in_bucket = bucket_indices[np.argmax(bucket_returns)]

            frontier_vols.append(results.volatilities[best_in_bucket])
            frontier_rets.append(results.returns[best_in_bucket])
            frontier_sharpe.append(results.sharpe_ratios[best_in_bucket])

    return np.array(frontier_vols), np.array(frontier_rets), np.array(frontier_sharpe)


# =============================================================================
# SEZIONE 6: SALVATAGGIO E REPORT
# =============================================================================


def save_simulation_results(results: SimulationResults, optimal: dict, filename: str = "simulation_results.csv") -> Path:
    """
    Salva i risultati della simulazione in CSV.
    
    Args:
        results: Risultati della simulazione
        optimal: Portafogli ottimali
        filename: Nome del file
    
    Returns:
        Path del file salvato
    """

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Salva tutti i portafogli
    df = results.to_dataframe()
    filepath = RESULTS_PATH / filename
    df.to_csv(filepath, index=False)
    print(f"Risultati salvati in: {filepath}")
    
    return filepath


def save_optimal_portfolios(optimal: dict, filename: str = "optimal_portfolios.json") -> Path:
    """
    Salva i portafogli ottimali in JSON.
    
    Args:
        optimal: Dizionario con portafogli ottimali
        filename: Nome del file
    
    Returns:
        Path del file salvato
    """

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_PATH / filename
    
    # Aggiungi metadata
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'portfolios': optimal
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Portafogli ottimali salvati in: {filepath}")
    
    return filepath


def print_simulation_summary(results: SimulationResults, optimal: dict) -> None:
    """
    Stampa un riepilogo della simulazione.
    
    Args:
        results: Risultati della simulazione
        optimal: Portafogli ottimali
    """

    print("\n" + "=" * 70)
    print("RIEPILOGO SIMULAZIONE MONTE CARLO")
    print("=" * 70)
    
    print(f"\nPortafogli simulati: {results.n_portfolios:,}")
    print(f"Numero titoli: {results.n_assets}")
    print(f"Titoli: {', '.join(results.asset_names)}")
    
    # Statistiche generali
    print("\n" + "-" * 70)
    print("DISTRIBUZIONE DEI PORTAFOGLI SIMULATI")
    print("-" * 70)
    print(f"Rendimento: min={results.returns.min():.2%}, max={results.returns.max():.2%}, media={results.returns.mean():.2%}")
    print(f"Volatilità: min={results.volatilities.min():.2%}, max={results.volatilities.max():.2%}, media={results.volatilities.mean():.2%}")
    print(f"Sharpe Ratio: min={results.sharpe_ratios.min():.3f}, max={results.sharpe_ratios.max():.3f}, media={results.sharpe_ratios.mean():.3f}")
    
    # Portafogli ottimali
    for key, portfolio in optimal.items():
        print("\n" + "-" * 70)
        print(f"PORTAFOGLIO: {portfolio['name'].upper()}")
        print("-" * 70)
        print(f"Rendimento atteso: {portfolio['return']:.2%}")
        print(f"Volatilità: {portfolio['volatility']:.2%}")
        print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
        print("\nAllocazione:")
        
        # Ordina pesi per valore decrescente
        sorted_weights = sorted(
            portfolio['weights'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for asset, weight in sorted_weights:
            if weight > 0.001:  # Mostra solo pesi > 0.1%
                print(f"  {asset}: {weight:.2%}")
    
    print("\n" + "=" * 70)


# =============================================================================
# SEZIONE 7: FUNZIONE PRINCIPALE
# =============================================================================


def run_simulation(n_portfolios: int = 10000, risk_free_rate: float = DEFAULT_RISK_FREE_RATE, seed: Optional[int] = 42, save_results: bool = True) -> Tuple[SimulationResults, dict]:
    """
    Esegue l'intera simulazione Monte Carlo.
    
    Args:
        n_portfolios: Numero di portafogli da simulare
        risk_free_rate: Tasso risk-free annualizzato
        seed: Seed per riproducibilità
        save_results: Se True, salva i risultati su file
    
    Returns:
        Tuple: (SimulationResults, dizionario portafogli ottimali)
    """

    print("\n" + "=" * 70)
    print("SIMULAZIONE MONTE CARLO - PORTAFOGLI")
    print("=" * 70)
    
    # 1. Carica dati e calcola rendimenti
    print("\n[1/4] Caricamento dati...")
    prices = load_cleaned_data()
    returns = calculate_return(prices, method='simple')
    
    # 2. Esegui simulazione
    print(f"\n[2/4] Simulazione {n_portfolios:,} portafogli...")
    results = simulate_portfolios(
        returns, 
        n_portfolios=n_portfolios,
        risk_free_rate=risk_free_rate,
        seed=seed
    )
    
    # 3. Trova portafogli ottimali
    print("\n[3/4] Identificazione portafogli ottimali...")
    optimal = find_optimal_portfolios(results)
    
    # 4. Salva risultati
    if save_results:
        print("\n[4/4] Salvataggio risultati...")
        save_simulation_results(results, optimal)
        save_optimal_portfolios(optimal)
    
    # Stampa riepilogo
    print_simulation_summary(results, optimal)
    
    return results, optimal


# =============================================================================
# ESEMPIO DI UTILIZZO
# =============================================================================

if __name__ == "__main__":
    # Esegui simulazione con 10.000 portafogli
    results, optimal = run_simulation(
        n_portfolios=10000,
        risk_free_rate=0.02,
        seed=42,
        save_results=True
    )
    
    # Estrai frontiera efficiente
    frontier_vol, frontier_ret, frontier_sharpe = extract_efficient_frontier(results)
    print(f"\nPunti sulla frontiera efficiente: {len(frontier_vol)}")

