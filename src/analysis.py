"""
Modulo per l'analisi finanziaria del portafoglio.
Calcola rendimenti, volatilita', covarianza e Sharpe Ratio.
"""

import pandas as pd 
import numpy as np 
from pathlib import Path
from typing import Tuple, Optional, Literal 

# Percorsi di default 
CLEANED_DATA_PATH = Path("data/cleaned")

# Costanti 
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02 # 2% annuo 

# ================================================================================================
# SEZIONE 1: CARICAMENTO DATI
# ================================================================================================


def load_cleaned_data(filename: str = "sp500_cleaned_prices.csv") -> pd.DataFrame:
    """
    Carica i dati puliti dal file CSV.

    Args: 
        filename: Nome del file in data/cleaned

    Returns: 
        DataFrame con prezzi (Date come indice, Ticker come colonne)
    """

    filepath =  CLEANED_DATA_PATH / filename 

    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    print(f"Dati caricati: {df.shape[0]} giorni x {df.shape[1]} ticekr")
    print(f"Periodo: {df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')}")

    return df


# =============================================================================
# SEZIONE 2: CALCOLO RENDIMENTI
# =============================================================================

def calculate_return(prices: pd.DataFrame,
                     method: Literal['simple', 'log'] = 'simple') -> pd.DataFrame:
    """
    Calcola i rendimenti dai prezzi.

    Args: 
        prices: DataFrame con prezzi (Date x Ticker)
        methos: 'simple' per rendimenti percentuali, 'lo' per logaritmici

    Returns:
        DataFrame con rendimenti giornalieri
    """

    if method == 'simple':
        # Rendimento semplice: (P_t - P_{t-1} / P_{t-1})
        returns = prices.pct_change()
    
    elif method == 'log':
        # Rendimento logaritimico: ln(p_t / P_{t-1})
        reurns = np.log(prices / prices.shift(1))

    else: 
        raise ValueError(f"Metodo non riconosciuto: {method}. Usa 'simole' o 'log'")
    
    # Rimuovi la prima riga (NaN)
    returns = returns.dropna()

    return returns 


def calculate_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i rendimenti cumulativi.

    Args: 
        returns: DataFrame con rendimenti giornalieri

    Returns:
        DataFrame con rendimenti cumulativi
    """

    cumulative = (1 + returns).cumprod() - 1 

    return cumulative


# =============================================================================
# SEZIONE 3: STATISTICHE DESCRITTIVE
# =============================================================================


def calculate_mean_returns(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Calcola il rendimento medio per ogni ticker.

    Args: 
        returns: DataFrame con rendimenti giornalieri
        annualize: Se True, annualizza i rendimenti

    Returns:
        Series con rendimento medio per ogni ticker 
    """

    mean_returns = returns.mean()

    if annualize:
        mean_returns = mean_returns * TRADING_DAYS_PER_YEAR

    return mean_returns


def calculate_volatility(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Calcola la volatilita' (deviazione standard) per ogni ticker.

    Args: 
        returns: DataFrame con rendimenti giornalieri
        annualize: Se True, annualizza la volatilita'

    Returns:
        Series con volatilita' per ogni ticker
    """

    volatility = returns.std()

    if annualize:
        volatility = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    return volatility


def calculate_covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Calcola la matrice covarianza.

    Args: 
        returns: DataFrame con rendimenti giornalieri
        annualize: Se True, annualizza la covarianza

    Returns:
        Matrice di covarianza (DataFrame)
    """

    cov_matrix = returns.cov()

    if annualize:
        cov_matrix = cov_matrix * TRADING_DAYS_PER_YEAR

    return cov_matrix


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la matrice di correlazione.
    
    Args:
        returns: DataFrame con rendimenti giornalieri
    
    Returns:
        Matrice di correlazione (DataFrame)
    """

    return returns.corr()


# =============================================================================
# SEZIONE 4: SHARPE RATIO
# =============================================================================


def calculate_sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> pd.Series:
    """
    Calcola lo Sharpe Ratio per ogni ticker.

    Formula: SR = (Rendimento_annuo - Risk_Fress_Rate) / Volatilita_annua

    Args: 
        returns: DataFrame con renidmenti giornalieri
        risk_free_rate: Tasso risk-free annualizzato (default 2%)

    Returns:
        Series con Sharpe Ratio per ogni ticker
    """

    mean_returns = calculate_mean_returns(returns, annualize=True)

    volatility = calculate_volatility(returns, annualize=True)

    sharpe_ratio = (mean_returns - risk_free_rate) /volatility

    return sharpe_ratio


# =============================================================================
# SEZIONE 5: METRICHE DI PORTAFOGLIO
# =============================================================================


def calculate_portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Calcolata il rendimento atteso di un portafoglio.

    Formula: R_p = Σ(w_i x R_i) = w^T x R

    Args:
        weights: Array dei pesi (devono sommare a 1)
        mean_returns: Array dei rendimenti medi annualizzati
    
    Returns:
        Rendimento atteso del portafoglio
    """

    return np.dot(weights, mean_returns)


def calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calcola la volatilità di un portafoglio.
    
    Formula: sigma_p = √(w^T x Σ x w)
    
    Args:
        weights: Array dei pesi
        cov_matrix: Matrice di covarianza annualizzata
    
    Returns:
        Volatilità del portafoglio
    """

    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    return np.sqrt(portfolio_variance)


def calculate_portfolio_sharpe(portfolio_return: float, portfolio_volatility: float, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """
    Calcola lo Sharpe Ratio di un portafoglio.
    
    Args:
        portfolio_return: Rendimento atteso annualizzato
        portfolio_volatility: Volatilità annualizzata
        risk_free_rate: Tasso risk-free annualizzato
    
    Returns:
        Sharpe Ratio del portafoglio
    """
    if portfolio_volatility == 0:
        return 0.0
    return (portfolio_return - risk_free_rate) / portfolio_volatility


def calculate_portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> dict:
    """
    Calcola tutte le metriche di un portafoglio dato un set di pesi.
    
    Args:
        weights: Array dei pesi (devono sommare a 1)
        returns: DataFrame con rendimenti giornalieri
        risk_free_rate: Tasso risk-free annualizzato
    
    Returns:
        Dizionario con tutte le metriche del portafoglio
    """

    # Calcola statistiche base
    mean_returns = calculate_mean_returns(returns, annualize=True).values
    cov_matrix = calculate_covariance_matrix(returns, annualize=True).values
    
    # Calcola metriche portafoglio
    port_return = calculate_portfolio_return(weights, mean_returns)
    port_volatility = calculate_portfolio_volatility(weights, cov_matrix)
    port_sharpe = calculate_portfolio_sharpe(port_return, port_volatility, risk_free_rate)
    
    return {
        'return': port_return,
        'volatility': port_volatility,
        'sharpe_ratio': port_sharpe,
        'weights': weights
    }



# =============================================================================
# SEZIONE 6: REPORT E RIEPILOGO
# =============================================================================



def get_individual_stats(returns: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> pd.DataFrame:
    """
    Genera statistiche individuali per ogni ticker.
    
    Args:
        returns: DataFrame con rendimenti giornalieri
        risk_free_rate: Tasso risk-free annualizzato
    
    Returns:
        DataFrame con statistiche per ogni ticker
    """

    stats = pd.DataFrame({
        'Rendimento_Annuo': calculate_mean_returns(returns, annualize=True),
        'Volatilità_Annua': calculate_volatility(returns, annualize=True),
        'Sharpe_Ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'Min': returns.min(),
        'Max': returns.max(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })
    
    # Ordina per Sharpe Ratio decrescente
    stats = stats.sort_values('Sharpe_Ratio', ascending=False)
    
    return stats


def print_analysis_summary(returns: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> None:
    """
    Stampa un riepilogo completo dell'analisi.
    
    Args:
        returns: DataFrame con rendimenti giornalieri
        risk_free_rate: Tasso risk-free annualizzato
    """

    print("\n" + "=" * 60)
    print("RIEPILOGO ANALISI FINANZIARIA")
    print("=" * 60)
    
    # Info generali
    print(f"\nPeriodo analizzato: {returns.index.min().strftime('%Y-%m-%d')} -> {returns.index.max().strftime('%Y-%m-%d')}")
    print(f"Giorni di trading: {len(returns)}")
    print(f"Numero titoli: {len(returns.columns)}")
    print(f"Tasso risk-free: {risk_free_rate:.2%}")
    
    # Statistiche individuali
    stats = get_individual_stats(returns, risk_free_rate)
    
    print("\n" + "-" * 60)
    print("STATISTICHE INDIVIDUALI (ordinate per Sharpe Ratio)")
    print("-" * 60)
    
    # Formatta l'output
    stats_display = stats.copy()
    stats_display['Rendimento_Annuo'] = stats_display['Rendimento_Annuo'].apply(lambda x: f"{x:.2%}")
    stats_display['Volatilità_Annua'] = stats_display['Volatilità_Annua'].apply(lambda x: f"{x:.2%}")
    stats_display['Sharpe_Ratio'] = stats_display['Sharpe_Ratio'].apply(lambda x: f"{x:.3f}")
    stats_display['Min'] = stats_display['Min'].apply(lambda x: f"{x:.2%}")
    stats_display['Max'] = stats_display['Max'].apply(lambda x: f"{x:.2%}")
    stats_display['Skewness'] = stats_display['Skewness'].apply(lambda x: f"{x:.3f}")
    stats_display['Kurtosis'] = stats_display['Kurtosis'].apply(lambda x: f"{x:.3f}")
    
    print(stats_display.to_string())
    
    # Top e Bottom performer
    print("\n" + "-" * 70)
    print("TOP 3 PERFORMER (Sharpe Ratio)")
    print("-" * 60)
    for i, (ticker, row) in enumerate(stats.head(3).iterrows(), 1):
        print(f"{i}. {ticker}: Sharpe={row['Sharpe_Ratio']:.3f}, Rend={row['Rendimento_Annuo']:.2%}, Vol={row['Volatilità_Annua']:.2%}")
    
    print("\n" + "-" * 70)
    print("BOTTOM 3 PERFORMER (Sharpe Ratio)")
    print("-" * 70)
    for i, (ticker, row) in enumerate(stats.tail(3).iterrows(), 1):
        print(f"{i}. {ticker}: Sharpe={row['Sharpe_Ratio']:.3f}, Rend={row['Rendimento_Annuo']:.2%}, Vol={row['Volatilità_Annua']:.2%}")
    
    # Correlazione media
    corr_matrix = calculate_correlation_matrix(returns)

    # Escludi la diagonale (correlazione con se stessi)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_correlation = corr_matrix.where(mask).stack().mean()
    
    print("\n" + "-" * 70)
    print("CORRELAZIONE")
    print("-" * 70)
    print(f"Correlazione media tra titoli: {avg_correlation:.3f}")
    
    print("\n" + "=" * 70)


def run_analysis(filename: str = "sp500_cleaned_prices.csv", risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Esegue l'analisi completa sui dati.
    
    Args:
        filename: Nome del file con i dati puliti
        risk_free_rate: Tasso risk-free annualizzato
    
    Returns:
        Tuple: (rendimenti, matrice_covarianza, statistiche_individuali)
    """

    # Carica dati
    prices = load_cleaned_data(filename)
    
    # Calcola rendimenti
    returns = calculate_return(prices, method='simple')
    
    # Calcola covarianza
    cov_matrix = calculate_covariance_matrix(returns, annualize=True)
    
    # Statistiche individuali
    stats = get_individual_stats(returns, risk_free_rate)
    
    # Stampa riepilogo
    print_analysis_summary(returns, risk_free_rate)
    
    return returns, cov_matrix, stats


# =============================================================================
# ESEMPIO DI UTILIZZO
# =============================================================================

if __name__ == "__main__":
    # Esegui analisi completa
    returns, cov_matrix, stats = run_analysis()
    
    # Esempio: calcola metriche per un portafoglio equiponderato
    n_assets = len(returns.columns)
    equal_weights = np.ones(n_assets) / n_assets
    
    print("\n" + "=" * 70)
    print("PORTAFOGLIO EQUIPONDERATO")
    print("=" * 70)
    
    portfolio = calculate_portfolio_metrics(equal_weights, returns)
    print(f"Rendimento atteso: {portfolio['return']:.2%}")
    print(f"Volatilità: {portfolio['volatility']:.2%}")
    print(f"Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")