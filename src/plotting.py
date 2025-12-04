"""
Modulo per la visualizzazione grafica dei risultati.
Crea grafici della frontiera efficiente e altre analisi visive.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Tuple, List
import seaborn as sns

# Import dei moduli precedenti 
from analysis import (
    load_cleaned_data,
    calculate_return,
    calculate_cumulative_returns, 
    calculate_mean_returns,
    calculate_volatility,
    calculate_correlation_matrix,
    calculate_sharpe_ratio
)

from simulation import (
    SimulationResults,
    extract_efficient_frontier
)

# Configurazione stile grafici 
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Percorsi
PLOTS_PATH = Path("data/plots")


# =============================================================================
# SEZIONE 1: CONFIGURAZIONE GRAFICI
# =============================================================================

def setup_plot_style():
    """Configura lo stile generale dei grafici"""
    plt.rcParams['figure.figsize'] = (12,8) 
    plt.rcParams['font.size'] = 10 
    plt.rcParams['axes.labelsize'] = 12 
    plt.rcParams['axes.titlesize'] = 14 
    plt.rcParams['xtick.labelsize'] = 10 
    plt.rcParams['ytick.labelsize'] = 10 
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


 # ============================================================================
# SEZIONE 2: GRAFICO FRONTIERA EFFICIENTE
# =============================================================================


def plot_efficient_frontier(results: SimulationResults, optimal: dict, save: bool = True, filename: str = "efficient_frontier.png") -> plt.Figure:
    """
    Crea il grafico della frontire efficiente.

    Args: 
        results: Risultati della simulazione Monte Carlo
        optimal: Dizionario con portafogli ottimali
        save: Se True, salva il grafico
        filename: Nome del file

    Returns:
        Figure di matplotlib  
    """

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14,9))

    # 1. Scatter plot di tutti i portafogli (colorati pre Sharpe Ratio)
    scatter = ax.scatter(
        results.volatilities,
        results.returns,
        c=results.sharpe_ratios,
        cmap='viridis',
        alpha=0.6,
        s=20,
        edgecolors='none'
    )
    
    # Colorbar per lo Sharpe Ratio
    cbar = plt.colorbar(scatter,ax=ax)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)

    # 2. Linea della frontiera efficiente 
    frontier_vol, frontier_ret, _ = extract_efficient_frontier(results, n_points=100)
    ax.plot(
        frontier_vol,
        frontier_ret,
        'r--',
        linewidth=3,
        label='Frontiera Efficiente'
    )

    # 3. Evidenzia i portafogli ottimali
    # Max Sharpe Ratio
    max_sharpe = optimal['max_sharpe']
    ax.scatter(
        max_sharpe['volatility'],
        max_sharpe['return'],
        marker='*',
        s=800,
        c='gold',
        edgecolors='black',
        linewidth=2,
        label=f"Max Sharpe Ratio ({max_sharpe['sharpe_ratio']:.3f})",
        zorder=5
    )

    # Min VOlatility
    min_vol = optimal['min_volatility']
    ax.scatter(
        min_vol['volatility'],
        min_vol['return'],
        marker='D',
        s=400,
        c='lime',
        edgecolors='black',
        linewidth=2,
        label=f"Min Volatilità ({min_vol['volatility']:.2%})",
        zorder=5
    )

    # Max Return
    max_ret = optimal['max_return']
    ax.scatter(
        max_ret['volatility'],
        max_ret['return'],
        marker='^',
        s=400,
        c='red',
        edgecolors='black',
        linewidth=2,
        label=f"Max Rendimento ({max_ret['return']:.2%})",
        zorder=5
    )

    # 4. Etichette e formattazione
    ax.set_xlabel('Volatilità Annualizzata', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rendimento Atteso Annualizzato', fontsize=14, fontweight='bold')
    ax.set_title('Frontiera Efficiente - Simulazione Monte Carlo', fontsize=16, fontweight='bold', pad=20)

    # Formatta assi come percentuali
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))

    # Griglia e legenda
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()

    # Salva
    if save:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = PLOTS_PATH / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {filepath}")

    return fig

# =============================================================================
# SEZIONE 3: RENDIMENTI CUMULATIVI
# =============================================================================


def plot_cumulative_returns(returns: pd.DataFrame, save: bool=True, filename: str = "cumulative_returns.png") -> plt.Figure:
    """
    Grafico dei rendimenti cumulativei nel tempo.

    Args: 
        returns: DataFrame con rendimenti giornalieri
        save: Se True, salva il grafico
        filename: Nome del file

    Returns: 
        Figure di matplotlib
    """

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14,8))

    # Calcola rendimenti cumulativi
    cumulative = calculate_cumulative_returns(returns)

    # Plot per ogni asset 
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=col, linewidth=2)

    # Formattazione 
    ax.set_xlabel('Data', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rendimento Cumulativo', fontsize=12, fontweight='bold')
    ax.set_title('Evoluzione dei Rendimenti Cumulativi', fontsize=14, fontweight='bold', pad=20)

    # Formatta asse Y come percentuale
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    if save:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = PLOTS_PATH / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {filepath}")

    return fig

# =============================================================================
# SEZIONE 4: MATRICE DI CORRELAZIONE
# =============================================================================

def plot_correlation_matrix(returns: pd.DataFrame, save: bool=True, filename: str = "correlation_matrix.png") -> plt.Figure:
    """
    heatmap della matrice di correlazione.

    Args:
        returns: DataFrame con rendimenti gionralieri
        save: Se True, salva il grafico
        filename: Nomde del file

    Returns:
        Figure di matplotlib
    """

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10,8))

    # Calcola la matrice di correlazione
    corr_matrix = calculate_correlation_matrix(returns)

    # Crea heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlazione'},
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title('Matrice di Correlazione tra Titoli', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = PLOTS_PATH / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {filepath}")

    return fig 

# =============================================================================
# SEZIONE 5: DISTRIBUZIONE RENDIMENTI
# =============================================================================

def plot_returns_distribution(returns: pd.DataFrame, save: bool = True, filename: str = "returns_distribution.png") -> plt.Figure:
    """
    Istogrammi della distribuzione dei rendimenti per ogni asset.
    
    Args:
        returns: DataFrame con rendimenti giornalieri
        save: Se True, salva il grafico
        filename: Nome del file
    
    Returns:
        Figure di matplotlib
    """
    setup_plot_style()
    n_assets = len(returns.columns)
    
    # Calcola dimensioni griglia (es. 2x2 per 4 asset, 2x3 per 5-6 asset, ecc.)
    n_cols = 2
    n_rows = (n_assets + n_cols - 1) // n_cols  # Arrotonda per eccesso
    
    # Crea subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(returns.columns):
        ax = axes[i]
        
        # Istogramma
        ax.hist(returns[col], bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Aggiungi curva normale per confronto
        mu = returns[col].mean()
        sigma = returns[col].std()
        x = np.linspace(returns[col].min(), returns[col].max(), 100)
        ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2),
                'r-', linewidth=2, label='Normale')
        
        # Formattazione
        ax.set_xlabel('Rendimento Giornaliero', fontsize=10)
        ax.set_ylabel('Densità', fontsize=10)
        ax.set_title(f'{col} - Distribuzione Rendimenti', fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Nascondi subplot vuoti se ce ne sono
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Distribuzione dei Rendimenti Giornalieri', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = PLOTS_PATH / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {filepath}")
    
    return fig
    
# =============================================================================
# SEZIONE 6: ALLOCAZIONE PORTAFOGLI OTTIMALI
# =============================================================================

def plot_optimal_allocations(optimal: dict, save: bool=True, filename: str = "optimal_allocations.png") -> plt.Figure:
    """
    bar cahrt delle allocazioni dei portafogli ottimali.

    Args: 
        optimal: Dizionario con portafogli ottimali
        save: Se True, salva il grafico
        filename: Nome del file
    
    Returns: 
        Figure di matplotlib
    """

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    portfolios = ['max_sharpe', 'min_volatility', 'max_return']
    titles = ['Max Sharpe Ration', 'Min Volatility', 'Max Return']

    for idx, (portfolio_key, title) in enumerate(zip(portfolios, titles)):
        ax = axes[idx]
        portfolio = optimal[portfolio_key]

        # Estrai pesi
        weights = portfolio['weights']
        assets = list(weights.keys())
        values = list(weights.values())

        # Ordina per peso decrescente
        sorted_data = sorted(zip(assets, values), key=lambda x: x[1], reverse=True)
        assets, values = zip(*sorted_data)

        # Bar chart
        bars = ax.bar(assets, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(assets))), edgecolor='black', linewidth=1.5)

        # Aggiungi percentuali sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

        # Formattazione
        ax.set_ylabel('Peso nel Portafoglio', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nSharpe: {portfolio["sharpe_ratio"]:.3f}', fontweight='bold', fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')

    
    fig.suptitle('Allocazione dei Portafogli Ottimali', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = PLOTS_PATH / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {filepath}")
    
    return fig

# =============================================================================
# SEZIONE 7: RISK-RETURN ASSET INDIVIDUALI
# =============================================================================

def plot_individual_assets(returns: pd.DataFrame, risk_free_rate: float = 0.02, save: bool=True, filename: str = "individual_assets.png") -> plt.Figure:
    """
    Scatter plot rischio/rendimento per ogni asset individuale.

    Args:
        returns: DataFrame con rendimenti giornalieri
        risk_frre_rate: tasso risk-free annualizzato
        save: Se True, salva il grafico 
        filename: Nome del file

    Returns:
        Figure di matplotlib
    """

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12,8))

    # Calcola metriche per ogni asset
    mean_returns = calculate_mean_returns(returns, annualize=True)
    volatility = calculate_volatility(returns, annualize=True)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)

    # Scatter plot
    for asset in returns.columns:
        ax.scatter(
            volatility[asset],
            mean_returns[asset],
            s=300,
            alpha=0.7,
            edgecolors='black',
            linewidth=2,
            label=f'{asset} (SR: {sharpe[asset]:.2f})'
        )

        # Aggiungi etichetta vicino al punto
        ax.annotate(
            asset,
            (volatility[asset], mean_returns[asset]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )

    # Formattazione
    ax.set_xlabel('Volatilità Annualizzata', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rendimento Atteso Annualizzato', fontsize=12, fontweight='bold')
    ax.set_title('Profilo Rischio/Rendimento - Asset Individuali', fontsize=14, fontweight='bold', pad=20)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save:
        PLOTS_PATH.mkdir(parents=True, exist_ok=True)
        filepath = PLOTS_PATH / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato: {filepath}")
    
    return fig

# =============================================================================
# SEZIONE 8: FUNZIONE PRINCIPALE - CREA TUTTI I GRAFICI
# =============================================================================

def create_all_plots(results: SimulationResults, optimal: dict, returns: pd.DataFrame, save: bool = True) -> dict:
    """
    Crea tutti i grafici principali.
    
    Args:
        results: Risultati simulazione
        optimal: Portafogli ottimali
        returns: DataFrame rendimenti
        save: Se True, salva i grafici
    
    Returns:
        Dizionario con tutte le figure create
    """

    print("\n" + "=" * 70)
    print("GENERAZIONE GRAFICI")
    print("=" * 70)
    
    figures = {}
    
    print("\n[1/6] Frontiera Efficiente...")
    figures['efficient_frontier'] = plot_efficient_frontier(results, optimal, save)
    
    print("[2/6] Rendimenti Cumulativi...")
    figures['cumulative_returns'] = plot_cumulative_returns(returns, save)
    
    print("[3/6] Matrice di Correlazione...")
    figures['correlation_matrix'] = plot_correlation_matrix(returns, save)
    
    print("[4/6] Distribuzione Rendimenti...")
    figures['returns_distribution'] = plot_returns_distribution(returns, save)
    
    print("[5/6] Allocazioni Ottimali...")
    figures['optimal_allocations'] = plot_optimal_allocations(optimal, save)
    
    print("[6/6] Asset Individuali...")
    figures['individual_assets'] = plot_individual_assets(returns, save=save)
    
    print("\n" + "=" * 70)
    print("TUTTI I GRAFICI GENERATI!")
    print("=" * 70)
    
    return figures

# =============================================================================
# ESEMPIO DI UTILIZZO
# =============================================================================

if __name__ == "__main__":
    # Carica dati necessari
    from simulation import run_simulation
    
    # Esegui simulazione
    results, optimal = run_simulation(n_portfolios=10000, seed=42, save_results=True)
    
    # Carica rendimenti
    prices = load_cleaned_data()
    returns = calculate_return(prices)
    
    # Crea tutti i grafici
    figures = create_all_plots(results, optimal, returns, save=True)
    
    # Mostra grafici
    plt.show()
