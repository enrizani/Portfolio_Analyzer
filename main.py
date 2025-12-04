"""
Script principale per l'analisi di portafoglio.
Orchestra l'intero processo: download, pulizia, analisi, simulazione e visualizzazione.
"""

import sys 
from pathlib import Path
import argparse

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import (
    get_sp500_subset, 
    run_download,
    run_cleaning
)

from src.analysis import run_analysis
from src.simulation import run_simulation
from src.plotting import create_all_plots
from src.utils import (
    ProjectConfig,
    print_section_header,
    print_subsection_header,
    print_key_value,
    save_dict_to_json,
)

import matplotlib.pyplot as plt



# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def run_full_pipeline(download_data: bool=False, clean_data: bool=False, years: int=10, n_portfolios: int=10000, risk_free_rate: float=0.02,seed: int=42, show_plots: bool=True)-> dict:
    """
    Esegue l'intera pipeline del progetto.

    Args:
        download_data: Se True, scarica nuovi dati
        clean_data: Se True, pulisce i dati raw
        n_portfolios: Numero di portafogli da simulare
        risk_free_rate: Tasso risk-free annualizzato
        seed: Seed per riproducibilita'
        show_plots: Se True, mostra igrafici a schermo

    Returns: 
        Dizionario con tutti i risultati
    """
    print_section_header("ANALISI PORTAFOGLIO - AVVIO PIPELINE")

    # 1. Setup iniziale
    print_subsection_header("SETUP INIZIALE")
    config = ProjectConfig()
    config.create_directories()
    print("Directories create")

    results_summary = {
        'config': config.to_dict(),
        'parameters': {
            'n_portfolios': n_portfolios,
            'risk_free_rate': risk_free_rate,
            'seed': seed
        }
    }

    # 2. Download dati (opzionale)
    if download_data:
        print_section_header("STEP 1: DOWNLOAD DATI ")
        tickers = get_sp500_subset()
        print(f"Ticker da scaricare: {len(tickers)}")

        data_path, report_path = run_download(tickers, years=years)
        results_summary['download'] = {
            'data_path': str(data_path),
            'report_path': str(report_path)
        }
    else:
        print_section_header("STEP 1: DOWNLOAD DATI [SALTATO]")
        print("Utilizzo dati esistenti in data/raw/")

    # 3. Pulizia dati (opzionale)
    if clean_data:
        print_section_header("STEP 2: PULIZIA DATI")
        cleaned_path, cleaning_report_path = run_cleaning()
        results_summary['cleaning'] = {
            'cleaned_path': str(cleaned_path),
            'report_path': str(report_path)
        }
    else:
        print_section_header("STEP 2: PULIZIA DATI [SALTATO]")
        print("Utilizzo dati esistenti in data/cleaned/")
    
    # 4. Analisi finanziaria
    print_section_header("STEP 3: ANALISI FINANZIARIA")
    returns, cov_matrix, stats = run_analysis(risk_free_rate=risk_free_rate)

    results_summary['analysis'] = {
        'n_assets': len(returns.columns),
        'n_days': len(returns),
        'date_range': {
            'start': returns.index.min().strftime('%Y-%m-%d'),
            'end': returns.index.max().strftime('%Y-%m-%d')
        },
        'top_sharpe': {
            'ticker': stats.index[0],
            'sharpe_ratio': float(stats.iloc[0]['Sharpe_Ratio'])
        }
    }

    # 5. Simulazione Monte Carlo
    print_section_header("STEP 4: SIMULAZIONE MONTE CARLO")
    simulation_results, optimal_portfolios = run_simulation(
        n_portfolios=n_portfolios,
        risk_free_rate=risk_free_rate,
        seed=seed,
        save_results=True
    )

    results_summary['simulation'] = {
        'n_portfolios': n_portfolios,
        'optimal_portfolios': optimal_portfolios
    }

    # 6. Generazione grafici
    print_section_header('STEP 5: GENERAZIONE GRAFICI')
    figures = create_all_plots(
        simulation_results,
        optimal_portfolios,
        returns,
        save=True
    )

    results_summary['plots'] = {
        'n_plots_created': len(figures),
        'plots_path': str(config.plots_path)
    }

    # 7. Salva il riepilogo finale
    print_section_header("STEP 6: SALVATAGGIO RIEPILOGO")
    summary_path = config.results_path / "final_summary.json"
    save_dict_to_json(results_summary, summary_path)
    print(f"Riepilogo salvato: {summary_path}")

    # 8. Report finale
    print_final_report(results_summary, optimal_portfolios)

    # 9. Mostra i grafici
    #if show_plots:
        #print("\nMostra grafici...")
        #plt.show()
    
    return results_summary


# =============================================================================
# REPORT FINALE
# =============================================================================

def print_final_report(summary: dict, optimal: dict) -> None:
    """
    Stampa un report finale completo.

    Args:
        summary: Dizionario con riepilogo
        optimal: Portafogli ottimali
    """
    print_section_header("REPORT FINALE")

    # Analisi
    print_subsection_header("ANALISI DATI")
    analysis = summary['analysis']
    print_key_value("Titoli analizzati", analysis['n_assets'])
    print_key_value("Giorni di trading", analysis['n_days'])
    print_key_value("Periodo", f"{analysis['date_range']['start']} -> {analysis['date_range']['end']}")
    print_key_value("Top performer (Sharpe)", f"{analysis['top_sharpe']['ticker']} ({analysis['top_sharpe']['sharpe_ratio']:.3f})")

    # Simulazione
    print_subsection_header("SIMULAZIONE MONTE CARLO")
    sim = summary['simulation']
    print_key_value("Portafogli simulati", f"{sim['n_portfolios']:,}")

    # Portafogli ottimali
    print_subsection_header("PORTAFOGLI OTTIMALI")

    for key, portfolio in optimal.items():
        print(f"\n  ▶ {portfolio['name']}")
        print_key_value("Rendimento atteso", portfolio['return'], indent=4)
        print_key_value("Volatilità", portfolio['volatility'], indent=4)
        print_key_value("Sharpe Ratio", portfolio['sharpe_ratio'], indent=4)

        print("Allocazione (top 3):")
        sorted_weights = sorted(
            portfolio['weights'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for ticker, weight in sorted_weights:
            print(f"    {ticker}: {weight:.2%}")

    # Output generati
    print_subsection_header("OUTPUT GENERATI")
    print("\nANALISI COMPLETATA CON SUCCESSO!" .center(70))
    print("=" * 70)


# =============================================================================
# FUNZIONI DI UTILITÀ PER ESECUZIONI PARZIALI
# =============================================================================

def run_analysis_only(risk_free_rate: float=0.02) -> None:
    """Esegue solo l'analisi (senza simulazione)"""
    print_section_header("ANALISI FINANZIARIA")
    returns, cov_matrix, stats = run_analysis(risk_free_rate=risk_free_rate)
    print("\nAnalisi completata")

def run_simulation_only(n_portfolios: int=10000, risk_free_rate: float=0.02, seed: int=42)-> None:
    """Esegue solo la simulazione (pressuppone dati gia' puliti)"""
    print_section_header("SIMULAZIONE MONTE CARLO")
    results, optimal = run_simulation(
        n_portfolios=n_portfolios,
        risk_free_rate=risk_free_rate,
        seed=seed,
        save_results=True
    )
    print("\nSimulazione completata")

def run_plots_only()-> None:
    """Genera solo i grafici (presuppone simulazione gia' eseguita)"""
    from src.data_loader import load_cleaned_data
    from src.analysis import calculate_return
    from src.utils import load_dict_from_json
    import pandas as pd

    print_section_header("GENERAZIONE GRAFICI")

    # Carica risultati simulazione
    optimal = load_dict_from_json("data/results/optima_portfolios.json")['portfolios']
    results_df = pd.read_csv("data/results/simulation_results.csv")

    # Ricostruisci SimulationResults
    from src.simulation import SimulationResults
    prices = load_cleaned_data()
    returns = calculate_return(prices)

    results = SimulationResults(
        returns = results_df['Return'].values,
        volatilities=results_df['Volatility'].values,
        sharpe_ratios=results_df['Sharpe_Ratio'].values,
        weights=results_df[[c for c in results_df.columns if c.startwith('Weight_')]].values,
        n_portfolios=len(results_df),
        n_assets=len(returns.columns),
        asset_names=returns.columns.tolist()
    )

    # Crea grafici
    create_all_plots(results, optimal, returns, save=True)
    print("\nGrafici generati")
    plt.show()


# =============================================================================
# CLI - COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse degli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description='Analisi di Portafoglio - Simulazione Monte Carlo'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Esegue la pipeline completa (download + cleaning + analisi + simulazione + grafici)'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Scarica nuovi dati'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Pulisce i dati raw'
    )
    
    parser.add_argument(
        '--analysis',
        action='store_true',
        help='Esegue solo l\'analisi'
    )
    
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Esegue solo la simulazione'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Genera solo i grafici'
    )
    
    parser.add_argument(
        '--n-portfolios',
        type=int,
        default=10000,
        help='Numero di portafogli da simulare (default: 10000)'
    )
    
    parser.add_argument(
        '--risk-free',
        type=float,
        default=0.02,
        help='Tasso risk-free annualizzato (default: 0.02)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed per riproducibilità (default: 42)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Non mostrare i grafici a schermo'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Anni di dati storici da scaricare (default: 10)'
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Funazione main"""
    args = parse_arguments()

    try:
        if args.full:
            # Pipeline completa
            run_full_pipeline(
                download_data=True,
                clean_data=True,
                years=args.years,
                n_portfolios=args.n_portfolios,
                risk_free_rate=args.risk_free,
                seed=args.seed,
                show_plots=not args.no_show
            )
        elif args.analysis:
            # Solo analisi 
            run_analysis_only(risk_free_rate=args.risk_free)
        elif args.analysis:
            # Solo analisi 
            run_simulation_only(
                n_portfolios=args.n_portfolios,
                risk_free_rate=args.risk_free,
                seed=args.seed
            )
        elif args.plots:
            # Solo grafici
            run_plots_only()

        else:
            # Default: pipeline senza download/cleaning (usa dati esistenti)
            run_full_pipeline(
                download_data=args.download,
                clean_data=args.clean,
                n_portfolios=args.n_portfolios,
                risk_free_rate=args.risk_free,
                seed=args.seed,
                show_plots=not args.no_show
            )
        
    except Exception as e:
        print(f"\nERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()