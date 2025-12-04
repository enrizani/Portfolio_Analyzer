"""
Modulo con funzioni di utilita' trasversali.
Continee helper per validazione, conversione, formattazione e logging.
"""

import numpy as np
import pandas as pd 
from pathlib import Path 
from typing import Union, List, Any
from datetime import datetime
import json 


# =============================================================================
# SEZIONE 1: VALIDAZIONE INPUT
# =============================================================================

def validate_weights(weights: np.array, tolerance: float=1e-6) -> bool:
    """
    Valida che i pesi del portafoglio siano corretti.

    Args:
        weights: Array di pesi
        tolerance: Tollerazna per la somma (default 1e-6)
    
    Retruns:
        True se validi, False altrimenti
    """

    # Varfica che tutti i pesi siano positivi
    if np.any(weights < 0):
        return False

    # Verifica che la somma sia ~1
    if not np.isclose(weights.sum(), 1.0, atol=tolerance):
        return False
    
    return True


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Valida che il range di date sia corretto.

    Args:
        start_date: Data inizio 
        end_date: Data fine
        --> formato: 'YYYY-MM-DD'

    Returns:
        True se valido, False altrimenti
    """

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Verifica che end > start
        if end <= start:
            return False
        
        # Verifica che le date non siano nel futuro
        if end > pd.Timestamp.now():
            return False
        
        return True
    except:
        return False
    

def validate_risk_free_rate(rate: float) -> bool:
    """
    Valida che il tasso risk-free sia ragionevole.

    Args:
        rate: Tasso risk-free (es. 0.02 per 2%)

    Returns:
        True se valido, False altrimenti
    """

    # Deve essere tra -5% e 20% (valori estremi ma realistici)
    return -0.05 <= rate <= 0.20


# =============================================================================
# SEZIONE 2: CONVERSIONE FORMATI
# =============================================================================

def convert_numpy_to_python(obj: Any) -> Any:
    """
    Converte ricorsivamente tipi NumPy in tipi Python nativi.
    
    Args:
        obj: Oggetto da convertire
    
    Returns:
        Oggetto con tipi Python nativi
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):  # ← Cambiato da np.array
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj
    

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formatta un numero come percentuale.
    
    Args:
        value: Valore da formattare (es. 0.1523 per 15.23%)
        decimals: Numero di decimali
    
    Returns:
        Stringa formattata (es. "15.23%")
    """
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = "$") -> str:
    """
    Formatta un numero come valuta.
    
    Args:
        value: Valore da formattare
        currency: Simbolo valuta
    
    Returns:
        Stringa formattata (es. "$1,234.56")
    """
    return f"{currency}{value:,.2f}"


# =============================================================================
# SEZIONE 3: GESTIONE FILE E DIRECTORY
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Crea una directory se non esiste.

    Args:
        path: Percorso della direcotory

    Returns:
        Path della directory creata
    """

    path =Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_dict_to_json(data: dict, filepath: Union[str, Path]) -> Path:
    """
    Salva un dizionario in formato json

    Args:
        data: Dizionario da salvare
        filepath: Percorso del file

    Returns:
        Path del file salvato
    """

    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    # Converti tipi numpy
    data = convert_numpy_to_python(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath

def load_dict_from_json(filepath: Union[str, Path]) -> dict:
    """
    Carica un dizionario da file json

    Args:
        filepath: Percorso del file

    Returns:
        Dizionario caricato
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


# =============================================================================
# SEZIONE 4: STATISTICHE HELPER
# =============================================================================

def calculate_annualized_return(cumulative_return: float, n_years: float) -> float:
    """
    Calcola il rendimento annualizzato da un rendimento cumulativo

    Args:
        cumulative_return: Rendimento cumulativo totale
        n_years: Numero di anni

    Returns:
        Rendimento annualizzato
    """

    return (1 + cumulative_return) ** (1 / n_years) - 1

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calcola il Maximum Dreawdown di una serie di rendimenti.

    Args:
        returns: Serie di rendimenti

    Returns:
        Maximum Drawdown (valore negativo)
    """

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float=0.02, periods_per_year: int= 252) -> float:
    """
    Calcola il Sortino Ratio (come Sharpe ma usa solo volatilita' negativa).

    Args:
        returns: Serie di rendimenti giornalieri
        risk_free_rate: Tasso risk_free annualizzato
        periods_per_year: Periodi per anno

    Returns:
        Sortino Ratio
    """

    # Rendimento annualizzato
    mean_return = returns.mean() * periods_per_year

    # Downside deviation (solo rendimenti negativi)
    negative_returns = returns[returns < 0]
    downside__std = negative_returns.std() * np.sqrt(periods_per_year)

    if downside__std == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / downside__std


# =============================================================================
# SEZIONE 5: LOGGING E REPORT
# =============================================================================

def print_section_header(title: str, width: int=70) -> None:
    """
    Stampa un header formattato per sezioni.

    Args:
        title: Titolo della sezione 
        width: Larghezza della linea 
    """

    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def print_subsection_header(title: str, width: int = 70) -> None:
    """
    Stampa un header per sotto-sezioni.
    
    Args:
        title: Titolo della sotto-sezione
        width: Larghezza della linea
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)

def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """
    Stampa una coppia chiave-valore formattata.
    
    Args:
        key: Chiave
        value: Valore
        indent: Livello di indentazione
    """
    spaces = " " * indent
    
    # Formatta il valore in base al tipo
    if isinstance(value, float):
        if abs(value) < 1:
            value_str = format_percentage(value)
        else:
            value_str = f"{value:.4f}"
    else:
        value_str = str(value)
    
    print(f"{spaces}{key}: {value_str}")

def create_summary_dict(name: str, **kwargs) -> dict:
    """
    Crea un dizionario di riepilogo con timestamp.
    
    Args:
        name: Nome del riepilogo
        **kwargs: Coppie chiave-valore da includere
    
    Returns:
        Dizionario di riepilogo
    """
    summary = {
        'name': name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': kwargs
    }
    return summary


# =============================================================================
# SEZIONE 6: CONFIGURAZIONE PROGETTO
# =============================================================================

class ProjectConfig:
    """Classe per gestire la configurazione del progetto."""

    def __init__(self):
        self.data_path = Path("data")
        self.raw_path = self.data_path / "raw"
        self.cleaned_path = self.data_path / "cleaned"
        self.results_path = self.data_path / "results"
        self.plots_path = self.data_path / "plots"

        # Parametri di default
        self.trading_days_per_year = 252
        self.default_risk_free_rate = 0.02
        self.default_n_portfolios = 10000
        self.random_seed = 42

    def create_directories(self) -> None:
        """Crea tutte le directories necessarie."""
        for path in [self.raw_path, self.cleaned_path, self.results_path, self.plots_path]:
            ensure_directory(path)
        
    def to_dict(self) -> dict:
        """Converte la configurazione in dizionario."""
        return{
            'paths': {
                'data': str(self.data_path),
                'raw': str(self.raw_path),
                'cleaned': str(self.cleaned_path),
                'results': str(self.results_path),
                'plots': str(self.plots_path)
            },
            'parameters': {
                'trading_days_per_year': self.trading_days_per_year,
                'risk_free_rate': self.default_risk_free_rate,
                'n_portfolios': self.default_n_portfolios,
                'random_seed': self.random_seed
            }
        }

    def save(self, filename: str = "project_config.json") -> Path:
        """Salva la configurazione su file."""
        filepath = self.data_path / filename
        return save_dict_to_json(self.to_dict(), filepath)
    

# =============================================================================
# SEZIONE 7: PROGRESS BAR SEMPLICE
# =============================================================================

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int=50) -> None:
    """
    Stampa una progress bar testuale.

    Args:
        iteration: Iterazione corrente
        total: Totale iterazioni
        prefix: Testo prima della barra
        suffix: Testo dopo la barra
        length: Lunghezza della barra
    """

    percent = f"{100 * (iteration / float(total)):.1f}"
    filled = int(length * iteration // total)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')

    # Print New Line on Complete 
    if iteration == total:
        print()


# =============================================================================
# ESEMPIO DI UTILIZZO
# =============================================================================

if __name__ == "__main__":
    # Test validazione
    print_section_header("TEST VALIDAZIONE")
    
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"Pesi validi: {validate_weights(weights)}")
    
    print(f"Date valide: {validate_date_range('2020-01-01', '2023-12-31')}")
    print(f"Risk-free valido: {validate_risk_free_rate(0.02)}")
    
    # Test configurazione
    print_section_header("TEST CONFIGURAZIONE")
    
    config = ProjectConfig()
    config.create_directories()
    print("Directory create:")
    for key, path in config.to_dict()['paths'].items():
        print(f"  {key}: {path}")
    
    # Test formattazione
    print_section_header("TEST FORMATTAZIONE")
    
    print_key_value("Rendimento", 0.1523)
    print_key_value("Volatilità", 0.2156)
    print_key_value("Sharpe Ratio", 0.7065)
    