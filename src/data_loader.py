"""
Modulo per il donwload e salvataggio dei dati grezzi da Yahoo Finance.
Scarica i prezzi storici di un subset di azioni dell' S&P 500.
"""

import yfinance as yf 
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path 
from typing import List, Tuple 
import json 


# Percorsi di default 
RAW_DATA_PATH = Path("data/raw")
REPORT_PATH = Path("data/raw")
CLEANED_DATA_PATH = Path("data/cleaned")


def get_sp500_subset() -> List[str]:
    """
    Restituisce un subset bilanciato di 100 ticker dell'S&P 500.
    
    La selezione è bilanciata per settore, rispettando approssimativamente
    i pesi dell'indice S&P 500. I titoli sono selezionati per capitalizzazione
    e liquidità.
    
    Distribuzione:
    - Information Technology: 30 titoli (~32%)
    - Financials: 13 titoli (~13%)
    - Health Care: 10 titoli (~10%)
    - Consumer Discretionary: 10 titoli (~11%)
    - Communication Services: 8 titoli (~9%)
    - Industrials: 8 titoli (~8%)
    - Consumer Staples: 6 titoli (~5.5%)
    - Energy: 5 titoli (~3%)
    - Utilities: 4 titoli (~2.3%)
    - Real Estate: 3 titoli (~2%)
    - Materials: 3 titoli (~2%)
    
    Returns:
        Lista di 100 ticker symbols
    """

    # Information Technology (30 titoli)
    tech = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # Nvidia
        "AVGO",   # Broadcom
        "ORCL",   # Oracle
        "CRM",    # Salesforce
        "AMD",    # Advanced Micro Devices
        "CSCO",   # Cisco
        "IBM",    # IBM
        #"INTC",   # Intel
        #"QCOM",   # Qualcomm
        #"TXN",    # Texas Instruments
        #"NOW",    # ServiceNow
        #"AMAT",   # Applied Materials
        #"INTU",   # Intuit
        #"MU",     # Micron Technology
        #"LRCX",   # Lam Research
        #"ADI",    # Analog Devices
        #"KLAC",   # KLA Corporation
        #"PANW",   # Palo Alto Networks
        #"ADBE",   # Adobe
        #"SNPS",   # Synopsys
        #"CDNS",   # Cadence Design Systems
        #"CRWD",   # CrowdStrike
        #"ANET",   # Arista Networks
        #"APH",    # Amphenol
        #"FTNT",   # Fortinet
        #"MCHP",   # Microchip Technology
        #"MPWR",   # Monolithic Power Systems
        #"NXPI",   # NXP Semiconductors
    ]
    # Financials (13 titoli)
    financials = [
        "BRK-B",  # Berkshire Hathaway
        "JPM",    # JPMorgan Chase
        "V",      # Visa
        #"MA",     # Mastercard
        #"BAC",    # Bank of America
        #"WFC",    # Wells Fargo
        #"GS",     # Goldman Sachs
        #"MS",     # Morgan Stanley
        #"SPGI",   # S&P Global
        #"BLK",    # BlackRock
        #"C",      # Citigroup
        #"SCHW",   # Charles Schwab
        #"AXP",    # American Express
    ]
    
    # Health Care (10 titoli)
    healthcare = [
        #"LLY",    # Eli Lilly
        #"UNH",    # UnitedHealth Group
        #"JNJ",    # Johnson & Johnson
        #"ABBV",   # AbbVie
        #"MRK",    # Merck
        #"PFE",    # Pfizer
        #"TMO",    # Thermo Fisher Scientific
        #"ABT",    # Abbott Laboratories
        #"AMGN",   # Amgen
        #"ISRG",   # Intuitive Surgical
    ]
    
    # Consumer Discretionary (10 titoli)
    consumer_disc = [
        "AMZN",   # Amazon
        "TSLA",   # Tesla
        #"HD",     # Home Depot
        #"MCD",    # McDonald's
        #"NKE",    # Nike
        #"LOW",    # Lowe's
        #"SBUX",   # Starbucks
        #"BKNG",   # Booking Holdings
        #"TJX",    # TJX Companies
        #"CMG",    # Chipotle
    ]
    
    # Communication Services (8 titoli)
    comm_services = [
        "GOOGL",  # Alphabet Class A
        "META",   # Meta Platforms
        #"NFLX",   # Netflix
        #"DIS",    # Walt Disney
        #"CMCSA",  # Comcast
        #"VZ",     # Verizon
        #"T",      # AT&T
        #"TMUS",   # T-Mobile US
    ]
    
    # Industrials (8 titoli)
    industrials = [
        #"GE",     # GE Aerospace
        #"CAT",    # Caterpillar
        #"RTX",    # RTX Corporation
        #"HON",    # Honeywell
        #"UNP",    # Union Pacific
        #"BA",     # Boeing
        #"LMT",    # Lockheed Martin
        #"DE",     # Deere & Company
    ]
    
    # Consumer Staples (6 titoli)
    consumer_staples = [
        #"PG",     # Procter & Gamble
        #"KO",     # Coca-Cola
        #"PEP",    # PepsiCo
        #"COST",   # Costco
        #"WMT",    # Walmart
        #"PM",     # Philip Morris
    ]
    
    # Energy (5 titoli)
    energy = [
        #"XOM",    # ExxonMobil
        #"CVX",    # Chevron
        #"COP",    # ConocoPhillips
        #"SLB",    # Schlumberger
        #"EOG",    # EOG Resources
    ]
    
    # Utilities (4 titoli)
    utilities = [
        #"NEE",    # NextEra Energy
        #"DUK",    # Duke Energy
        #"SO",     # Southern Company
        #"D",      # Dominion Energy
    ]
    
    # Real Estate (3 titoli)
    real_estate = [
        #"PLD",    # Prologis
        #"AMT",    # American Tower
        #"EQIX",   # Equinix
    ]
    
    # Materials (3 titoli)
    materials = [
        #"LIN",    # Linde
        #"APD",    # Air Products
        #"NEM",    # Newmont
    ]

    # Combina tutti i settori
    all_tickers = (
        tech + financials + healthcare + consumer_disc +
        comm_services + industrials + consumer_staples +
        energy + utilities + real_estate + materials
    )

    return all_tickers


def get_sector_mapping() -> dict:
    """
    Restituisce un dizionario che mappa ogni ticker al suo settore.
    
    Returns:
        Dict con ticker come chiave e settore come valore
    """
    sectors = {
        # Information Technology
        "AAPL": "Information Technology", "MSFT": "Information Technology",
        "NVDA": "Information Technology", "AVGO": "Information Technology",
        "ORCL": "Information Technology", "CRM": "Information Technology",
        "AMD": "Information Technology", "CSCO": "Information Technology",
        #"IBM": "Information Technology", "INTC": "Information Technology",
        #"QCOM": "Information Technology", "TXN": "Information Technology",
        #"NOW": "Information Technology", "AMAT": "Information Technology",
        #"INTU": "Information Technology", "MU": "Information Technology",
        #"LRCX": "Information Technology", "ADI": "Information Technology",
        #"KLAC": "Information Technology", "PANW": "Information Technology",
        #"ADBE": "Information Technology", "SNPS": "Information Technology",
        #"CDNS": "Information Technology", "CRWD": "Information Technology",
        #"ANET": "Information Technology", "APH": "Information Technology",
        #"FTNT": "Information Technology", "MCHP": "Information Technology",
        #"MPWR": "Information Technology", "NXPI": "Information Technology",
        
        # Financials
        "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
        #"MA": "Financials", "BAC": "Financials", "WFC": "Financials",
        #"GS": "Financials", "MS": "Financials", "SPGI": "Financials",
        #"BLK": "Financials", "C": "Financials", "SCHW": "Financials",
        #"AXP": "Financials",
        
        # Health Care
        #"LLY": "Health Care", "UNH": "Health Care", "JNJ": "Health Care",
        #"ABBV": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
        #"TMO": "Health Care", "ABT": "Health Care", "AMGN": "Health Care",
        #"ISRG": "Health Care",
        
        # Consumer Discretionary
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        #"HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
        #"NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
        #"SBUX": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
        #"TJX": "Consumer Discretionary", "CMG": "Consumer Discretionary",
        
        # Communication Services
        "GOOGL": "Communication Services", "META": "Communication Services",
        #"NFLX": "Communication Services", "DIS": "Communication Services",
        #"CMCSA": "Communication Services", "VZ": "Communication Services",
        #"T": "Communication Services", "TMUS": "Communication Services",
        
        # Industrials
        #"GE": "Industrials", "CAT": "Industrials", "RTX": "Industrials",
        #"HON": "Industrials", "UNP": "Industrials", "BA": "Industrials",
        #"LMT": "Industrials", "DE": "Industrials",
        
        # Consumer Staples
        #"PG": "Consumer Staples", "KO": "Consumer Staples",
        #"PEP": "Consumer Staples", "COST": "Consumer Staples",
        #"WMT": "Consumer Staples", "PM": "Consumer Staples",
        
        # Energy
        #"XOM": "Energy", "CVX": "Energy", "COP": "Energy",
        #"SLB": "Energy", "EOG": "Energy",
        
        # Utilities
        #"NEE": "Utilities", "DUK": "Utilities",
        #"SO": "Utilities", "D": "Utilities",
        
        # Real Estate
        #"PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
        
        # Materials
        #"LIN": "Materials", "APD": "Materials", "NEM": "Materials",
    }

    return sectors


def download_single_ticker(ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, bool, str]:
    """
    Scarica i dati di un singolo ticker.
    
    Args:
        ticker: Simbolo del titolo
        start_date: Data inizio (YYYY-MM-DD)
        end_date: Data fine (YYYY-MM-DD)
    
    Returns:
        Tuple: (DataFrame, successo, messaggio_errore)
    """

    try: 
        data = yf.download( 
            ticker, 
            start=start_date,
            end=end_date,
            progress=False
        )

        if data.empty:
            return pd.DataFrame(), False, "Nessun dato disponibile"
        
        # Gestisci MultiIndex nelle colonne (yfinance a volte lo crea)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Mantieni solo le colonne necessarie
        columns =["Open", "High", "Low", "Close", "Volume"]
        data = data[columns].copy()

        # L'indice di yfinance e' gia' datetimeindex con nome 'Date'
        # Reset index per avere Date come colonna
        data.index.name = 'Date'
        data = data.reset_index()

        # Aggiungi colonna Ticker
        data['Ticker'] = ticker

        return data, True, ""
    
    except Exception as e: 
        return pd.DataFrame(), False, str(e)
    

def download_raw_data(tickers: List[str], years: int = 15, end_date: str = None) -> Tuple[pd.DataFrame, dict]:
    """
    Scarica i dati grezzi per una lista di ticker.
    Args:
        tickers: Lista di simboli dei titoli
        years: Numero di anni di dati da scaricare (default 10)
        end_date: Data di fine (Default: oggi)

    Returns:
        Tuple: (DataFrame con tutti i dati,report download)
    """

    # Calcola date
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    days = years * 365 
    start_date = (datetime.today() - timedelta(days)).strftime('%Y-%m-%d')


    # Inizializza report 
    report = {
        'start_date': start_date,
        'end_date': end_date,   
        'total_tickers': len(tickers),
        'successful': [],
        'failed': [],
        'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }    
    
    all_data = []

    print(f"Download dati dal {start_date} al {end_date}")
    print(f"Ticker da scaricare: {len(tickers)}")
    print("-" * 70 )

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Scaricando {ticker}...", end=" ")

        data, success, error = download_single_ticker(ticker, start_date, end_date)

        if success:
            all_data.append(data)
            report['successful'].append(ticker)
            print("OK")
        else:
            report['failed'].append({'ticker': ticker, 'error': error})
            print(f"ERRORE: {error}")

    # Concatena tutti i DataFrame

    if all_data:
        combined_df = pd.concat(all_data, axis=0, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    # Aggiungi statistiche al report
    report['successful_count'] = len(report['successful'])
    report['failed_count'] = len(report['failed'])
    report['success_rate'] = f"{len(report['successful'])/len(tickers)*100:.1f}%"
    print("-" * 70 )
    print(f"Completato: {report['successful_count']} / {len(tickers)} ticker scaricati")

    return combined_df, report


def save_raw_data(df: pd.DataFrame, filename: str = "sp500_raw_prices.csv") -> Path:
    """
    Salva i dati grezzi in formato CSV.
    
    Args: 
        df: DataFrame con i dati 
        filename: Nome del file CSV

    Returns:
        Path del file salvato
    """

    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath =  RAW_DATA_PATH / filename

    df.to_csv(filepath, index=False)
    print(f"Dati salvati in: {filepath}")

    return filepath


def save_report(report: dict, filename: str = "download_report.json") -> Path:
    """
    Salva il report di download in formato JSON.

    Args: 
        report: Dizionario con il report
        filename: Nome del file

    Returns: 
        Path del file salvato
    """

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    filepath = REPORT_PATH / filename

    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report salvato in {filepath}")

    return filepath


def print_report_summary(report: dict) -> None:
    """
    Stampa un riepilogo del report di download.

    Args: 
        report: Dizionario con il report
    """

    print("\n" + "=" * 70)
    print("REPORT DOWNLOAD")
    print("=" * 70)
    print(f"Data download: {report['download_time']}")
    print(f"Periodo: {report['start_date']} → {report['end_date']}")
    print(f"Ticker richiesti: {report['total_tickers']}")
    print(f"Scaricati con successo: {report['successful_count']}")
    print(f"Falliti: {report['failed_count']}")
    print(f"Tasso di successo: {report['success_rate']}")

    if report['failed']:
        print("\nTicker falliti:")
        for item in report['failed']:
            print(f" - {item['ticker']}: {item['error']}")
    
    print("=" * 70 )



# =========================================================================================
# SEZIONE 2: PULIZIA DATI (RAW -> CLEANED)
# =========================================================================================


def load_raw_data(filename: str = "sp500_raw_prices.csv") -> pd.DataFrame:
    """
    Carica i dati grezzi dal file CSV 

    Args: 
        filename: Nome del file in data/raw

    Return: 
        DtaFrame con i dati grezzi
    """

    filepath = RAW_DATA_PATH / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])

    print(f"Dati caricati: {len(df)} righe, {df['Ticker'].nunique()} ticekr")
    return df


def pivot_to_wide_format(df: pd.DataFrame, value_column: str = 'Close') -> pd.DataFrame:
    """
    Converte i dati da formato lungo a formato wide.

    da: Date | Ticker | Open | High | Low | Close | Volume
    a: Date | AAPL | MSFT | GOOGL | .. (solo close)

    Args: 
        df: DataFrame in formato lungo 
        value_column: Colonna da usare come valore (dafault "Close)

    Returns: 
        DataFrame in formato wide con Date come indice 
    """
    # Seleziona solo le colonne necessarie prima del pivot
    df_subset = df[ ['Date', 'Ticker', value_column]].copy()

    # Rimuovi eventuali righe con Ticker nullo
    df_subset = df_subset.dropna(subset=['Ticker'])


    pivot_df = df_subset.pivot(index='Date', columns = 'Ticker', values=value_column)
    pivot_df.sort_index(inplace=True)

    # Rimuovi il nome delle colonne (Ticker) per pulizia
    pivot_df.columns.name = None 

    print(f"Formato wide: {pivot_df.shape[0]} giorni x {pivot_df.shape[1]} ticekr")
    return pivot_df


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizza i valori mancanti per ogni ticker.

    Args: 
        df: Dataframe in formato wide

    Retunrs:
        DataFrame con statidtiche sui missing values 
    """

    missing_stats = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'first_valid': df.apply(lambda x: x.first_valid_index()),
        'last_valid': df.apply(lambda x: x.last_valid_index())
    })

    missing_stats.sort_values('missing_pct', ascending=False, inplace=True)

    return missing_stats


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill', max_missing_pct: float =5.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gestisce i valori mancanti nei dati.

    Args: 
        df: DataFrame in formato wide
        methos: Metodo di interpolazione ('ffill', 'bfill', 'interpolate')
        max_missing_pct: Percentuale massima di missing tollerata per ticker

    Returns: 
        Tuple: (DataFrame pulito, lista ticker rimossi)
    """

    # Analizza missing values
    missing_stats = analyze_missing_values(df)

    # Identifica ticker con troppi missing 
    tickers_to_remove = missing_stats[
            missing_stats['missing_pct'] > max_missing_pct
        ].index.tolist()
    
    if tickers_to_remove:
        print(f"Ticker rimossi (>{max_missing_pct}% missing): {tickers_to_remove}")
        df = df.drop(columns=tickers_to_remove)

    # Applica metodo filling
    if method == 'ffill':
        df = df.ffill() # Forward fill

    elif method == 'bfill':
        df = df.bfill() # Backward fill

    elif method == 'interpolate': 
        df = df.interpolate(method='linear')

    # Rimuovi eventuali righe ancon con NaN (es all'inizio)
    df = df.dropna()

    print(f"Dopo pulizia: {df.shape[0]} giorni x {df.shape[1]} ticker")
    return df, tickers_to_remove


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Esegue controlli di qualita sui dati.

    Args: 
        df: DataFrame in formato wide (pulito)
    
    Returns: 
        Dizionario con report di qualita 
    """
    # Selezione solo colonne numeriche per i controlli
    numeric_df = df.select_dtypes(include=[np.number])

    # Assicurati che l'indice sia datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    quality_report = {
        'n_days': len(df),
        'n_tickers': len(numeric_df.columns),
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end': df.index.max().strftime('%Y-%m-%d')
        },
        'missing_values': numeric_df.isnull().sum().sum(),
        'negative_values': (numeric_df < 0).sum().sum(),
        'zero_values': (numeric_df == 0).sum().sum(),
        'duplicated_dates': df.index.duplicated().sum(),
        'issues': []
    }

    # Verifica problemi
    if quality_report['missing_values'] > 0:
        quality_report['issues'].append("Presenti valori mancanti")
    
    if quality_report['negative_values'] > 0:
        quality_report['issues'].append("Presenti valori negativi")
    
    if quality_report['zero_values'] > 0:
        quality_report['issues'].append("Presenti valori zero")
    
    if quality_report['duplicated_dates'] > 0:
        quality_report['issues'].append("Date duplicate")
    
    # Verifica gap nelle date (> 5 giorni lavorativi)
    date_diff = df.index.to_series().diff().dt.days
    max_gap = date_diff.max()
    if max_gap > 5:
        quality_report['issues'].append(f"Gap massimo tra date: {max_gap} giorni")
    
    quality_report['is_clean'] = len(quality_report['issues']) == 0
    
    return quality_report


def save_cleaned_data(df: pd.DataFrame, filename: str = "sp500_cleaned_prices.csv") -> Path:
    """
    Salva i dati puliti in formato CSV.
    
    Args:
        df: DataFrame pulito in formato wide
        filename: Nome del file di output
    
    Returns:
        Path del file salvato
    """
    CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = CLEANED_DATA_PATH / filename
    
    df.to_csv(filepath)
    print(f"Dati puliti salvati in: {filepath}")
    
    return filepath


def convert_numpy_types(obj):
    """
    Converte ricorsivamente i tipi numpy in tipi Python nativi.
    
    Args:
        obj: Oggetto da convertire (dict, list, o valore singolo)
    
    Returns:
        Oggetto con tipi Python nativi
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj
    

def save_cleaning_report(report: dict, removed_tickers: List[str], filename: str = "cleaning_report.json") -> Path:
    """
    Salva il report di pulizia.
    
    Args:
        report: Report di qualità
        removed_tickers: Lista ticker rimossi
        filename: Nome del file
    
    Returns:
        Path del file salvato
    """
    CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = CLEANED_DATA_PATH / filename
    
    full_report = {
        'cleaning_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'quality_report': report,
        'removed_tickers': removed_tickers,
        'final_tickers_count': report['n_tickers']
    }
    
    # Converti tipi numpy in tipi Python nativi
    full_report = convert_numpy_types(full_report)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"Report pulizia salvato in: {filepath}")
    return filepath


def run_cleaning(raw_filename: str = "sp500_raw_prices.csv", max_missing_pct: float = 5.0) -> Tuple[Path, Path]:
    """
    Esegue l'intero processo di pulizia dei dati.
    
    Args:
        raw_filename: Nome del file raw da pulire
        max_missing_pct: Percentuale massima di missing tollerata
    
    Returns:
        Tuple: (path_dati_puliti, path_report)
    """
    print("\n" + "=" * 70)
    print("PULIZIA DATI")
    print("=" * 70)
    
    # 1. Carica dati grezzi
    print("\n[1/5] Caricamento dati grezzi...")
    raw_df = load_raw_data(raw_filename)
    
    # 2. Pivot a formato wide
    print("\n[2/5] Conversione a formato wide...")
    wide_df = pivot_to_wide_format(raw_df, value_column='Close')
    
    # 3. Analisi missing values
    print("\n[3/5] Analisi valori mancanti...")
    missing_stats = analyze_missing_values(wide_df)
    print(f"Ticker con missing values: {(missing_stats['missing_count'] > 0).sum()}")
    
    # 4. Gestione missing values
    print("\n[4/5] Gestione valori mancanti...")
    cleaned_df, removed_tickers = handle_missing_values(
        wide_df, 
        method='ffill',
        max_missing_pct=max_missing_pct
    )
    
    # 5. Controllo qualità finale
    print("\n[5/5] Controllo qualità...")
    quality_report = check_data_quality(cleaned_df)
    
    if quality_report['is_clean']:
        print("✓ Dati puliti senza problemi")
    else:
        print("⚠ Problemi rilevati:")
        for issue in quality_report['issues']:
            print(f"  - {issue}")
    
    # Salvataggio
    print("\nSalvataggio...")
    data_path = save_cleaned_data(cleaned_df)
    report_path = save_cleaning_report(quality_report, removed_tickers)
    
    print("\n" + "=" * 70)
    print("RIEPILOGO PULIZIA")
    print("=" * 70)
    print(f"Periodo: {quality_report['date_range']['start']} → {quality_report['date_range']['end']}")
    print(f"Giorni di trading: {quality_report['n_days']}")
    print(f"Ticker finali: {quality_report['n_tickers']}")
    print(f"Ticker rimossi: {len(removed_tickers)}")
    print("=" * 70)
    
    return data_path, report_path


def run_download(tickers: List[str], years: int = 15) -> Tuple[Path, Path]:
    """
    Funzione principale per eseguire il download completo.
    
    Args:
        tickers: Lista dei ticker da scaricare
        years: Anni di storico
    
    Returns:
        Tuple: (path_dati, path_report)
    """
    # Download
    df, report = download_raw_data(tickers, years)
    
    # Salvataggio
    data_path = save_raw_data(df)
    report_path = save_report(report)
    
    # Stampa riepilogo
    print_report_summary(report)
    
    return data_path, report_path



# Esempio di utilizzo
if __name__ == "__main__":
    # =================================================================
    # STEP 1: DOWNLOAD DATI RAW
    # =================================================================
    # Opzione A: Test con pochi ticker
    test_tickers = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # Nvidia
        "AVGO",   # Broadcom
        "ORCL",   # Oracle
        "CRM",    # Salesforce
        "AMD",    # Advanced Micro Devices
        "CSCO",   # Cisco
        "IBM",    # IBM
        "BRK-B",  # Berkshire Hathaway
        "JPM",    # JPMorgan Chase
        "V",      # Visa
        "AMZN",   # Amazon
        "TSLA",   # Tesla
        "GOOGL",  # Alphabet Class A
        "META",   # Meta Platforms
        ]
    data_path, report_path = run_download(test_tickers, years=5)
    
    # Decommentare per eseguire il download completo
    
    # =================================================================
    # STEP 2: PULIZIA DATI (RAW -> CLEANED)
    # =================================================================
    # Decommentare dopo aver eseguito il download
    cleaned_path, cleaning_report_path = run_cleaning(
         raw_filename="sp500_raw_prices.csv",
         max_missing_pct=5.0
     )