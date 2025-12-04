import pandas as pd

# 1. Carica raw
df = pd.read_csv("data/raw/sp500_raw_prices.csv")
print("=== DATI RAW ===")
print(f"Righe totali: {len(df)}")
print(f"Ticker unici: {df['Ticker'].unique().tolist()}")

# 2. Simula il pivot
df['Date'] = pd.to_datetime(df['Date'])
df_subset = df[['Date', 'Ticker', 'Close']].copy()
df_subset = df_subset.dropna(subset=['Ticker'])

pivot_df = df_subset.pivot(index='Date', columns='Ticker', values='Close')
pivot_df.columns.name = None

print("\n=== DOPO PIVOT ===")
print(f"Shape: {pivot_df.shape}")
print(f"Colonne: {pivot_df.columns.tolist()}")

# 3. Analizza missing per ogni colonna
print("\n=== MISSING VALUES PER TICKER ===")
for col in pivot_df.columns:
    missing = pivot_df[col].isnull().sum()
    total = len(pivot_df)
    pct = (missing / total) * 100
    print(f"{col}: {missing}/{total} missing ({pct:.2f}%)")


# ============ secodo debug ============
print("=" *50)
print("SECONDO DEBUG")
print("=" *50)

df = pd.read_csv("data/raw/sp500_raw_prices.csv")

# Vedi le prime righe per ogni ticker
for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
    print(f"\n=== {ticker} ===")
    subset = df[df['Ticker'] == ticker]
    print(f"Righe: {len(subset)}")
    print(subset[['Date', 'Ticker', 'Close']].head(3))

# Vedi anche le righe con Ticker = nan
print("\n=== RIGHE CON TICKER NaN ===")
nan_rows = df[df['Ticker'].isna()]
print(f"Righe: {len(nan_rows)}")
print(nan_rows.head(5))


# ============ terzo debug ============
print("=" *50)
print("TERZO DEBUG")
print("=" *50)

# Carica il file cleaned
df = pd.read_csv("data/cleaned/sp500_cleaned_prices.csv", index_col=0)

print("Tipo indice:", type(df.index))
print("Prime 5 date (raw):", df.index[:5].tolist())
print("Tipo primo elemento:", type(df.index[0]))

# ============ quarto debug ============
print("=" *50)
print("QUARTO DEBUG")
print("=" *50)


# Carica raw
df = pd.read_csv("data/raw/sp500_raw_prices.csv")

print("=== PRIMA DEL PIVOT ===")
print("Tipo colonna Date:", df['Date'].dtype)
print("Prime 3 date:", df['Date'].head(3).tolist())

# Converti Date
df['Date'] = pd.to_datetime(df['Date'])
print("\nDopo pd.to_datetime:")
print("Tipo colonna Date:", df['Date'].dtype)
print("Prime 3 date:", df['Date'].head(3).tolist())

# Fai il pivot
df_subset = df[['Date', 'Ticker', 'Close']].copy()
df_subset = df_subset.dropna(subset=['Ticker'])
pivot_df = df_subset.pivot(index='Date', columns='Ticker', values='Close')

print("\n=== DOPO IL PIVOT ===")
print("Tipo indice:", type(pivot_df.index))
print("Prime 3 date indice:", pivot_df.index[:3].tolist())


print("=" *50)
print("=" *50)
print("=" *50)
df = pd.read_csv("data/raw/sp500_raw_prices.csv")

print("Colonne:", df.columns.tolist())
print("\nTipo colonna Date:", df['Date'].dtype)
print("\nPrime 5 righe:")
print(df[['Date', 'Ticker', 'Close']].head(5))