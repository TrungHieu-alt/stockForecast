import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

RAW_DIR = "rawData"
CLEAN_DIR = "cleanDataset"

# Regex to collapse multiple spaces in date strings
date_space_pattern = re.compile(r"\s+")

def robust_parse_date(date_str):
    """Parse messy date strings by collapsing spaces and coercing to datetime."""
    if pd.isna(date_str):
        return pd.NaT
    # Collapse multiple spaces and strip
    s = date_space_pattern.sub(" ", str(date_str)).strip()
    try:
        # Try default parsing
        return pd.to_datetime(s, errors="raise")
    except Exception:
        # Fallback to coercion
        return pd.to_datetime(s, dayfirst=False, yearfirst=False, errors="coerce")


def clean_stock_csv(input_file, output_file):
    # Read with utf-8-sig to remove BOM if present
    df = pd.read_csv(input_file, sep=';', engine='python', encoding='utf-8-sig')
    # Normalize column names
    df.columns = [col.strip() for col in df.columns]

    # Parse dates
    if 'Date' not in df.columns:
        raise KeyError(f"Column 'Date' not found in {input_file}")
    df['Date'] = df['Date'].apply(robust_parse_date)

    # Check date parsing ratio
    na_dates = df['Date'].isna().mean()
    if na_dates > 0.1:
        print(f"⚠️ Warning: {os.path.basename(input_file)} has {na_dates*100:.1f}% unparsed dates.")

    # Clean numeric columns: remove commas and spaces
    num_cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in df.columns]
    for col in num_cols:
        # Remove thousand separators and stray spaces
        df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
        # Convert to numeric, coerce errors
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing critical data
    df = df.dropna(subset=['Date', 'Close']).reset_index(drop=True)
    # Sort by date ascending
    df = df.sort_values('Date').reset_index(drop=True)

    # Save cleaned CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Cleaned & saved: {output_file} (Rows: {len(df)})")
    return df


def normalize_and_split(df, out_dir, col='Close', train_ratio=0.8, val_ratio=0.1):
    # Extract series and scale
    values = df[col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series = scaler.fit_transform(values).flatten()

    # Compute split indices
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train, val, test = series[:train_end], series[train_end:val_end], series[val_end:]

    # Save splits
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'val.npy'), val)
    np.save(os.path.join(out_dir, 'test.npy'), test)
    print(f"✅ Saved splits → {out_dir} (train:{len(train)}, val:{len(val)}, test:{len(test)})")


def main():
    tickers = ["HoaPhat", "fpt", "HAGL", "mb", "VNM"]
    # Ensure output dir exists
    os.makedirs(CLEAN_DIR, exist_ok=True)

    for name in tickers:
        raw_path = os.path.join(RAW_DIR, f"{name}.csv")
        clean_path = os.path.join(CLEAN_DIR, f"{name}_cleaned.csv")
        split_dir = os.path.join(CLEAN_DIR, f"split_{name}")

        if not os.path.exists(raw_path):
            print(f"⚠️  {raw_path} not found, skipping.")
            continue
        # Clean and parse
        df = clean_stock_csv(raw_path, clean_path)
        # Normalize and split
        normalize_and_split(df, split_dir)


if __name__ == '__main__':
    main()
