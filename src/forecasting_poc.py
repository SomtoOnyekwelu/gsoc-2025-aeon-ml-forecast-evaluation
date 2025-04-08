# GSoC 2025 Proof-of-Concept for Aeon Project #2: ML Forecasting Evaluation
# Demonstrates: Time series loading, feature engineering (lagging), basic ML model
# training & evaluation for forecasting. Uses argparse for flexibility.

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Default Configuration ---
DEFAULT_DATA_FILEPATH = 'data/sample_air_quality.csv'
DEFAULT_TIME_COL = 'date.utc'
DEFAULT_TARGET_COL = 'value'
DEFAULT_FILTER_COL = 'parameter'
DEFAULT_FILTER_VAL = 'no2'
N_LAGS_DEFAULT = 5
TRAIN_TEST_SPLIT_RATIO_DEFAULT = 0.8

# --- Functions ---

def load_and_prepare_data(filepath: str,
                          time_col: str,
                          target_col: str,
                          filter_col: str | None = None,
                          filter_val: str | None = None) -> pd.Series:
    """
    Loads, preprocesses, filters, sorts, and extracts target time series from CSV.

    Args:
        filepath: Path to the CSV file.
        time_col: Name of the timestamp column.
        target_col: Name of the target value column.
        filter_col: Optional column name to filter data by.
        filter_val: Optional value to keep in filter_col (requires filter_col).

    Returns:
        Processed time series (pd.Series) with time index.

    Purpose (GSoC Context): Demonstrates foundational data handling. Full GSoC
                         project will interface with diverse standard benchmark datasets
                         and require more sophisticated preprocessing.
    """
    print(f"--> Loading data from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        # Load specifying date column - improves efficiency and robustness
        df = pd.read_csv(filepath, parse_dates=[time_col])
        print(f"    Initial shape: {df.shape}")
    except ValueError as e:
        raise ValueError(
            f"Error parsing date column '{time_col}'. "
            f"Ensure column exists and is in a recognizable date format. Error: {e}"
        ) from e
    except Exception as e:
         raise IOError(f"Failed to load or read CSV '{filepath}'. Error: {e}") from e

    available_cols = df.columns.tolist() # Get columns before filtering

    # Optional Filtering step
    if filter_col and filter_val:
        print(f"--> Filtering where '{filter_col}' == '{filter_val}'...")
        if filter_col not in df.columns:
            raise ValueError(
                f"Filter column '{filter_col}' not found. "
                f"Available columns: {available_cols}"
            )
        df = df[df[filter_col] == filter_val].copy()
        if df.empty:
            raise ValueError(
                f"No data remaining after filtering for '{filter_col}' == '{filter_val}'. "
                "Check filter value or data source."
            )
        print(f"    Shape after filtering: {df.shape}")
    elif filter_col or filter_val:
        # Warn if only one is provided - prevents unexpected behavior
        print(
            "Warning: Filtering requires both --filter_col and --filter_val. "
            "Proceeding without filtering."
         )

    # --- Column Validation (Post-Filtering) ---
    essential_cols = {time_col, target_col}
    missing_cols = essential_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Essential columns missing after loading/filtering: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Sort by time - CRUCIAL step
    print(f"--> Sorting data by time column '{time_col}'...")
    df.sort_values(by=time_col, inplace=True)

    # Set time as index
    print(f"--> Setting '{time_col}' as index...")
    df.set_index(time_col, inplace=True)

    # Check for duplicate timestamps (common issue in real data)
    if df.index.duplicated().any():
        duplicates = df.index[df.index.duplicated()].unique()
        print(f"Warning: Duplicate timestamps found: {duplicates.tolist()[:5]}...") # Show first few
        # For PoC, keep first occurrence. Real project might need aggregation/error.
        df = df[~df.index.duplicated(keep='first')]
        print(f"    Removed duplicates, keeping first occurrences. Shape now: {df.shape}")


    # Select the target column
    time_series = df[target_col]

    # --- Data Cleaning ---
    print(f"--> Cleaning target column '{target_col}'...")
    # Ensure numeric, coerce errors (creates NaN), then drop NaNs
    time_series = pd.to_numeric(time_series, errors='coerce')
    original_len = len(time_series)
    time_series.dropna(inplace=True) # Simple strategy for PoC
    dropped_count = original_len - len(time_series)
    if dropped_count > 0:
        print(f"    Dropped {dropped_count} non-numeric/NaN rows from target.")

    if time_series.empty:
        raise ValueError(f"Target time series '{target_col}' is empty after cleaning.")

    print(f"--> Data loading and preparation complete. Final series length: {len(time_series)}")
    return time_series

def create_lagged_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    """
    Creates DataFrame of lagged features (t-1, t-2, ..., t-n) for a time series.

    Args:
        series: Input time series.
        n_lags: Number of lags to create.

    Returns:
        DataFrame with lagged features.

    Purpose (GSoC Context): Demonstrates foundational feature engineering for time series ML.
                         The full GSoC project will expand this into robust, reusable aeon
                         transformers including rolling features, calendar features, etc., crucial
                         for supporting SETAR-Tree, GBM integration, and framework transparency.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input 'series' must be a pandas Series.")
    if not isinstance(n_lags, int) or n_lags < 1:
        raise ValueError("'n_lags' must be a positive integer.")

    series_len = len(series)
    print(f"--> Creating {n_lags} lagged features for series of length {series_len}...")
    if series_len <= n_lags:
         print(f"Warning: Series length ({series_len}) is <= n_lags ({n_lags}). Resulting features will contain only NaNs.")
         # Still return DataFrame structure expected by downstream code
         return pd.DataFrame(index=series.index, columns=[f'lag_{i}' for i in range(1, n_lags + 1)])


    df_features = pd.DataFrame(index=series.index)
    for i in range(1, n_lags + 1):
        df_features[f'lag_{i}'] = series.shift(i)

    print(f"    Lagged features DataFrame created with shape {df_features.shape}.")
    return df_features

def train_evaluate_forecaster(series: pd.Series, n_lags: int, split_ratio: float):
    """
    Performs feature engineering, simple train/test split, trains a placeholder model,
    and evaluates basic forecasting metrics.

    Args:
        series: The target time series.
        n_lags: Number of lags to use.
        split_ratio: Fraction for chronological training split.

    Purpose (GSoC Context): Shows end-to-end conceptual flow. Full GSoC project replaces
                         placeholder model (RandomForest) with implemented SETAR-Tree /
                         (stretch:GBM), uses aeon's rigorous time series cross-validation,
                         standard benchmarks, richer metrics (MASE), and robustness tests.
    """
    # --- Validation (Redundant if called by main, good practice in function) ---
    if not isinstance(series, pd.Series): raise TypeError("'series' must be pd.Series.")
    if not (isinstance(n_lags, int) and n_lags >= 1): raise ValueError("'n_lags' > 0 needed.")
    if not (isinstance(split_ratio, float) and 0 < split_ratio < 1): raise ValueError("0 < 'split_ratio' < 1 needed.")

    # 1. Feature Engineering
    X_features = create_lagged_features(series, n_lags)
    y_target = series # Define target explicitly

    # 2. Combine, Align, Clean (handle NaNs from lagging)
    # Use inner join via index alignment to ensure features and target match, dropping NaN rows
    df_combined = pd.concat([y_target.rename('target'), X_features], axis=1)
    df_clean = df_combined.dropna()
    print(f"--> Combined features and target. Dataset size after NaN removal: {len(df_clean)} samples.")

    if len(df_clean) < n_lags + 2: # Need enough points for lags + at least 1 train + 1 test sample
        raise ValueError(
            f"Not enough non-NaN data ({len(df_clean)}) after creating lags ({n_lags}) "
            "to perform meaningful train/test split. Try fewer lags or more data."
        )

    # 3. Chronological Train/Test Split (NO SHUFFLING)
    print(f"--> Performing chronological train/test split (ratio: {split_ratio:.1%})...")
    split_index = int(len(df_clean) * split_ratio)

    # Ensure split results in non-empty train AND test sets
    if split_index < 1:
        raise ValueError(f"Training split point ({split_index}) too small. Adjust split_ratio or check data.")
    if split_index >= len(df_clean):
         raise ValueError(f"Training split point ({split_index}) leaves no test data. Adjust split_ratio.")

    X_train, X_test = df_clean.iloc[:split_index, 1:], df_clean.iloc[split_index:, 1:] # Select feature columns
    y_train, y_test = df_clean.iloc[:split_index, 0], df_clean.iloc[split_index:, 0]   # Select target column

    print(f"    Split complete: {len(X_train)} train samples, {len(X_test)} test samples.")
    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
         # This check is technically redundant given earlier checks, but belt-and-suspenders
        raise ValueError("Train or test set is unexpectedly empty after splitting.")

    # 4. Train Placeholder Model
    print(f"--> Training placeholder RandomForestRegressor (n_estimators=50, n_jobs=1)...")
    # Using RandomForest: tree-based, common ML baseline. Fixed state & jobs=1 for PoC reproducibility.
    # Full GSoC: Implement SETAR-Tree and evaluate vs. aeon baselines (KNeighbors, Rocket, TSF, Dummy).
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    print("    Model training complete.")

    # 5. Predict on Test Set
    print(f"--> Making predictions on {len(X_test)} test samples...")
    y_pred = model.predict(X_test)

    # 6. Basic Evaluation
    print("--> Evaluating predictions...")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Display Results Clearly
    print("\n--- PoC Evaluation Results (Placeholder Model on Single Split) ---")
    print(f"  Test Set MAE:  {mae:.4f}")
    print(f"  Test Set RMSE: {rmse:.4f}")
    print("-----------------------------------------------------------------")
    print("Note: Actual GSoC project would include SETAR-Tree & GBM implementation,")
    print("      rigorous time series cross-validation, benchmark datasets,")
    print("      standardized metrics (e.g., MASE), and robustness analysis.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parsing for Flexibility ---
    parser = argparse.ArgumentParser(
        description="Aeon GSoC PoC: Basic Time Series Forecasting with ML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    # Input Data Arguments
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILEPATH,
                        help="Path to the input CSV data file.")
    parser.add_argument("--time_col", type=str, default=DEFAULT_TIME_COL,
                        help="Name of the timestamp column.")
    parser.add_argument("--target_col", type=str, default=DEFAULT_TARGET_COL,
                        help="Name of the target value column to forecast.")
    # Optional Filtering Arguments
    parser.add_argument("--filter_col", type=str, default=DEFAULT_FILTER_COL,
                        help="Optional: Column name to filter data by (e.g., sensor ID).")
    parser.add_argument("--filter_val", type=str, default=DEFAULT_FILTER_VAL,
                        help="Optional: Value to keep in filter_col (requires --filter_col).")
    # Modeling Arguments
    parser.add_argument("--n_lags", type=int, default=N_LAGS_DEFAULT,
                        help="Number of past time steps (lags) to use as features.")
    parser.add_argument("--split_ratio", type=float, default=TRAIN_TEST_SPLIT_RATIO_DEFAULT,
                        help="Fraction of data for training set (0 < ratio < 1).")

    args = parser.parse_args()

    print("--- Running Forecasting PoC Script ---")
    print(f"Configuration:")
    print(f"  Data File:      {args.data}")
    print(f"  Time Column:    {args.time_col}")
    print(f"  Target Column:  {args.target_col}")
    if args.filter_col and args.filter_val:
      print(f"  Filtering:      '{args.filter_col}' == '{args.filter_val}'")
    elif args.filter_col or args.filter_val:
      print(f"  Filtering:      DISABLED (requires both --filter_col and --filter_val)")
    else:
       print(f"  Filtering:      DISABLED (no filter args provided)")
    print(f"  Num Lags:       {args.n_lags}")
    print(f"  Train Split:    {args.split_ratio:.1%}")
    print("--------------------------------------")


    try:
        # Execute main workflow
        time_series = load_and_prepare_data(
            filepath=args.data,
            time_col=args.time_col,
            target_col=args.target_col,
            filter_col=args.filter_col,
            filter_val=args.filter_val
        )

        if len(time_series) > args.n_lags + 1: # Need points for lags + >=1 train + >=1 test
            train_evaluate_forecaster(time_series, args.n_lags, args.split_ratio)
        else:
             print(f"\nERROR: Insufficient data ({len(time_series)} points) "
                   f"after loading/cleaning to create {args.n_lags} lags and perform train/test split.")
             print("Consider using fewer lags (--n_lags), a different filter, or more initial data.")

    except (FileNotFoundError, ValueError, TypeError, IOError) as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Message: {e}")
        print("Please check file paths, column names, data format, and parameters.")
        print("---------------------")

    except Exception as e:
        # Catch-all for unexpected errors during development/execution
        print(f"\n--- UNEXPECTED CRITICAL ERROR ---")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print(f"Error: {e}")
        print("-------------------------------")


    print("\n--- PoC Script Finished ---")