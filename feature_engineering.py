import pandas as pd
import numpy as np
import os
import indicator as ind  # Ensure indicator.py is in the same directory


def run_feature_engineering(input_dir="stock_data", output_dir="final_features"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==============================================================================
    # 1. Prepare Market Background Data (SPY & VIX)
    # ==============================================================================
    print("Loading market background data (SPY, VIX)...")
    try:
        spy_df = pd.read_csv(os.path.join(input_dir, "SPY_data.csv"), header=[0, 1], index_col=0, parse_dates=True)
        vix_df = pd.read_csv(os.path.join(input_dir, "^VIX_data.csv"), header=[0, 1], index_col=0, parse_dates=True)
        # Clean multi-level headers
        spy_df.columns = spy_df.columns.get_level_values(0)
        vix_df.columns = vix_df.columns.get_level_values(0)
    except FileNotFoundError:
        print("Error: SPY_data.csv or ^VIX_data.csv not found. Please download data first.")
        return

    # --- A. Calculate SPY Historical Features (For Input) ---
    spy_info = pd.DataFrame(index=spy_df.index)

    # SPY Multi-period Returns (Input Features)
    spy_info['1d_spy_ret'] = np.log(spy_df['Close'] / spy_df['Close'].shift(1))
    spy_info['5d_spy_ret'] = np.log(spy_df['Close'] / spy_df['Close'].shift(5))
    spy_info['10d_spy_ret'] = np.log(spy_df['Close'] / spy_df['Close'].shift(10))
    spy_info['20d_spy_ret'] = np.log(spy_df['Close'] / spy_df['Close'].shift(20))

    # Market Volatility and VIX
    spy_info['spy_vol'] = spy_info['1d_spy_ret'].rolling(20).std()
    spy_info['vix'] = vix_df['Close'] / 100.0  # Normalize VIX

    # --- B. Calculate SPY Future Returns (For Alpha Label Calculation) ---
    # Calculated for: Stock Future Return - SPY Future Return
    spy_future = pd.DataFrame(index=spy_df.index)
    spy_future['5d'] = spy_df['Close'].shift(-5) / spy_df['Close'] - 1
    spy_future['10d'] = spy_df['Close'].shift(-10) / spy_df['Close'] - 1
    spy_future['20d'] = spy_df['Close'].shift(-20) / spy_df['Close'] - 1

    # ==============================================================================
    # 2. Iterate Through Stocks to Calculate Features
    # ==============================================================================
    ticker_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and "SPY" not in f and "VIX" not in f]
    print(f"Processing basic features for {len(ticker_files)} stocks...")

    all_stocks_data = {}

    # Collect future returns for all stocks to calculate cross-sectional standard deviation later
    raw_returns_5d = {}
    raw_returns_10d = {}
    raw_returns_20d = {}

    for f in ticker_files:
        ticker = f.replace("_data.csv", "")
        file_path = os.path.join(input_dir, f)

        try:
            df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
            df.columns = df.columns.get_level_values(0)
        except Exception as e:
            print(f"Failed to read {ticker}: {e}")
            continue

        if len(df) < 250: continue

        # --- Feature Engineering Starts ---
        feat = pd.DataFrame(index=df.index)

        # 1. Price Momentum (Log Returns)
        feat['1d_close_log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        feat['5d_close_log_ret'] = np.log(df['Close'] / df['Close'].shift(5))
        feat['10d_close_log_ret'] = np.log(df['Close'] / df['Close'].shift(10))
        feat['20d_close_log_ret'] = np.log(df['Close'] / df['Close'].shift(20))

        # 2. Volume Momentum
        # Add clip protection to prevent infinite values from data errors
        vol_change = (df['Volume'] + 1e-8) / (df['Volume'].shift(1) + 1e-8)
        feat['1d_vol_log_ret'] = np.log(vol_change).clip(-5, 5)
        feat['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # 3. Candlestick Geometry Features
        feat['k_body'] = (df['Close'] - df['Open']) / df['Open']
        feat['k_upper'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        feat['k_lower'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

        # 4. Moving Average System (EMA Short-term, SMA Long-term)
        ema_df = ind.EMA(df['Close'], [10, 20, 30])
        feat['ema10_gap'] = (df['Close'] - ema_df['EMA_10']) / ema_df['EMA_10']
        feat['ema20_gap'] = (df['Close'] - ema_df['EMA_20']) / ema_df['EMA_20']
        feat['ema30_gap'] = (df['Close'] - ema_df['EMA_30']) / ema_df['EMA_30']

        sma_df = ind.SMA(df['Close'], [20, 60, 100, 200])
        feat['sma60_gap'] = (df['Close'] - sma_df['SMA_60']) / sma_df['SMA_60']
        feat['sma100_gap'] = (df['Close'] - sma_df['SMA_100']) / sma_df['SMA_100']
        feat['sma200_gap'] = (df['Close'] - sma_df['SMA_200']) / sma_df['SMA_200']

        # 5. RSI Multi-period
        rsi_df = ind.RSI(df['Close'], [7, 14, 26])
        feat['rsi7'] = rsi_df['RSI_7'] / 100.0
        feat['rsi14'] = rsi_df['RSI_14'] / 100.0
        feat['rsi26'] = rsi_df['RSI_26'] / 100.0

        # 6. MACD (Position and Momentum)
        macd_df = ind.MACD(df['Close'])
        # Add 1e-8 to prevent division by zero
        feat['macd_dif'] = macd_df['DIF'] / (df['Close'] + 1e-8)
        feat['macd_hist'] = macd_df['MACD'] / (df['Close'] + 1e-8)

        # 7. BOLL (Position and Width)
        boll_df = ind.BOLL(df['Close'], window=20, k=2)
        upper = boll_df['BOLL_Upper']
        lower = boll_df['BOLL_Lower']
        feat['bb_position'] = (df['Close'] - lower) / ((upper - lower) + 1e-8)
        feat['bb_width'] = (upper - lower) / (boll_df['BOLL_Middle'] + 1e-8)

        # 8. OBV (Momentum)
        obv_df = ind.OBV(df['Close'], df['Volume'])
        obv_ma = obv_df['OBV'].rolling(20).mean()
        feat['obv_momentum'] = (obv_df['OBV'] - obv_ma) / (obv_ma.abs() + 1e-8)

        hv_df = ind.HV(df['Close'], [10, 60])

        # (Advanced) Volatility Change: Ratio of current volatility to past 60 days
        # > 1 implies volatility expansion, < 1 implies volatility squeeze
        feat['hv_ratio'] = hv_df['HV_10'] / (hv_df['HV_60'] + 1e-8)

        # 9. Merge Market Background (SPY, VIX)
        feat = feat.join(spy_info, how='left')

        # 10. Calculate Relative Strength (Alpha Features)
        feat['1d_relative_ret'] = feat['1d_close_log_ret'] - feat['1d_spy_ret']
        feat['5d_relative_ret'] = feat['5d_close_log_ret'] - feat['5d_spy_ret']
        feat['10d_relative_ret'] = feat['10d_close_log_ret'] - feat['10d_spy_ret']
        feat['20d_relative_ret'] = feat['20d_close_log_ret'] - feat['20d_spy_ret']

        # 11. Prepare Raw Labels (For Z-Score Calculation)
        # Calculate raw returns here, convert to Alpha Score later
        feat['target_5d_raw'] = df['Close'].shift(-5) / df['Close'] - 1
        feat['target_10d_raw'] = df['Close'].shift(-10) / df['Close'] - 1
        feat['target_20d_raw'] = df['Close'].shift(-20) / df['Close'] - 1

        all_stocks_data[ticker] = feat

        # Collect data for cross-sectional standard deviation calculation
        raw_returns_5d[ticker] = feat['target_5d_raw']
        raw_returns_10d[ticker] = feat['target_10d_raw']
        raw_returns_20d[ticker] = feat['target_20d_raw']

    # ==============================================================================
    # 3. Calculate Cross-Sectional Alpha Score (5d, 10d, 20d)
    # ==============================================================================
    print("Calculating Cross-Sectional Alpha Scores (5d, 10d, 20d)...")

    # Helper: Calculate (Stock - SPY) / Daily Cross-Sectional Std Dev
    def calculate_alpha_score(stock_returns_dict, spy_returns_series):
        if not stock_returns_dict: return None

        df_stock_ret = pd.DataFrame(stock_returns_dict)
        aligned_spy = spy_returns_series.reindex(df_stock_ret.index)

        # Numerator: Excess Returns
        excess_returns = df_stock_ret.sub(aligned_spy, axis=0)

        # Denominator: Daily Market Dispersion (Cross-sectional Std)
        market_dispersion = df_stock_ret.std(axis=1)

        # Result: Alpha Score
        scores = excess_returns.div(market_dispersion, axis=0)

        return scores.clip(-3, 3)  # Clip extreme values

    # Calculate labels for three periods
    label_5d = calculate_alpha_score(raw_returns_5d, spy_future['5d'])
    label_10d = calculate_alpha_score(raw_returns_10d, spy_future['10d'])
    label_20d = calculate_alpha_score(raw_returns_20d, spy_future['20d'])

    if label_20d is None:
        print("Error: Unable to calculate labels. Check data sources.")
        return

    # ==============================================================================
    # 4. Save Final Files
    # ==============================================================================
    print("Saving final feature files...")
    for ticker, feat in all_stocks_data.items():
        # Merge calculated labels back
        if ticker in label_20d.columns:
            feat['label_5d'] = label_5d[ticker]
            feat['label_10d'] = label_10d[ticker]
            feat['label_20d'] = label_20d[ticker]

            # Drop intermediate raw targets to save space, keep only Z-Score Labels
            feat.drop(columns=['target_5d_raw', 'target_10d_raw', 'target_20d_raw'], inplace=True)

            # Drop NaNs (Includes early MA NaNs and late Label NaNs)
            final_df = feat.dropna()

            if not final_df.empty:
                final_df.to_csv(os.path.join(output_dir, f"final_{ticker}.csv"))

    print("✅ Feature engineering completed! Data is ready for training.")


if __name__ == "__main__":
    run_feature_engineering()
