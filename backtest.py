import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from train import Config, MultiTaskLSTM

# Set plotting style and font
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'  # Changed to standard font for compatibility
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. Pre-computation (Predictions & Price Alignment)
# ==============================================================================
def precompute_scores_and_prices():
    print("Preparing backtest data (Dual source: Features + Raw Prices)...")
    
    raw_data_dir = "stock_data"
    
    # Check if scaler exists
    if not os.path.exists(Config.scaler_save_path):
        print(f"Error: Scaler not found at {Config.scaler_save_path}")
        return None, None, None
        
    scaler = joblib.load(Config.scaler_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = MultiTaskLSTM(len(Config.feature_cols), Config.hidden_size, Config.num_layers, len(Config.label_cols),
                          Config.dropout).to(device)
    try:
        # Load weights (weights_only=False is needed for full checkpoint dictionaries)
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device, weights_only=False))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    all_scores, all_opens, all_closes = [], [], []
    files = [f for f in os.listdir(Config.feature_dir) if f.endswith('.csv')]
    print(f"Scanning {len(files)} stocks...")

    for file in tqdm(files):
        ticker = file.replace("final_", "").replace(".csv", "")
        feat_path = os.path.join(Config.feature_dir, file)
        
        # Locate raw price file (Handle both naming conventions)
        raw_path = os.path.join(raw_data_dir, f"{ticker}_data.csv")
        if not os.path.exists(raw_path): 
            raw_path = os.path.join(raw_data_dir, f"{ticker}.csv")
        if not os.path.exists(raw_path): continue

        try:
            df_feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            df_raw = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True)
            df_raw.columns = df_raw.columns.get_level_values(0)

            # Find start index
            start_idx_list = df_feat.index.get_indexer([pd.Timestamp(Config.test_start_date)], method='nearest')
            if len(start_idx_list) == 0: continue
            start_idx = start_idx_list[0]
            if start_idx < Config.seq_len: continue

            # Slice data
            df_feat_test = df_feat.iloc[start_idx - Config.seq_len:]
            df_raw_test = df_raw.reindex(df_feat_test.index)
            if len(df_feat_test) <= Config.seq_len: continue

            price_open = df_raw_test['Open'].rename(ticker)
            price_close = df_raw_test['Close'].rename(ticker)

            # Check for missing columns
            missing = [c for c in Config.feature_cols if c not in df_feat_test.columns]
            if missing: continue

            # Scale and Predict
            features = scaler.transform(df_feat_test[Config.feature_cols])
            X_batch, valid_dates = [], []
            num_samples = len(features) - Config.seq_len
            
            for i in range(num_samples):
                X_batch.append(features[i: i + Config.seq_len])
                valid_dates.append(df_feat_test.index[i + Config.seq_len])
            
            if not X_batch: continue

            X_tensor = torch.tensor(np.array(X_batch), dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = model(X_tensor)
                scores = preds[:, 0].cpu().numpy()

            # Shift scores by 1 day (Prediction made today applies to tomorrow)
            score_series = pd.Series(scores, index=valid_dates).shift(1)
            score_series.name = ticker
            
            all_scores.append(score_series)
            all_opens.append(price_open)
            all_closes.append(price_close)
        except Exception:
            continue

    if not all_scores: return None, None, None
    
    df_scores = pd.concat(all_scores, axis=1)
    df_opens = pd.concat(all_opens, axis=1)
    df_closes = pd.concat(all_closes, axis=1)
    
    # Filter by date range
    mask = (df_scores.index >= Config.test_start_date) & (df_scores.index <= Config.test_end_date)
    return df_scores[mask], df_opens.reindex(df_scores[mask].index), df_closes.reindex(df_scores[mask].index)


# ==============================================================================
# 2. Portfolio Class (Supports Dynamic Position Sizing)
# ==============================================================================
class Portfolio:
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}
        self.history = []
        self.trade_log = []

    def get_total_equity(self, current_prices):
        """Calculate total equity (Cash + Market Value of Holdings)"""
        equity = self.cash
        for ticker, info in self.holdings.items():
            price = current_prices.get(ticker)
            if price and not np.isnan(price):
                equity += info['shares'] * price
            else:
                # Fallback to cost price if current price is missing
                equity += info['shares'] * info['cost_price']
        return equity

    def update_holding_days(self):
        for ticker in self.holdings:
            self.holdings[ticker]['days_held'] += 1

    def sell(self, ticker, price, date, reason):
        shares = self.holdings[ticker]['shares']
        proceeds = shares * price
        self.cash += proceeds

        cost = self.holdings[ticker]['cost_price']
        pnl_pct = (price / cost) - 1

        self.trade_log.append({
            'Date': date, 'Action': 'SELL', 'Ticker': ticker,
            'Price': price, 'Shares': shares, 'Reason': reason,
            'PnL_Pct': pnl_pct, 'Days': self.holdings[ticker]['days_held']
        })
        del self.holdings[ticker]

    def buy(self, ticker, price, date, score, trade_amount, rank):
        # Check if sufficient funds exist
        if self.cash < trade_amount:
            # If funds are insufficient for full allocation, use remaining cash (All-in remainder)
            # But if remainder is too small (e.g., < $1000), skip to avoid high fee ratios
            if self.cash > 1000:
                trade_amount = self.cash
            else:
                return False

        shares = int(trade_amount // price)
        if shares == 0: return False

        cost = shares * price
        self.cash -= cost

        self.holdings[ticker] = {
            'shares': shares,
            'entry_date': date,
            'cost_price': price,
            'entry_score': score,
            'days_held': 0
        }

        self.trade_log.append({
            'Date': date, 'Action': 'BUY', 'Ticker': ticker,
            'Price': price, 'Shares': shares, 'Score': f"{score:.2f}",
            'Amount': cost, 'Rank': rank
        })
        return True


# ==============================================================================
# 3. Run Backtest (Compounding Mode)
# ==============================================================================
def run_backtest():
    df_scores, df_opens, df_closes = precompute_scores_and_prices()
    if df_scores is None: 
        print("No scoring data available. Exiting.")
        return

    # --- Strategy Parameter Settings ---
    initial_capital = 100000
    portfolio = Portfolio(initial_cash=initial_capital)

    # Strategy: Max N positions, equal allocation 1/N
    MAX_POSITIONS = 5
    TARGET_ALLOCATION = 1.0 / MAX_POSITIONS  # e.g., 5 positions -> 20% each

    print(f"\nStarting Backtest: Initial ${initial_capital}, Max {MAX_POSITIONS} positions, Allocation {TARGET_ALLOCATION * 100:.1f}% per stock")
    dates = df_scores.index

    for date in tqdm(dates):
        portfolio.update_holding_days()

        daily_opens = df_opens.loc[date]
        daily_closes = df_closes.loc[date]
        daily_scores = df_scores.loc[date]

        # --- A. Sell Logic (Check every 7 days) ---
        current_holdings = list(portfolio.holdings.keys())
        for ticker in current_holdings:
            days_held = portfolio.holdings[ticker]['days_held']

            # Check periodically (Day 7, 14, 21...)
            if days_held > 0 and days_held % 7 == 0:
                current_score = daily_scores.get(ticker, -999)
                if current_score < 0:  # Score weakened
                    open_price = daily_opens.get(ticker)
                    if open_price and not np.isnan(open_price):
                        portfolio.sell(ticker, open_price, date, reason=f"Day {days_held}: Score < 0")

        # --- B. Buy Logic (Dynamic Position Sizing) ---
        # 1. Calculate available slots
        current_positions = len(portfolio.holdings)
        slots_available = MAX_POSITIONS - current_positions

        if slots_available > 0:
            # 2. Calculate total equity (Cash + Holdings)
            total_equity = portfolio.get_total_equity(daily_opens)

            # 3. Dynamic Trade Size Calculation (Core of Compounding!)
            # Trade Amount = Total Equity * Target %
            # Example: Equity 400k * 20% = 80k per trade
            target_trade_size = total_equity * TARGET_ALLOCATION

            # 4. Stock Selection (Find score > 0.3 and sort descending)
            candidates = daily_scores[daily_scores > 0.3].sort_values(ascending=False)

            bought_count = 0
            rank_counter = 1

            for ticker, score in candidates.items():
                if bought_count >= slots_available: break  # Max positions reached
                if ticker in portfolio.holdings: continue  # Already holding

                open_price = daily_opens.get(ticker)
                if open_price and not np.isnan(open_price):
                    success = portfolio.buy(ticker, open_price, date, score, target_trade_size, rank_counter)
                    if success:
                        bought_count += 1

                rank_counter += 1

        # --- C. Settlement / Daily Stats ---
        equity = portfolio.get_total_equity(daily_closes)
        portfolio.history.append({'Date': date, 'Equity': equity, 'Cash': portfolio.cash})

    # ==============================================================================
    # 4. Analyze Results
    # ==============================================================================
    results = pd.DataFrame(portfolio.history).set_index('Date')
    trades = pd.DataFrame(portfolio.trade_log)

    # --- Load Benchmark (SPY) ---
    try:
        spy_path = os.path.join("stock_data", "SPY_data.csv")
        # Key Fix 1: Handle MultiIndex headers from yfinance
        spy = pd.read_csv(spy_path, header=[0, 1], index_col=0, parse_dates=True)
        spy.columns = spy.columns.get_level_values(0)  # Flatten headers

        # Key Fix 2: Align dates and fill gaps (ffill/bfill)
        spy = spy.reindex(results.index)
        spy = spy.ffill().bfill()

        # Calculate SPY Equity Curve (Buy & Hold Simulation)
        initial_spy_price = spy['Close'].iloc[0]
        if pd.isna(initial_spy_price) or initial_spy_price == 0:
            raise ValueError("SPY initial price is empty")

        spy_cum = (spy['Close'] / initial_spy_price) * portfolio.initial_cash
        spy_ret_pct = (spy_cum.iloc[-1] - portfolio.initial_cash) / portfolio.initial_cash * 100

    except Exception as e:
        print(f"⚠️ Failed to load SPY benchmark: {e}")
        # If failed, plot a flat line
        spy_cum = pd.Series(portfolio.initial_cash, index=results.index)
        spy_ret_pct = 0.0

    final_equity = results['Equity'].iloc[-1]
    ret_pct = (final_equity - portfolio.initial_cash) / portfolio.initial_cash * 100

    print("\n" + "=" * 40)
    print(f"Backtest Period: {results.index[0].date()} -> {results.index[-1].date()}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {ret_pct:.2f}% (SPY: {spy_ret_pct:.2f}%)")
    print(f"Total Trades: {len(trades)}")
    print("=" * 40)

    if not trades.empty:
        trades.to_csv("backtest_trades.csv")
        print("Detailed trade log saved to backtest_trades.csv")

    # Plotting
    plt.figure(figsize=(12, 6))
    # Strategy Curve (Blue)
    plt.plot(results.index, results['Equity'], label='AI Strategy (Compound)', color='blue', linewidth=2)
    # SPY Benchmark (Gray Dashed)
    plt.plot(results.index, spy_cum, label=f'S&P 500 (+{spy_ret_pct:.0f}%)', color='gray', linestyle='--')

    plt.title(f'AI Compounding Strategy Backtest (Total Return: {ret_pct:.0f}%)')
    plt.ylabel('Account Equity ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_backtest()
