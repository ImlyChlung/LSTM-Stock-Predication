import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'  # Use standard sans-serif font
plt.rcParams['axes.unicode_minus'] = False


def analyze_performance(file_path='backtest_trades.csv'):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}. Please run backtest.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # We only analyze 'SELL' records because they represent Realized PnL
    trades = df[df['Action'] == 'SELL'].copy()

    if trades.empty:
        print("⚠️ No sell records found. Cannot analyze performance.")
        return

    # 2. Calculate Core Metrics
    total_trades = len(trades)
    winning_trades = trades[trades['PnL_Pct'] > 0]
    losing_trades = trades[trades['PnL_Pct'] <= 0]

    win_rate = len(winning_trades) / total_trades * 100

    # Return Statistics (PnL_Pct)
    avg_return = trades['PnL_Pct'].mean() * 100
    median_return = trades['PnL_Pct'].median() * 100
    max_return = trades['PnL_Pct'].max() * 100
    min_return = trades['PnL_Pct'].min() * 100

    # Average Win vs Average Loss (For Risk/Reward Ratio)
    avg_win = winning_trades['PnL_Pct'].mean() * 100 if not winning_trades.empty else 0
    avg_loss = losing_trades['PnL_Pct'].mean() * 100 if not losing_trades.empty else 0

    # Profit Factor = Gross Profit / |Gross Loss|
    gross_profit = winning_trades['PnL_Pct'].sum()
    gross_loss = abs(losing_trades['PnL_Pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    # Holding Period Analysis
    avg_days_held = trades['Days'].mean()

    # 3. Print Report
    print("\n" + "=" * 40)
    print(f"📊 Trading Performance Report")
    print("=" * 40)
    print(f"Total Trades (Closed): {total_trades}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print("-" * 40)
    print(f"Avg Return:            {avg_return:.2f}%")
    print(f"Median Return:         {median_return:.2f}%")
    print(f"Max Single Win:        {max_return:.2f}%")
    print(f"Max Single Loss:       {min_return:.2f}%")
    print("-" * 40)
    print(f"Avg Win Trade:         +{avg_win:.2f}%")
    print(f"Avg Loss Trade:        {avg_loss:.2f}%")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print(f"Avg Holding Days:      {avg_days_held:.1f} days")
    print("=" * 40)

    # 4. Plot Charts
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Chart 1: Return Distribution Histogram
    sns.histplot(trades['PnL_Pct'] * 100, bins=50, kde=True, ax=axes[0], color='blue')
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_title('Trade Return Distribution')
    axes[0].set_xlabel('Return (%)')
    axes[0].set_ylabel('Count')

    # Chart 2: Cumulative PnL Curve
    # Sort by date to simulate the equity curve of realized trades
    trades_sorted = trades.sort_values(by='Date')
    cumulative_pnl = trades_sorted['PnL_Pct'].cumsum()

    axes[1].plot(trades_sorted['Date'], cumulative_pnl, color='green', linewidth=2)
    axes[1].fill_between(trades_sorted['Date'], cumulative_pnl, 0, alpha=0.1, color='green')
    axes[1].set_title('Cumulative Realized PnL (Sum of %)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Return Points')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_performance()
