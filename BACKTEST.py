import os
os.system('clear')
os.system('clear')


from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')


import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# ---------------------------------------------------
# STEP 1: Download stock data
# ---------------------------------------------------
tickers = [
   'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'NFLX',
   'CRM', 'CSCO', 'QCOM', 'AVGO', 'ORCL', 'ADBE', 'IBM', 'TXN', 'MU', 'SHOP',
   'BA', 'CAT', 'LMT', 'GE', 'F', 'GM', 'NKE', 'SBUX', 'DIS', 'MCD',
   'JPM', 'GS', 'BAC', 'MS', 'WFC', 'C', 'PYPL', 'AXP', 'V', 'MA',
   'T', 'VZ', 'TMUS', 'XOM', 'CVX', 'COP', 'OXY', 'PEP', 'KO', 'WMT'
]

tickers1 = [
    # Top 20 confirmed
    "AAPL","GOOGL","MSFT","AMZN","META","BRK-B","JNJ","JPM","XOM","BAC",
    "WFC","WMT","V","CVX","PG","HD","INTC","VZ","UNH","PFE",

    # Estimated 21–50
    "KO","PEP","ORCL","CSCO","MRK","AMGN","GS","MS","MMM",
    "HON","LMT","MCD","IBM","NKE","CAT","RTX","UNP","SBUX","ABBV",
    "TXN","CVS","MDT","GILD","BLK","AXP","TMO","ADBE","QCOM","BA"
]



raw_data = yf.download(tickers, start="2022-01-01", end= "2023-01-01" , auto_adjust=False)[['Close', 'Volume']]
prices_df = raw_data['Close'].dropna()
volume_df = raw_data['Volume'].dropna()
adv_21 = (prices_df * volume_df).rolling(window=21).mean()


# ---------------------------------------------------
# STEP 2: Calculate daily returns
# ---------------------------------------------------
returns_df = prices_df.pct_change().dropna() * 100
returns_df = returns_df.round(2)
# Full trading calendar (all trading days, not just signal days)
all_days = returns_df.index




# ---------------------------------------------------
# STEP 3: Get top 5 gainers per day (Q = 5)
# ---------------------------------------------------
top_k = 5
top_gainers = {}
for date, row in returns_df.iterrows():
   if max(row) < 4.0:
       continue
   top = row.sort_values(ascending=False).head(top_k).index.tolist()
   top_gainers[date] = top


# ---------------------------------------------------
# STEP 4: Walk-forward backtest
# ---------------------------------------------------
STARTING_CAPITAL = 25000
utilization_log = []  # list to store daily capital usage
capital = STARTING_CAPITAL


returns_log = []
win_count = 0
loss_count = 0
positive_returns = []
negative_returns = []




first_pick_returns = []
second_pick_returns = []


first_pick_profit = []
second_pick_profit = []


first_pick_count = 0
second_pick_count = 0






TRADING_COST = 0.1
window_size = 30
min_support = 3
min_avg_return = 0.5





dates_sorted = sorted(top_gainers.keys())
last_exit_pos = -1  # prevent overlapping holds (index in all_days)

for i in range(window_size, len(dates_sorted)):
   today = dates_sorted[i]  # signal day from your strategy

   # Map the signal day onto the full trading calendar to get real t+1 / t+2
   try:
       pos = all_days.get_loc(today)
   except KeyError:
       continue  # skip if signal day not in calendar (shouldn't happen)

   # Need two future trading days available
   if pos + 2 >= len(all_days):
       break

   entry_day = all_days[pos + 0]  # true next trading day
   exit_day  = all_days[pos + 1]  # true day after next

   # ❌ No overlapping trades: skip if our entry would be before/at the last exit
   if (pos + 1) <= last_exit_pos:
       continue








#    # Only trade between June 1 and July 2, 2025
#    TRADE_START = pd.Timestamp("2024-01-01")
#    TRADE_END = pd.Timestamp("2025-01-02")  # exclusive
#    if not (TRADE_START <= today < TRADE_END):
#        continue






   past_window = dates_sorted[i - window_size:i + 1]
   # --- Mine sequences ---
   sequence_stats = defaultdict(list)
   for j in range(len(past_window) - 1):
       d1, d2 = past_window[j], past_window[j + 1]
       for a in top_gainers[d1]:
           for b in top_gainers[d2]:
               if a != b:
                   sequence_stats[(a, b)].append(returns_df.loc[d2, b])


   # --- Filter strong sequences ---
   profitable_sequences = {}
   for (a, b), rets in sequence_stats.items():
       if len(rets) >= min_support:
           avg_ret = np.mean(rets)
           if min(rets) > 0 and avg_ret >= min_avg_return:
               score = avg_ret * len(rets)
               profitable_sequences[(a, b)] = (b, score)


   # --- Aggregate predictions ---
   today_top = top_gainers[today]
   candidate_counter = Counter()


   for a in today_top:
       for (x, b), (_, score) in profitable_sequences.items():
           if x == a:
               candidate_counter[b] += score


   if not candidate_counter:
       continue


   top_stocks = candidate_counter.most_common(2)  # top 1 and 2


   used_capital = 0
   day_start_capital = capital


   for idx, (stock, score) in enumerate(top_stocks):
       if used_capital >= day_start_capital:
           break
       adv_today = adv_21.loc[today, stock]
       max_position = adv_today * 0.0075


       # Allocate full capital to top stock; leftover goes to second
       available_capital = capital - used_capital
       position = min(available_capital, max_position)


       if position <= 0:
           continue


       ret = returns_df.loc[exit_day, stock] - TRADING_COST
       #print(f"Signal = {today}, Entry = {entry_day}, Exit = {exit_day}, Picked = {stock}, Used Return From {exit_day}")
       profit = position * (ret / 100)
       capital += profit
       used_capital += position
       # Block new entries until after this trade's exit day
       last_exit_pos = all_days.get_loc(exit_day)



       # Capital utilization tracking
       if idx == 1:  # only log once after both trades (top 1 and 2) attempted
           capital_utilized_pct = (used_capital / day_start_capital) * 100
           unused_capital = day_start_capital - used_capital
           utilization_log.append((str(today.date()), round(used_capital, 2), round(unused_capital, 2), round(capital_utilized_pct, 2)))




      










       if idx == 0:
           first_pick_returns.append(ret)
           first_pick_profit.append(profit)
           first_pick_count += 1
       else:
           second_pick_returns.append(ret)
           second_pick_profit.append(profit)
           second_pick_count += 1
















       returns_log.append((str(today.date()), stock, round(ret, 2), round(capital, 2)))


       if ret > 0:
           win_count += 1
           positive_returns.append(ret)
       else:
           loss_count += 1
           negative_returns.append(ret)


       # After 1st stock, only try 2nd if there’s leftover capital
      


# ---------------------------------------------------
# STEP 5: Print results
# ---------------------------------------------------
print("\n📊 First 5 Trades:")
for row in returns_log[:10]:
   print(f"{row[0]} → Buy {row[1]} → Return = {row[2]}% → Capital = ${row[3]}")


print("\n📊 Trade Log (Last 10 Trades):")
for row in returns_log[-10:]:
   print(f"{row[0]} → Buy {row[1]} → Return = {row[2]}% → Capital = ${row[3]}")


print(f"\n📈 Final Capital: ${capital:.2f}")
print(f"📦 Total Trades: {len(returns_log)}")
print(f"✅ Win Rate: {win_count}/{len(returns_log)} = {win_count / len(returns_log) * 100:.2f}%")
print(f"📊 Avg Gain: {np.mean(positive_returns):.2f}%  |  Avg Loss: {np.mean(negative_returns):.2f}%")
net_return = ((capital - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
print(f"⚖️  Net Strategy Return: {net_return:.2f}%")




print("\n📊 First Pick Performance:")
print(f"  Trades: {first_pick_count}")
print(f"  Avg Return: {np.mean(first_pick_returns):.2f}%")
print(f"  Total Profit: ${np.sum(first_pick_profit):,.2f}")


print("\n📊 Second Pick Performance:")
print(f"  Trades: {second_pick_count}")
if second_pick_count > 0:
   print(f"  Avg Return: {np.mean(second_pick_returns):.2f}%")
   print(f"  Total Profit: ${np.sum(second_pick_profit):,.2f}")
else:
   print("  Avg Return: N/A")
   print("  Total Profit: $0.00")


print("\n📊 Capital Utilization Sample (First 5 Days):")
for row in utilization_log[:5]:
   print(f"{row[0]} → Deployed: ${row[1]} | Idle: ${row[2]} | Utilized: {row[3]}%")


print("\n📊 Capital Utilization Sample (Last 5 Days):")
for row in utilization_log[-5:]:
   print(f"{row[0]} → Deployed: ${row[1]} | Idle: ${row[2]} | Utilized: {row[3]}%")



# ===== DEBUG: identify loss drivers in backtest (first pick only) =====
try:
    # returns_log rows: (date, stock, ret%, capital_after)
    bt_rows = []
    equity = STARTING_CAPITAL
    for date, stock, ret, _cap_after in returns_log:
        # We only logged first pick trades in your backtest
        dollar_change = equity * (ret / 100.0)
        equity_after = equity + dollar_change
        bt_rows.append([date, stock, ret, equity, equity_after, dollar_change])
        equity = equity_after

    bt_df = pd.DataFrame(
        bt_rows,
        columns=['Date','Symbol','RetPct','EquityBefore','EquityAfter','DollarChange']
    )
    bt_df.to_csv('backtest_first_pick_path.csv', index=False)

    worst_pct_bt = bt_df.nsmallest(15, 'RetPct')
    worst_dollars_bt = bt_df.nsmallest(15, 'DollarChange')

    print("\n🔎 [Backtest] Worst 15 trades by % return (first pick):")
    print(worst_pct_bt[['Date','Symbol','RetPct','EquityBefore','DollarChange']].to_string(index=False))

    print("\n🔎 [Backtest] Worst 15 trades by $ impact (first pick):")
    print(worst_dollars_bt[['Date','Symbol','RetPct','EquityBefore','DollarChange']].to_string(index=False))

    print("📝 Saved backtest_first_pick_path.csv")
except Exception as _e:
    print("⚠️ Could not build backtest_first_pick_path:", _e)
















# number of trading days in dataset
total_days = len(returns_df)

# number of days with >= 1 stock moving 4%+
days_with_4pct = (returns_df.max(axis=1) >= 4.0).sum()

# percentage
percent_days = (days_with_4pct / total_days) * 100

print(f"Total trading days: {total_days}")
print(f"Days with at least one 4%+ move: {days_with_4pct}")
print(f"That’s {percent_days:.2f}% of all days")





















# per year summary
trades = pd.DataFrame(returns_log, columns=['SignalDate','Symbol','RetPct','EquityAfter'])
trades['SignalDate'] = pd.to_datetime(trades['SignalDate'])
trades = trades.sort_values('SignalDate')
trades['Year'] = trades['SignalDate'].dt.year

yearly = trades.groupby('Year').agg(
    Trades=('Symbol','count'),
    EndEquity=('EquityAfter','last')
).reset_index()

yearly['StartEquity'] = yearly['EndEquity'].shift(1).fillna(STARTING_CAPITAL)
yearly['ReturnPct'] = (yearly['EndEquity']/yearly['StartEquity'] - 1)*100

print("\n📊 Per-Year Performance:")
print(yearly)




trades['Month'] = trades['SignalDate'].dt.to_period('M').astype(str)
by_symbol = trades.groupby('Symbol')['RetPct'].agg(['count','mean']).sort_values('mean')
by_month  = trades.groupby('Month')['RetPct'].mean()
print(by_symbol.head(10))  # worst names
print(by_symbol.tail(10))  # best names
print(by_month.tail(12))   # recent months



spy = yf.download('SPY', start='2018-01-01', end='2025-01-01')

# Flatten if multi-level
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = [c[0] for c in spy.columns]

# Now spy['Close'] is a clean Series
spy['SPYRet'] = spy['Close'].pct_change()

# Merge only the return column
trades = trades.merge(spy[['SPYRet']], left_on='SignalDate', right_index=True, how='left')


print("Correlation(strategy trade returns, same-day SPY):", trades['RetPct'].corr(trades['SPYRet']*100))

# Volatility regime (realized 20d vol)
# Volatility regime (realized 20d vol, annualized)
spy['SPYVol'] = spy['SPYRet'].rolling(20).std() * np.sqrt(252)

# Quantile binning into 4 regimes
vol_bins = pd.qcut(spy['SPYVol'].dropna(), 4, labels=['low','med','high','very_high'])
regime = pd.DataFrame({'VolRegime': vol_bins})

# Align by date and merge with trades
trades = trades.merge(regime, left_on='SignalDate', right_index=True, how='left')

print(trades.groupby('VolRegime')['RetPct'].mean())  # which regimes hurt/help
