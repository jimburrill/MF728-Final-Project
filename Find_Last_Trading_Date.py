import pandas as pd
import pandas_market_calendars as mcal
from datetime import timedelta

def is_trading_date(date):
    return nyse.valid_days(start_date=date, end_date=date).shape[0] > 0

df = pd.read_csv("Master Price Data Jack Pull.csv")
df['Date'] = pd.to_datetime(df['Date'])
unique_dates = pd.to_datetime(df['Date'].unique())
nyse = mcal.get_calendar('NYSE')

mapping_dates = {}

for i in pd.to_datetime(unique_dates):
    init_date = i
    while is_trading_date(i) == False:
        i = i - timedelta(days=1)
    mapping_dates[init_date] = i

df['Last Trading Date'] = df['Date'].replace(mapping_dates)
