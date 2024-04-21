import math
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve

# Read the data
data = pd.read_csv('Rank_Model_Residual_2D.csv', parse_dates=['Date', 'Maturity'])

# Calculate time to maturity in years
data['T'] = (data.Maturity - data.Date) / np.timedelta64(1, 'D') / 365

# Calculate number of coupon payments until maturity
data['N'] = data['T'] * data['Cpn Freq']

# Initialize columns for YTM and Modified Duration
data['YTM'] = 0
data['Modified Duration'] = 0

# Define the function to calculate Macaulay Duration
def macaulay_duration(N, frequency, coupon, ytm):
    mac_duration = 0
    for t in range(1, int(N) + 1):
        mac_duration += t * (coupon / frequency) / ((1 + ytm / frequency) ** (frequency * t))
    return mac_duration

# Calculate YTM for each bond
def get_diff(y, N, frequency, coupon, price):
    Nc = math.floor(N * coupon) / coupon
    discount_factor = 1 / (1 + y / frequency)
    fitted_price = coupon / 2 * discount_factor * (1 - (discount_factor) ** (Nc)) / (1 - discount_factor) + 100 * discount_factor ** N
    return price - fitted_price

ytms = []
for index, row in data.iterrows():
    price = row['Ask Price']
    try:
        price = np.array(price).astype(float)
    except ValueError:
        price = np.nan
    N = row['N']
    frequency = np.array(row['Cpn Freq']).astype(float)
    coupon = np.array(row['Cpn']).astype(float)
    ytm = fsolve(lambda y: get_diff(y, N, frequency, coupon, price), 0.05)
    ytms.append(ytm[0])

# Calculate Modified Duration for each bond
for i, row in data.iterrows():
    ytm = row['YTM']
    frequency = row['Cpn Freq']
    coupon = row['Cpn']
    modified_duration = macaulay_duration(row['N'], frequency, coupon, ytm) / (1 + ytm / frequency)
    data.at[i, 'Modified Duration'] = modified_duration

# Save the updated DataFrame with YTMs and Modified Duration
data.to_csv('Modified_Duration_Master_Pricing_File_04072024.csv', index=False)
