import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve

# Read the finance bonds data from the CSV file into a pandas DataFrame
data = pd.read_csv('Rank_Model_Residual_2D.csv', parse_dates=['Date', 'Maturity'])

# Calculate time to maturity in years for each bond
data['T'] = (data['Maturity'] - data['Date']) / np.timedelta64(1, 'D') / 365

# Calculate the number of coupon payments until maturity
data['N'] = data['T'] * data['Cpn Freq']

# Define a function to calculate the present value of cash flows for a given yield
def present_value(y, N, frequency, coupon):
    discount_factor = 1 / (1 + y / frequency)
    present_value = sum(coupon / frequency / (1 + y / frequency) ** (t / frequency) for t in range(1, int(N * frequency) + 1)) + 100 / (1 + y / frequency) ** (N * frequency)
    return present_value

# Calculate the modified duration for each bond
def calculate_modified_duration(y, N, frequency, coupon, price):
    present_val = present_value(y, N, frequency, coupon)
    modified_duration = sum((t * coupon / frequency / (1 + y / frequency) ** (t / frequency)) / present_val for t in range(1, int(N * frequency) + 1)) + (N * 100 / (1 + y / frequency) ** (N * frequency)) / present_val
    return modified_duration

modified_durations = []

for index, row in data.iterrows():
    ytm = row['YTM']
    N = row['N']
    frequency = row['Cpn Freq']
    coupon = row['Cpn']
    price = row['Ask Price']

    # Calculate modified duration using the bond's YTM
    modified_duration = calculate_modified_duration(ytm, N, frequency, coupon, price)
    modified_durations.append(modified_duration)

# Add the calculated modified durations as a new column to the dataset
data['Modified Duration'] = modified_durations

# Save the updated DataFrame with Modified Duration values to a new CSV file
data.to_csv('Rank_Model_Residual_2D_with_Modified_Duration.csv', index=False)
