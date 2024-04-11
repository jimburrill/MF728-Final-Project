import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve

data = pd.read_csv('Master_Pricing_File_04072024.csv',parse_dates=['Date','Maturity'])

data['T']= (data.Maturity-data.Date)/np.timedelta64(1, 'D')

data['T']=data['T']/365

data['N']=data['T']*data['Cpn Freq']

data['ytm']=0

def get_diff(y,N,frequency,coupon,price):

    
    discount_factor = 1/(1+y/frequency)
    fitted_price = coupon/2*discount_factor*(1-(discount_factor)**(N))/(1-discount_factor) + 100*discount_factor**N
    
    return price - fitted_price

ytms = []

for index,row in data.iterrows():

    price = row['Ask Price']
    
    try:
        # Try to convert the string to a float
        price = np.array(price).astype(float)
    except ValueError:
        # Handle the case where the conversion fails (e.g., non-numeric value)
        # You might want to do something specific here, like setting price to NaN
        price = np.nan
    
    N = row['N']
    frequency = np.array(row['Cpn Freq']).astype(float)
    coupon = np.array(row['Cpn']).astype(float)
    
    ytm = fsolve(lambda y :get_diff(y,N,frequency,coupon,price),0.05)
    
    ytms.append(ytm)
    
    
print(ytms)
    




# price = 100
# N = 8
# frequency=2
# coupon=5
# y=0.03
# get_diff(y,N,frequency,coupon,price)


# fsolve(lambda y :get_diff(y,N,frequency,coupon,price),0.05)
