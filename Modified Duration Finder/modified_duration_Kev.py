import numpy as np
import pandas as pd
from datetime import datetime

data = pd.read_csv('Rank_Model_Residual_2D.csv')

print(data)

def year_fraction(date1, date2): # returns decimal difference between two dates in years
    date1 = datetime.strptime(date1, "%m/%d/%Y")
    date2 = datetime.strptime(date2, "%m/%d/%Y")
    delta = date2 - date1
    
    years = delta.days / 365.25
    
    return years

print(year_fraction('12/31/2023', '10/15/2033'))

modified_durations = []

for i in range(len(data.index)):
    years_remaining = year_fraction(data.iloc[i]['Date'], data.iloc[i]['Maturity'])
    coupon = data.iloc[i]['Cpn']/100
    ytm = data.iloc[i]['YTM']
    num = 0
    den = 0

    # final face value payment
    num += years_remaining / ((1+ytm/2)**(years_remaining * 2)) # time of payment, weighted by present value of cash flow
    den += 1/ ((1+ytm/2)**(years_remaining * 2)) # weight (cash flow value) accrues to denominator

    while years_remaining > 0: # iterate through all coupon payments
        num += years_remaining * (coupon * 0.5) / ((1+ytm/2)**(years_remaining * 2)) # time of payment weighted by present value of cash flow
        den += (coupon * 0.5) / ((1+ytm/2)**(years_remaining * 2)) # weight (cash flow value) accrues to denominator
        years_remaining -= 0.5

    mac_duration = num / den
    modified_duration = mac_duration / (1+ytm/2)
    modified_durations.append(modified_duration)

data['Modified_Duration'] = pd.Series(modified_durations, index = data.index)

print(data)
data.to_csv("Rank_Model_Residual_Duration_Kev.csv")
