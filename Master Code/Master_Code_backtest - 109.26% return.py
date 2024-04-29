import pandas as pd
import numpy as np
import time
from scipy.optimize import fsolve
import math
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
start_time = time.time()
bond_ratings = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC+": 20,
    "CC": 21,
    "CC-": 22,
    "C+": 23,
    "C": 24
}

SLOPE_COEFFICIENT = 0.5
# steepens regression line's slope by this factor before calculating distance to line, thus taking rank into account
# 0 < SLOPE_COEFFICIENT < inf
# 0: prefers lower rank
# 1: doesn't look at rank at all, only residual
# inf: looks at rank only, no residual consideration whatsoever

#returns with various values of SLOPE_COEFFICIENT:
# 0.5 - all filters kept in model_rank method - 77.37%
# 0.5 - removed Bond_Rating and Ispread filters in model_rank method - 109.26%
# 0.6 - 74.81%
# 0.8 - 75.19%
# 1 - 42.23 % 
# 2 - 40.07 %       
# 20 - 23.8% 

def coupon_payment_dates(single_bond_data_df, oldest_date):
    '''Take in a dataframe contianing the data for a single bond. Calculate all of the coupon payment dates starting at
       the maturity date, until the begining of the data frame'''

    maturity_date = pd.to_datetime(single_bond_data_df['Maturity'].iloc[0])
    payment_dates = [maturity_date]  #Get maturity date of bond
    num_payments_per_year = single_bond_data_df['Cpn Freq'].iloc[0]     #Get number of coup payments per year
    yrs_to_matur = (maturity_date - pd.to_datetime(oldest_date)).days / 365 #Calculate years to maturity
    num_payments = math.ceil(yrs_to_matur * num_payments_per_year)  #Calculate total number of payments

    payment_dates = pd.date_range(end=maturity_date, periods=num_payments, freq=f'{int(365/num_payments_per_year)}D')

    return payment_dates

def calc_single_bond_return(single_bond_data_df):
    '''Pass in the bond data for a single bond for every quarter it is in the investible universe. Calculate the return
       of the bond between quarters. Returns the weighted returns of the bond per quarter in percentages'''

    #Find the coupon payment dates for the bond
    coupon_dates_all = coupon_payment_dates(single_bond_data_df, single_bond_data_df['Date'].iloc[0])


    all_quarter_returns = [0] * len(single_bond_data_df)
    for i in range(1, len(single_bond_data_df['Date'])):
        #Select the coup dates that occured this Quarter
        coupon_dates = coupon_dates_all[(coupon_dates_all <= pd.to_datetime(single_bond_data_df['Date'].iloc[i])) & (coupon_dates_all >= pd.to_datetime(single_bond_data_df['Date'].iloc[i-1]))]
        #Assume 100 face for coupon payments
        coupon_rate = (single_bond_data_df['Cpn'].iloc[0] / single_bond_data_df['Cpn Freq'].iloc[0]) / 100    #Want rate in decimal form
        coupon_cashflows = 100 * coupon_rate * len(coupon_dates)

        #Check if matures between quarters
        if (single_bond_data_df['Date'].iloc[i] >= single_bond_data_df['Maturity'].iloc[i-1]) and single_bond_data_df['Date'].iloc[i-1] <= single_bond_data_df['Maturity'].iloc[i-1]:
            total_proceeds = coupon_cashflows + 100      #100 is FV of the bond
            last_price = single_bond_data_df['Ask Price'].iloc[i-1]
            total_return = (total_proceeds - last_price) / last_price
            all_quarter_returns[i] = total_return

        else:
            current_price = single_bond_data_df['Bid Price'].iloc[i]
            last_price = single_bond_data_df['Ask Price'].iloc[i-1]
            total_proceeds = coupon_cashflows + current_price
            total_return = (total_proceeds - last_price) / last_price
            all_quarter_returns[i] = total_return

    single_bond_data_df['Returns'] = all_quarter_returns
    single_bond_data_df['Weighted Returns'] = single_bond_data_df['Returns'] * single_bond_data_df['Weights'].shift(1)
    single_bond_data_df['Weighted Returns'].iloc[0] = 0
    single_bond_data_df['Shifted Weights'] = single_bond_data_df['Weights'].shift(1)
    return single_bond_data_df

def get_diff(y,N,frequency,coupon,price):
    Nc = math.floor(N*coupon)/coupon
    discount_factor = 1/(1+y/frequency)
    fitted_price = coupon/2*discount_factor*(1-(discount_factor)**(Nc))/(1-discount_factor) + 100*discount_factor**N
    return price - fitted_price

def get_spot_rate(date, years_till_maturity, df):
    row = df.loc[df['Date'] == date]
    if row.empty:
        raise ValueError(f"Date {date} not found in the data.")

    maturities = df.columns[1:].astype(float).values

    if years_till_maturity> 100:
        return row[str(100)].values[0]/100
    elif years_till_maturity < .5:
        return row[str(.5)].values[0]/100
    elif years_till_maturity in maturities:
        return row[str(years_till_maturity)].values[0]/100
    else:
        lower_maturity = max(m for m in maturities if m <= years_till_maturity)
        upper_maturity = min(m for m in maturities if m >= years_till_maturity)

    lower_rate = row[str(lower_maturity)].values[0]
    upper_rate = row[str(upper_maturity)].values[0]

    slope = (upper_rate - lower_rate) / (upper_maturity - lower_maturity)
    spot_rate = lower_rate + slope * (years_till_maturity - lower_maturity)
    return spot_rate/100

def year_fraction(date1, date2):
    #difference two dates in years, as a decimal
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    delta = date2 - date1
    years = delta.days / 365.25

    return years
"""
Cleaning Price Data
"""
def cleaning_price_data():
    price_cols = ['Bid Price','Ask Price','Last Price']
    income_cols = ["Free Cash Flow","Net Income","EBIDTA"]
    df = pd.read_csv("Master_Pricing_File.csv",na_values=["#N/A"])
    df[price_cols] = df[price_cols].astype(float)
    df.drop_duplicates(inplace=True)
    df['CUSIP'] = df['CUSIP'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by="Date",inplace=True)
    df.dropna(subset=income_cols, how='all',inplace=True)
    df.dropna(subset=["Short Debt","Long Debt"], how='all',inplace=True)
    df.dropna(subset=["Revenue"], how='all',inplace=True)
    df.dropna(subset=price_cols, how='any',inplace=True)
    df['Mid Price'] = (df['Bid Price'] + df['Ask Price']) / 2
    df['Bid-Ask-Spread'] = (df['Bid Price'] - df['Ask Price']) / df['Mid Price']

    """
    Adding Filters
    """
    df = df[df['Bid-Ask-Spread'] > -.03]
    dfs = []
    for cusip_i in df['CUSIP'].unique():
        temp_df = df[df['CUSIP'] == cusip_i]
        x = (temp_df['Date'] - temp_df['Date'].shift(-1)).dt.days.fillna(0).astype(int)
        x = x.reset_index(drop=True)
        if len(x[x< -99]) > 0:
            index_start = x[x< -99].index[-1]
            temp_df = temp_df.iloc[index_start+1:]
        if len(temp_df) > 7:
            dfs.append(temp_df)

    pd.concat(dfs).to_csv("Cleaned_Master_Pricing.csv",index=False)
cleaning_price_data()

"""
YTM, Modified Duration & I-Spread Calc
"""
def Calc_ISpread():
    spot_curve_df = pd.read_csv("HQM_Curves.csv")
    spot_curve_df['Date'] = pd.to_datetime(spot_curve_df['Date'])

    bond_df = pd.read_csv("Cleaned_Master_Pricing.csv")
    bond_df['Date'] = pd.to_datetime(bond_df['Date'])
    bond_df['Maturity'] = pd.to_datetime(bond_df['Maturity'])
    bond_df['Years till Maturity'] = (bond_df['Maturity'] - bond_df['Date']).dt.days / 365.25
    bond_df['Spot Rate'] = np.nan
    bond_df['Spot Rate'] = bond_df.apply(lambda x: get_spot_rate(x['Date'], x['Years till Maturity'], spot_curve_df), axis=1)
    bond_df['N']=bond_df['Years till Maturity']*bond_df['Cpn Freq']
    bond_df['YTM']=0
    ytms = []
    for index,row in bond_df.iterrows():
        price = row['Ask Price']
        try:
            price = np.array(price).astype(float)
        except ValueError:
            price = np.nan
        N = row['N']
        frequency = np.array(row['Cpn Freq']).astype(float)
        coupon = np.array(row['Cpn']).astype(float)
        ytm = fsolve(lambda y :get_diff(y,N,frequency,coupon,price),0.05)
        ytms.append(ytm[0])
    bond_df['YTM'] = ytms
    bond_df['I-Spread'] = bond_df['YTM'] - bond_df['Spot Rate']
    bond_df['Bond Rating'] = bond_df['BBG Composite'].map(bond_ratings)
    bond_df.drop(['N'], axis=1,inplace=True)
    modified_durations = []
    for i in range(len(bond_df.index)):
        years_remaining = bond_df['Years till Maturity'].iloc[i]
        coupon = bond_df.iloc[i]['Cpn']/100
        ytm = bond_df.iloc[i]['YTM']
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

    bond_df['Modified_Duration'] = modified_durations
    bond_df.to_csv('Ispread_Master.csv', index=False)
Calc_ISpread()

"""
Model Ranking
"""
def model_rank():
    df = pd.read_csv("Ispread_Master.csv")
    #df = df[df['Bond Rating'] < 11]
    #df = df[df['I-Spread'] < .03]
    df = df[df['Years till Maturity'] < 40]
    profitable_cols = ["NI_Rev","FCF_Rev"]
    leverage_cols = ["FCF_Debt","NI_Debt","EBIDTA_Debt"]
    coverage_cols = ["FCF_IR_Exp","NI_IR_Exp","EBIDTA_IR_Exp"]

    df['FCF_Debt'] = df['Free Cash Flow'] / (df['Short Debt'] + df['Long Debt'])
    df['NI_Debt'] = df['Net Income'] / (df['Short Debt'] + df['Long Debt'])
    df['EBIDTA_Debt'] = df['EBIDTA'] / (df['Short Debt'] + df['Long Debt'])
    df['FCF_IR_Exp'] = df['Free Cash Flow'] / df['Interest Expense']
    df['NI_IR_Exp'] = df['Net Income'] / df['Interest Expense']
    df['EBIDTA_IR_Exp'] = df['EBIDTA'] / df['Interest Expense']
    df['FCF_Rev'] = df['Free Cash Flow'] / df['Revenue']
    df['NI_Rev'] = df['Net Income'] / df['Revenue']
    df.rename(columns={"Years till Maturity":"T","I-Spread":"Ispread"},inplace=True)
    list_dfs = []
    for date in df['Date'].unique():
        temp_df = df[df['Date'] == date]
        x = list(temp_df.iloc[:,-8:].columns)
        x.append("BBG Ticker")
        ratios_temp_df = temp_df[x].drop_duplicates()
        rank_temp_df = ratios_temp_df.iloc[:,:-1].rank(pct=True)
        """
        testing
        """
        rank_temp_df['AVG_Profitablity'] = rank_temp_df[profitable_cols].apply(lambda row: np.nanmean(row), axis=1)
        rank_temp_df['AVG_Leverage'] = rank_temp_df[leverage_cols].apply(lambda row: np.nanmean(row), axis=1)
        rank_temp_df['AVG_Coverage'] = rank_temp_df[coverage_cols].apply(lambda row: np.nanmean(row), axis=1)
        rank_temp_df['AVG_Rank'] = rank_temp_df[['AVG_Profitablity','AVG_Leverage','AVG_Coverage']].apply(lambda row: np.nanmean(row), axis=1)
        rank_temp_df['NaN Count'] = rank_temp_df.iloc[:,-3:].isna().sum(axis=1)
        rank_temp_df['BBG Ticker'] = temp_df['BBG Ticker']
        rank_temp_df['AVG_Rank'][rank_temp_df['NaN Count'] > 1] = np.nan
        temp_df['Model_Rank'] = np.nan
        x = temp_df.merge(rank_temp_df[['BBG Ticker','AVG_Rank']], how='left', left_on='BBG Ticker', right_on='BBG Ticker').iloc[:,-1]
        temp_df['Model_Rank'] = x.rank(pct=True).values
        temp_df = temp_df[temp_df['Ispread'].notna()]
        temp_df = temp_df[temp_df['Model_Rank'].notna()]
        regress_temp_df = temp_df[["Ispread","Model_Rank"]]
        regress_temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        regress_temp_df.dropna(inplace=True)
        y, X = dmatrices("Ispread ~ + Model_Rank", data=regress_temp_df, return_type='dataframe')
        ols_model = sm.OLS(y, X)
        ols_model = ols_model.fit()
        temp_df['Model Residual'] = ols_model.resid.values
        list_dfs.append(temp_df)

    df = pd.concat(list_dfs)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by="Date",inplace=True)
    dfs = []
    for cusip_i in df['CUSIP'].unique():
        temp_df = df[df['CUSIP'] == cusip_i]
        x = (temp_df['Date'] - temp_df['Date'].shift(-1)).dt.days.fillna(0).astype(int)
        x = x.reset_index(drop=True)
        if len(x[x< -99]) > 0:
            index_start = x[x< -99].index[-1]
            temp_df = temp_df.iloc[index_start+1:]
        if len(temp_df) > 7:
            dfs.append(temp_df)

    pd.concat(dfs).to_csv("Model_Rank_Master.csv",index=False)
model_rank()

"""
Optimization
"""
class Optimizer:
    ''' Built to work with data from one date. Slice data on one data first, then use it to instantiate the optimizer
    '''
    global SLOPE_COEFFICIENT

    def __init__(self, data, index_duration):
            self.data = data
            self.index_duration = index_duration
        
            col = self.data['Ispread']
            self.data['Ispread_scaled'] = (col - col.mean()) * (self.data['Model_Rank'].std()/col.std()) + col.mean() # subtract out mean, fix standard deviation, add mean back
            # make 'Ispread' have the same standard deviation as 'Model_rank,' equalizing importance. Can be optimizer later through backtest

    def run_regression(self):
        '''
        returns slope and intercept of regression of Ispread (y) vs Model_Rank (x)
        y = mx + b
        '''
        y = self.data['Ispread_scaled']
        X = self.data[['Model_Rank']].copy()
        X['Intercept'] = 1 # create intercept column
        model = sm.OLS(y, X)
        results = model.fit()
        m = results.params.loc['Model_Rank']
        b = results.params.loc['Intercept']
        return m, b

    def distance(self, p, q, m, b):
        '''
        computes orthogonal distance from point (p,q) to the line y = mx + b
        using formula (mp - q + b)/sqrt(m^2+1)
        '''
        return np.abs(m*p - q + b) / math.sqrt(m**2 + 1)

    def compute_distances(self):
        '''
        calculates orthogonal distance from each bond to regression line
        Appends two columns to the global dataframe, Distance and Filtered_Distance, then returns the dataframe
        will be selecting bonds with the maximum Filtered_Distance.
        '''

        if 'Distance' in self.data.columns:
            print("Distances already computed. Don'make me compute it again.")
            return self.data

        m, b = self.run_regression()
        distances = []
        for i in range(len(self.data.index)):
            x = self.data.iloc[i]['Model_Rank']
            y = self.data.iloc[i]['Ispread_scaled']
            distance = self.distance(x, y, m * SLOPE_COEFFICIENT, b) # STEEPEN line before calculating distance
            distances.append(distance)
        self.data['Distance'] = pd.Series(distances, index = self.data.index) # append column

        mask = (self.data['Model Residual'] > 0)# & (self.data['Model_Rank'] > 0.5)
        self.data['Filtered_Distance'] = np.where(mask, self.data['Distance'], 0)
        # ^^ if bond's rank is below 0.5 or under the regression line, ignore it by setting its distance to 0
        return self.data

    def get_weights(self):
        ''' trade the bonds with the greatest distance to line, matching duration
        '''
        # Use optimizer like last time
        start_time = time.time()
        if 'Distance' not in self.data.columns:
            self.compute_distances()

        mask = (self.data['Filtered_Distance'] != 0) & (self.data['Tradable'] == 0)
        # mask = self.data['Filtered_Distance'] != 0
        sliced = self.data[mask]
        # mask2 = self.data['Tradable'] != 1000
        # sliced = self.data[mask2]

        distances = sliced['Filtered_Distance']
        durations = sliced['Modified_Duration']
        tickers = sliced['Ticker'].values


        # print(time.time() - start_time, "Time to run compute_distance()")

        def objective_function(weights):
            return -np.dot(weights, distances)

        def weight_sum_constraint(weights):
            return np.sum(weights) - 1  # Sum of weights should be 1

        def duration_constraint(weights):
            return np.dot(weights, durations) - self.index_duration  # Portfolio duration should match target duration

        def max_company_constraint(weights):
            return 0.05 - np.max(np.bincount(pd.factorize(tickers)[0], weights=weights))

        def weight_limit_constraint(weights):
            return 0.020 - weights  # All weights should be under 0.05

        # Initial guess for weights
        initial_guess = np.ones(len(distances)) / len(distances)
        # initial_guess = np.zeros(len(distances))
        bounds = [(0, 1)] * len(distances) # all weights positive

        # Define constraints
        constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                      {'type': 'eq', 'fun': duration_constraint},
                      {'type': 'ineq', 'fun': weight_limit_constraint},
                      {'type': 'ineq', 'fun': max_company_constraint})

        start_time = time.time()

        result = minimize(fun=objective_function,
                          x0=initial_guess,
                          method = 'SLSQP',
                          constraints=constraints,
                          tol  = 0.00001, # decreaing tolerance -> better results but more runtime
                          bounds = bounds,
                          options = {'maxiter': 10000000}) # increase number to increase accuracy of optimizer

        # Extract optimal weights
        weights = result.x
        weights = [w if w > 0.005 else 0 for w in weights] # remove negligible weights

        # Map weights back to original dataframe
        original_weights = np.zeros(len(self.data))
        original_weights[mask] = weights

        # print(time.time() - start_time, "Time to run optimizer")
        self.data['Weights'] = original_weights
        return original_weights

    def get_data(self):
        return self.data
    
def optimize_weights():
    dfs = []
    data = pd.read_csv('Model_Rank_Master.csv')
    data['Tradable'] = 0
    index_data = pd.read_excel('LUACTRUU Index.xlsx')
    data['Date'] = pd.to_datetime(data['Date']) #normalize dates
    index_data['Date'] = pd.to_datetime(index_data['Date']) #normalize dates

    data.sort_values(by="CUSIP",inplace=True)
    data.sort_values(by="Date",inplace=True)
    for date_i in data['Date'].unique():
        temp_df = data[data['Date'] == date_i]
        if date_i == data['Date'].unique()[-1]:
            dfs.append(temp_df)
        else:
            bad_cusips = []
            for cusip_i in temp_df['CUSIP'].unique():
                temp_cusip_df = data[data['CUSIP'] == cusip_i]
                if temp_cusip_df['Date'].iloc[-1] == date_i:
                    x = temp_cusip_df[temp_cusip_df['Date'] == date_i].index.values[0]
                    temp_df['Tradable'].loc[x] = 10000

            index_date = '' # gping to locate row in index_data to look up index modified duration
            for i in range(len(index_data.index)):
                if abs(year_fraction(date_i, index_data.iloc[i]['Date'])) < 1/48: # find index row within 0.25 months of selected date
                    index_date = index_data.iloc[i]['Date']
                    break

            row = index_data[index_data['Date'] == index_date]
            index_duration = row.iloc[0]['Modified Duration'] # index duration is here
            optimizer = Optimizer(temp_df, index_duration) # create optimizer using data on that date, and the index duration
            weights = optimizer.get_weights()
            dfs.append(optimizer.get_data())

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv("Optimized_Weights_Master.csv",index=False)
optimize_weights()

"""
Calc Bond Returns
"""
def Bond_Returns():
    df = pd.read_csv('Optimized_Weights_Master.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    unique_cusip = df['CUSIP'].unique()
    #Select columns from existing dataframe
    combined_returns_df = pd.DataFrame({
        'Date': [],
        'CUSIP': [],
        'Returns': [],
        'Weighted Returns': [],
        'Shifted Weights': []
    })

    dfs = []
    #Loop over each unique bond and calculate its quarterly returns
    for i in range(0, len(unique_cusip)):
        cusip = unique_cusip[i]
        filtered_df = df[df['CUSIP'] == cusip]
        bond_returns = calc_single_bond_return(filtered_df)
        dfs.append(bond_returns)
    pd.concat(dfs).to_csv("Master_Backtesting.csv",index=None)
Bond_Returns()

"""
Backtest
"""
def Backtest():
    df = pd.read_csv("Master_Backtesting.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    # df = df[rel_cols]
    df.sort_values(by="CUSIP",inplace=True)
    df.sort_values(by="Date",inplace=True)
    holding_df = df[df['Weights'] > 0]

    portfolio_returns = [0]
    for date_i in holding_df['Date'].unique():
        temp_df = holding_df[holding_df['Date'] == date_i]
        portfolio_returns_date_i = 0
        for cusip_i in temp_df['CUSIP'].unique():
            temp_cusip_df = df[df['CUSIP'] == cusip_i]
            weight_at_t0 = temp_cusip_df["Weights"][temp_cusip_df['Date'] == date_i]
            weight_at_t0_index = weight_at_t0.index.values[0]
            return_from_t0_t1 = temp_cusip_df['Returns'].loc[weight_at_t0_index+1]
            date_check = temp_cusip_df['Date'].loc[weight_at_t0_index+1] - temp_cusip_df['Date'].loc[weight_at_t0_index]
            if date_check.days > 100:
                raise Exception("ERROR: Forward Returns missing", temp_cusip_df, date_i)
            portfolio_returns_date_i += weight_at_t0.values[0]*return_from_t0_t1
        portfolio_returns.append(portfolio_returns_date_i)
    portfolio_returns_df = pd.DataFrame({"Date": df['Date'].unique(), "Portfolio Returns":portfolio_returns})
    portfolio_returns_df['Cumulative Return'] = (1+portfolio_returns_df['Portfolio Returns']).cumprod()-1
    portfolio_returns_df.to_csv("Master_Portfolio_Returns.csv",index=None)
    print("Portfolio Return: ",str(round(portfolio_returns_df['Cumulative Return'].iloc[-1]*100,2))+"%")
    print("Time to run script: ",round(time.time() - start_time,3))
    print("")
Backtest()


###
