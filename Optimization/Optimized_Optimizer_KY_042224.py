import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import math
from datetime import datetime
import time
#from joblib import Parallel, delayed

class Optimizer:
    ''' Built to work with data from one date. Slice data on one data first, then use it to instantiate the optimizer
    '''
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
            distance = self.distance(x, y, m, b)
            distances.append(distance)
        self.data['Distance'] = pd.Series(distances, index = self.data.index) # append column

        mask = (self.data['Model_Rank'] > 0.5) & (self.data['Model Residual'] > 0)
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

        mask = self.data['Filtered_Distance'] != 0
        sliced = self.data[mask]

        distances = sliced['Filtered_Distance']
        durations = sliced['Modified_Duration']
        tickers = sliced['Ticker'].values

        print(time.time() - start_time, "Time to run compute_distance()")

        def objective_function(weights):
            return -np.dot(weights, distances)

        def weight_sum_constraint(weights):
            return np.sum(weights) - 1  # Sum of weights should be 1

        def duration_constraint(weights):
            return np.dot(weights, durations) - self.index_duration  # Portfolio duration should match target duration

        def max_company_constraint(weights):
            return 0.05 - np.max(np.bincount(pd.factorize(tickers)[0], weights=weights))

        def weight_limit_constraint(weights):
            return 0.025 - weights  # All weights should be under 0.05

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
                          tol = 0.01, # decreaing tolerance -> better results but more runtime
                          bounds = bounds,
                          options = {'maxiter': 10000}) # increase number to increase accuracy of optimizer

        # Extract optimal weights
        weights = result.x
        weights = [w if w > 0.0001 else 0 for w in weights] # remove negligible weights

        # Map weights back to original dataframe
        original_weights = np.zeros(len(self.data))
        original_weights[mask] = weights

        print(time.time() - start_time, "Time to run optimizer")
        self.data['Weights'] = original_weights
        return original_weights

    def get_data(self):
        return self.data

def year_fraction(date1, date2): # computes difference between two dates in years, as a decimal
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    delta = date2 - date1
    years = delta.days / 365.25

    return years

if __name__ == "__main__":
    print(year_fraction('03/31/2023', '10/15/2033')) # testing year fraction method

    data = pd.read_csv('Rank_Model_Residual_Duration.csv')
    index_data = pd.read_excel('LUACTRUU Index.xlsx')
    data['Date'] = pd.to_datetime(data['Date']) #normalize dates
    index_data['Date'] = pd.to_datetime(index_data['Date']) #normalize dates

    date_i = data['Date'].unique()[-7]
    data = data[data['Date'] == date_i]
    print(data)

    index_date = '' # gping to locate row in index_data to look up index modified duration
    for i in range(len(index_data.index)):
        if abs(year_fraction(date_i, index_data.iloc[i]['Date'])) < 1/48: # find index row within 0.25 months of selected date
            index_date = index_data.iloc[i]['Date']
            break

    row = index_data[index_data['Date'] == index_date]
    index_duration = row.iloc[0]['Modified Duration'] # index duration is here
    del index_data

    optimizer = Optimizer(data, index_duration) # create optimizer using data on that date, and the index duration
    weights = optimizer.get_weights()
    #temp_data['Weights'] = weights
    #temp_data.to_csv("optimized_"+str(date_i)[:10]+".csv")

    print("\nweights", weights)

    non_zero_weights = [w for w in weights if w > 0]
    print("\nnon-zero weights", non_zero_weights)
    print("\nnumber of non-zero weights", len(non_zero_weights))
    print("\nsum of weghts", np.sum(non_zero_weights)) # Less than 1 because we filtered out miniscule weights in Optimizer
    
    print("\nDuration of portfolio:", np.dot(data['Modified_Duration'], weights))
    print("\nIspread of portfolio", np.dot(data['Ispread'], weights))
    print("\n\n")
    print("resulting dataframe:")
    print(optimizer.get_data())

    #optimizer.get_data().to_csv("results.csv")

    #data['Weights'] = weights
    #data.to_csv("2019-03-31.csv")
    #print("modified dataframe in optimizer (two distance columns and scaled Ispread column added)")
    #print(optimizer.get_data())