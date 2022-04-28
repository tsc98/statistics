from statistics import stdev

import numpy
import numpy as np
import pandas as pd
from scipy.stats import zscore, norm, f
from math import sqrt

class Sample:
    def __init__(self, population, std_p, mean_p, sample_values):
        # If there are any values missing use '0'
        self.sample_values = sample_values
        self.mean_p = mean_p
        self.mean_s = (sum(self.sample_values))/len(self.sample_values)
        self.std_s = stdev(self.sample_values)
        self.std_p = std_p
        self.sample_size = len(self.sample_values)
        self.population = population
        self.zscore = zscore(self.sample_values)
        self.percentiles = norm.cdf(self.zscore)

    def One_sample_zTest(self):
        # use for a know population Std and a sample size of greater the 30.
        # We perform the One-Sample Z test when we want to compare a sample mean with the population mean
        z = (self.mean_s - self.mean_p)/(self.std_p/(sqrt(self.sample_size)))
        percentile = norm.cdf(z)
        print(f'Sample Z-Score:{z}')
        print(f'Percentile:{percentile}')
        return z

    def One_sample_tTest(self):
        # We perform a One-Sample t-test when we want to compare a sample mean with the population mean.
        # The difference from the Z Test is that we do not have the information on Population Variance here.
        # We use the sample standard deviation instead of population standard deviation in this case.
        t = (self.mean_s - self.mean_p)/(self.std_s/(sqrt(self.sample_size)))
        dof = self.sample_size - 1
        percentile = t.sf(abs(t), df = dof)
        print(f'Sample Z-Score:{t}')
        print(f'Percentile:{percentile}')
        return t

    def run_test(self):
        if self.std_p > 0 and self.sample_size > 30:
            yhat = self.One_sample_zTest()
            print(f"The Z-Test output is:{yhat}")
            return yhat
        else:
            yhat = self.One_sample_tTest()
            print(f"The T-Test output is:{yhat}")
            return yhat


#sample_v = [1,2,3,4,5]  # Input Data
#sample_1 = Sample(100, 0, 0, sample_v, ) # Class instance


# Use Anova to compare two or more features and establish whether there is a differece i the variance.
class Anova:
    def __init__(self, samples, confidence_int):
        self.confidence_int = confidence_int
        self.samples = pd.DataFrame(samples)
        self.sample_means = self.samples.mean(axis=0)
        self.total_mean = numpy.mean(self.sample_means)
        self.n = len(self.samples.columns) # number of samples per features
        self.n_features = len(self.samples.index) # number of columns
        self.differences = self.mean_difference()
        self.SSC = sum(self.differences)
        self.sample_variances = pd.DataFrame(self.sample_variance()).T
        self.SSE = sum(self.sample_variances.sum(axis=1))
        self.DoG_v1 = self.n_features - 1
        self.DoG_v2 = (self.n * self.n_features) - self.n_features# needs a complete matrix
        self.MSC = self.SSC/self.DoG_v1
        self.MSE = self.SSE/self.DoG_v2
        self.summary = self.create_dataframe()
        self.f = self.MSC/self.MSE
        self.critical_value = f.ppf((1-self.confidence_int), dfn = 2, dfd = 6)
        self.calculate()

    def mean_difference(self):
        sequence = np.arange(0,(self.n_features), 1)
        x = []
        for idx in self.sample_means:
            difference = self.n * ((idx-self.total_mean)**2)
            x.append(difference)
        return x

    def sample_variance(self):
        feat_means = self.sample_means
        df = self.samples
        variances = []
        for (index, column) in enumerate(df): # This enables you to pull the index within each column
            mean = feat_means[column] # this uses the columns index and matches it to the relevant index of column means
            x = (df[column] - mean)**2
            variances.append(x)
        return variances

    def create_dataframe(self):
        data = [self.SSC, self.DoG_v1, self.MSC], [self.SSE, self.DoG_v2, self.MSE]
        df1 = pd.DataFrame(data, columns=['Sum of Squares', ' Degrees of Freedom', 'Mean Sum of Squares'])
        return df1

    def calculate(self):
        if self.f > self.critical_value:
            print("Test Output: Reject the Null hypothesis")
        else:
            print("Test Output: Null hypothesis is valid ")
        print(f"F({self.DoG_v1},{self.DoG_v2}) = {self.critical_value}(for p of {self.confidence_int})")


# keep the numerical index to enable sample variance to work.
s = {0:[10, 12, 15], 1:[14, 18, 22], 2:[19, 43, 22]} # Input data
s_dict = {0:'group1', 1:'group2', 2:'group3'} # Data dictionary
Anova_1 = Anova(s, 0.1) # Create an instacne of the class



