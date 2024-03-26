# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:56:20 2024

@author: Priyanka
"""

'''
An analytics company has been tasked with the crucial job of finding out what factors affect 
a startup company and if it will be profitable or not. For this, they have collected some 
historical data and would like to apply multilinear regression to derive brief insights into their data.
Predict profit, given different attributes for various startup companies.

Business Problem:
What is the business objective?
Profitability in business is a matter of survival: 
If your business doesn't stay profitable, you don't stay in business. 
The simple definition of profitability is that your revenue is more than your expenses.
The objective of this is to find impact of research, administration ,Marketing Spend on profit


Are there any constraints?
The corporate world is quite fierce.There is always a competition going on between the giants.
There is a huge pool of aspiring individuals available. Selecting a suitable candidate 
that fits the job well enough is a peculiarly tricky task.Customer is the king. And that’s absolutely right. 
Winning a customer’s trust is one of the most important challenges that businesses in general – and startups in particular

'''

import pandas as pd
import numpy as np
import seaborn as sns
# loading the data
startup = pd.read_csv("C:\Data Set\Multilinear_Regression\50_Startups.csv")
startup=startup.rename(columns={'R&D Spend':'Research','Marketing Spend':'Marketing'})
# Exploratory data analysis:
# 1. Measures of central tendency
startup.dtypes
startup.describe()
#average cost for resaerch is 73721 and 
#min=0 and max is 165349
#avearge cost of administration is 121344
#min=51283 and max=182645
##Data is right skewed
#average cost for marketing is 211025 and 
#min=0 and max is 471784
#average profit is 112012 and 
#min=14681 and max is 192261

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
# R and D spend
plt.bar(height = startup['Research'], x = np.arange(1, 51, 1))
sns.distplot(startup['Research']) #histogram
# Data is almost normallly distributed
plt.boxplot(startup['Research']) #boxplot
# No ouliers
plt.bar(height = startup['Administration'], x = np.arange(1, 51, 1))
sns.distplot(startup['Administration']) #histogram
# Data is almost normal slight left skewed
plt.boxplot(startup['Administration']) #boxplot
#No outliers
plt.bar(height = startup['Marketing'], x = np.arange(1, 51, 1))
sns.distplot(startup['Marketing']) #histogram
# Data is almost normal
plt.boxplot(startup['Marketing']) #boxplo
# No outliers

plt.bar(height = startup['Profit'], x = np.arange(1, 51, 1))
plt.hist(startup['Profit']) #histogram
# Data is almost normal
plt.boxplot(startup['Profit']) #boxplo
# There is one outlier
# 
# Jointplot
import seaborn as sns
sns.jointplot(x=startup['Research'], y=startup['Profit'])
#Resaerch and profit are linear
sns.jointplot(x=startup['Administration'], y=startup['Profit'])
# There is weak linearity of admin cost and profit
sns.jointplot(x=startup['Marketing'], y=startup['Profit'])
# Marketing spend and profit are almost linear
startup.drop(['State'],axis=1,inplace=True)
# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startup['Research'])
sns.countplot(startup['Administration'])
sns.countplot(startup['Marketing'])
# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show()
# Data is normallly distributed

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:, :])
#Profit and Administration is nonliear 
#except marketing spend and R & D spend rest all are not linear
                          
# Correlation matrix 
startup.corr()
# There is high colinearity between Profit n R &D spend
#similarly profit and marketing spend it is desired
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
first_model = smf.ols('Profit ~Research + Administration+ Marketing', data = startup).fit() # regression model

# Summary
first_model.summary()
# R-squared: 0.951,p-values for administration is 0.602 more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(first_model)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

startup_new = startup.drop(startup.index[[49]])

# Preparing model                  
first_model_new = smf.ols('Profit ~Research + Administration+ Marketing', data = startup_new).fit()    

# Summary
first_model_new.summary()
# There is no change in p value of administration

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF should not be > 10 = colinearity
# calculating VIF's values of independent variables
rsq_research = smf.ols('Research ~ Administration +Marketing', data = startup).fit().rsquared  
vif_research = 1/(1 - rsq_research) 

rsq_administration = smf.ols('Administration ~ Research +Marketing', data = startup).fit().rsquared  
vif_administration = 1/(1 - rsq_administration)

rsq_marketing = smf.ols('Marketing ~Research +Administration', data = startup).fit().rsquared  
vif_marketing = 1/(1 - rsq_marketing) 

# Storing vif values in a data frame
d1 = {'Variables':['Research', 'Administration', 'Marketing',], 'VIF':[vif_research, vif_administration, vif_marketing]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# All are having less than 10 VIF value
# let us go for transformation
new_admin=np.log(startup['Administration'])
# Final model
trans_model = smf.ols('Profit ~ Research +new_admin + Marketing', data = startup).fit()
trans_model.summary() 
##New_admin has p value=0.689 which is not improved hence Administration feature has to be droped
# Prediction
final_ml = smf.ols('Profit ~ Research + Marketing', data = startup).fit()
final_ml.summary() 
pred = final_ml.predict(startup)
# R sqaure value is 0.95 and p values are in the range
# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startup, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Profit ~ Research + Marketing', data = startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_errors = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_errors * test_errors))
test_rmse


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

