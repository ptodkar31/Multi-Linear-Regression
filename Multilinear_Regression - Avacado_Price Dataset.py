# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:56:11 2024

@author: Priyanka
"""

"""
With the growing consumption of avocados in the USA, a freelance company would 
like to do some analysis on the patterns of consumption in different cities and
would like to come up with a prediction model for the price of avocados. 
For this to be implemented, build a prediction model using multilinear 
regression and provide your insights on it.

What is the business objective?
The data represents weekly 2018 retail scan data for National retail volume (units) and price. 
Retail scan data comes directly from retailers’ cash registers based on actual retail sales 
of Hass avocados.Starting in 2013, the data reflects an expanded, multi-outlet retail data set. 
Multi-outlet reporting includes an aggregation of the following channels: 
grocery, mass, club, drug, dollar and military. The Average Price (of avocados)
in the data reflects a per unit (per avocado) cost, even when multiple units (avocados) 
are sold in bags. The Product Lookup codes (PLU’s) in the table are only for Hass avocados. 
Other varieties of avocados (e.g. greenskins) are not included in this data.

"""


import pandas as pd
import numpy as np
import seaborn as sns
Avacado=pd.read_csv("C:\Data Set\Multilinear_Regression\Avacado_Price.csv")
Avacado_new=Avacado.iloc[:,0:11]


# Exploratory data analysis
#1.Measure the central tendency
#2.Measure the dispersion
#3.Third moment business decision
#4.Fourth moment business decision
#5.probability distribution
#6.Graphical represenation(Histogram,Boxplot)

Avacado_new.describe()
#Following columns have been dropped
#Total Volume:sum of Hass Avocado with PSU labels 
#Total Bags:sum of various bag size
#date column has been dropped as there too many levels
#region column has been dropped as there are too many levels

Avacado_new=Avacado_new.rename(columns={'XLarge Bags':'XLarge_Bags'})
Avacado_new.isna().sum()
'''
AveragePrice    0
Total_Volume    0
tot_ava1        0
tot_ava2        0
tot_ava3        0
Total_Bags      0
Small_Bags      0
Large_Bags      0
XLarge_Bags     0
type            0
year            0
'''
#There are no null values
import matplotlib.pyplot as plt
plt.bar(height=Avacado_new.AveragePrice,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.AveragePrice)
#Data is normal slight right skewed

plt.boxplot(Avacado_new.AveragePrice)
# There are several outliers
plt.bar(height=Avacado_new.Total_Volume,x=np.arange(1,18250,1))

sns.distplot(Avacado_new.Total_Volume)
#Data is normal but right skewed
plt.boxplot(Avacado_new.Total_Volume)
#There are several outliers
#let us check tot_ava1
plt.bar(height=Avacado_new.tot_Avacado1,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.tot_ava1)
#Data is normal but slight right skewed
plt.boxplot(Avacado_new.tot_Avacado1)
#There are several outliers
# let us check tot_ava2
plt.bar(height=Avacado_new.tot_Avacado2,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.tot_Avacado2)
#Data is normal but slight right skewed
plt.boxplot(Avacado_new.tot_Avacado2)
#There are several outliers
# let us check tot_ava3
plt.bar(height=Avacado_new.tot_Avacado3,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.tot_Avacado3)
#Data is normal but slight rt skewed
plt.boxplot(Avacado_new.tot_Avacado3)
#There are several outliers
#let us check Total_Bags
plt.bar(height=Avacado_new.Total_Bags,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.Total_Bags)
#Data is normal but slight rt skewed
plt.boxplot(Avacado_new.Total_Bags)
#There are several outliers

#let us check Small_Bags
plt.bar(height=Avacado_new.Small_Bags,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.Small_Bags)
#Data is normal but slight rt skewed
plt.boxplot(Avacado_new.Small_Bags)
#There are several outliers
#let us check Large_Bags
plt.bar(height=Avacado_new.Large_Bags,x=np.arange(1,18250,1))
sns.distplot(Avacado_new.Large_Bags)
#Data is normal but slight rt skewed
plt.boxplot(Avacado_new.Large_Bags)
#There are several outliers


#Data preprocessing

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
Avacado_new["year"]=lb.fit_transform(Avacado["year"])
Avacado_new["type"]=lb.fit_transform(Avacado["type"])
Avacado_new.dtypes
from feature_engine.outliers import Winsorizer
import seaborn as sns
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['AveragePrice'])
Avacado_t=winsor.fit_transform(Avacado_new[['AveragePrice']])
sns.boxplot(Avacado_t.AveragePrice)
Avacado_new['AveragePrice']=Avacado_t['AveragePrice']
plt.boxplot(Avacado_new.AveragePrice)
# let us check Total_Volume
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Total_Volume'])
Avacado_t=winsor.fit_transform(Avacado_new[['Total_Volume']])
sns.boxplot(Avacado_t.Total_Volume)
Avacado_new['Total_Volume']=Avacado_t['Total_Volume']
plt.boxplot(Avacado_new.Total_Volume)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava1'])
Avacado_t=winsor.fit_transform(Avacado_new[['tot_ava1']])
sns.boxplot(Avacado.tot_Avacado1)
Avacado_new['tot_ava1']=Avacado_t['tot_ava1']
plt.boxplot(Avacado_new.tot_Avacado1)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava2'])
Avacado_t=winsor.fit_transform(Avacado_new[['tot_ava2']])
sns.boxplot(Avacado_t.tot_Avacado2)
Avacado_new['tot_ava2']=Avacado_t['tot_ava2']
plt.boxplot(Avacado_new.tot_ava2)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava3'])
Avacado_t=winsor.fit_transform(Avacado_new[['tot_ava3']])
sns.boxplot(Avacado_t.tot_Avacado3)
Avacado_new['tot_ava3']=Avacado_t['tot_ava3']
plt.boxplot(Avacado_new.tot_Avacado3)
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Total_Bags'])
Avacado_t=winsor.fit_transform(Avacado_new[['Total_Bags']])
sns.boxplot(Avacado_t.Total_Bags)
Avacado_new['Total_Bags']=Avacado_t['Total_Bags']
plt.boxplot(Avacado_new.Total_Bags)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Small_Bags'])
Avacado_t=winsor.fit_transform(Avacado_new[['Small_Bags']])
sns.boxplot(Avacado_t.Small_Bags)
Avacado_new['Small_Bags']=Avacado_t['Small_Bags']
plt.boxplot(Avacado_new.Small_Bags)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Large_Bags'])
Avacado_t=winsor.fit_transform(Avacado_new[['Large_Bags']])
sns.boxplot(Avacado_t.Large_Bags)
Avacado_new['Large_Bags']=Avacado_t['Large_Bags']
plt.boxplot(Avacado_new.Large_Bags)




#Graphical represenation,Bivariant analysis
#Now let us check colinearity between Y and X1,X2,....plot joint plot,joint plot is to show scatter plot as well 
# histogram
Avacado_new.dtypes
import seaborn as sns
sns.jointplot(x=Avacado_new['Total_Volume'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['tot_ava1'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['tot_ava2'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['tot_ava3'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['Total_Bags'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['Small_Bags'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['Large_Bags'],y=Avacado_new['AveragePrice'])
sns.jointplot(x=Avacado_new['XLarge_Bags'],y=Avacado_new['AveragePrice'])


import numpy as np
corr_df = Avacado_new.corr(method='pearson')
plt.figure(figsize=(12,6),dpi=100)
sns.heatmap(corr_df,cmap='coolwarm',annot=True)
#only type and year are not correlated with other variables.

Avacado_new.dtypes


#QQ plot
from scipy import stats
import pylab
stats.probplot(Avacado_new['AveragePrice'],dist="norm",plot=pylab)
#Data is normal
stats.probplot(Avacado_new['Total_Volume'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['tot_ava1'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['tot_ava2'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['tot_ava3'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['Total_Bags'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['Small_Bags'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['Large_Bags'],dist="norm",plot=pylab)
#Data is not normal
stats.probplot(Avacado_new['XLarge_Bags'],dist="norm",plot=pylab)
#Data is normal

plt.show()
# Average Price data is normally distributed
# There are 28 scatter plots need to be plotted,one by one is difficult
#to plot,so we can use pair plots
import seaborn as sns
sns.pairplot(Avacado_new.iloc[:,:])
# you can check the collinearity problem between the input variables
# you can check plot between Total_ava1 and Total_ava2,they are strongly corelated
# same way you can check WT and VOL,it is also strongly correlated

# now let us check r value between variables
Avacado_new.dtypes
Avacado_new.corr()
#Except type and year,all other variables are poorly correlated with AveragePrice
#Except X_large_bags,type and year,all other variables are correlated with each other 
import statsmodels.formula.api as smf
rsq_tot_vol=smf.ols('Total_Volume~tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_tot_vol=1/(1-rsq_tot_vol)

rsq_tot_ava1=smf.ols('tot_ava1~Total_Volume+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_tot_ava1=1/(1-rsq_tot_ava1)

rsq_tot_ava2=smf.ols('tot_ava2~Total_Volume+tot_ava1+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_tot_ava2=1/(1-rsq_tot_ava2)

rsq_tot_ava3=smf.ols('tot_ava3~Total_Volume+tot_ava1+tot_ava2+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_tot_ava3=1/(1-rsq_tot_ava3)

rsq_tot_bags=smf.ols('Total_Bags~tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_tot_bags=1/(1-rsq_tot_bags)

rsq_Small_Bags=smf.ols('Small_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_Small_Bags=1/(1-rsq_Small_Bags)

rsq_Large_Bags=smf.ols('Large_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+XLarge_Bags+type+year',data=Avacado_new).fit().rsquared
vif_Large_Bags=1/(1-rsq_Large_Bags)

rsq_XLarge_Bags=smf.ols('XLarge_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+Large_Bags+type+year',data=Avacado_new).fit().rsquared
vif_XLarge_Bags=1/(1-rsq_XLarge_Bags)

rsq_type=smf.ols('type~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+Large_Bags+XLarge_Bags+year',data=Avacado_new).fit().rsquared
vif_type=1/(1-rsq_type)

rsq_year=smf.ols('Large_Bags~Total_Bags+tot_ava2+Total_Volume+tot_ava1+tot_ava3+Small_Bags+XLarge_Bags+type',data=Avacado_new).fit().rsquared
vif_year=1/(1-rsq_year)




d1={'Variables':['Total_Volume','tot_ava1','tot_ava2','tot_ava3','Total_Bags','Small_Bags','Large_Bags','XLarge_Bags','type','year'],
    'VIF':[vif_tot_vol,vif_tot_ava1,vif_tot_ava2,vif_tot_ava3,vif_tot_bags,vif_Small_Bags,vif_Large_Bags,vif_XLarge_Bags,vif_type,vif_year]}

vif_frame=pd.DataFrame(d1)
vif_frame
#Total_Volume,tot_av2,Total_Bags and Small_Bags have vif>10 hence dropping these columns
import statsmodels.formula.api as smf
ml=smf.ols('AveragePrice~tot_ava1+tot_ava3+Large_Bags+XLarge_Bags+type+year',data=Avacado_new).fit()
ml.summary()

# prediction
pred=ml.predict(Avacado_new)
import statsmodels.api as sm
##QQ plot
res=ml.resid
sm.qqplot(res)
plt.show()
# This QQ plot is on residual which is obtained on training data
#eerors are obtained on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

#let us plot the residual plot ,which takes the residuals values 
#and the data
sns.residplot(x=pred,y=Avacado_new.AveragePrice,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
# residual plots are used to check whether the errors are independent or not

# let us plot the influence plot
sm.graphics.influence_plot(ml)

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
Avacado_train,Avacado_test=train_test_split(Avacado_new,test_size=0.2)
#preparing the model on train data 
model_train=smf.ols('AveragePrice~tot_ava1+tot_ava3+Large_Bags+XLarge_Bags+type+year',data=Avacado_train).fit()
model_train.summary()
test_pred=model_train.predict(Avacado_test)

#test_errors
test_error=test_pred-Avacado_test.AveragePrice
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse