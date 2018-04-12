
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""

@author: xshitova
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df  = pd.read_csv("C:/Users/xshitova/Documents/Data science/corp_gov1.csv", delimiter=';')

dependent=["Stock_price_fall"]

corp_gov_vars = ["Discipline ", 	"Transparency",	"Independence",	"Accountability","Responsibility"
	,"CLSA_Fairness","Social"]	
    
firm_vars= ["Total asset turnover",	"Net profit margin",	"ROA","ROI",
	"ROE",	"Revenue total ", 	"Beta1"]	
#present in the dataset but reduce it too much

country_vars_cg=["Antidirector rights Djankov",	
                 "Ownership concentration",
                 "Antidirector rights Porta",
                 "Judicial Efficiency (Porta 2005 - ICRG)",
                 "Corruption perception index",	
                 "Rule of law - La porta 2005 Kaufmann 2003",
                 "RL",	
                 "Corruption - La Porta 2005 and Kaufmann 2003",
                 "Corruption",	
                 "Government effectiveness"]

country_vars_econ=["Market capitalization",	"GDP 2006",	"Ln (GDP)",	
                   "Market cap/GDP",	"GDP growth"]

controlling= ["oil gas mining",	"automobiles",	"chemicals",	"electricity",	"engineering ", "financial",
	"foods",	"technical",	"real estate investment",	"telecom",	"transport",	"general industrials",	
    "general retailers",	"goods",	"media",	"pharmaceuticals","China",	"Hong Kong",	"India",	
    "Indonesia",	"Korea", "Malaysia",	"Philippines",	"Singapore", "Taiwan",	"Thailand","EEMEA"	,
           "Latin America"]	

all=["CLSA_Fairness",
     "Antidirector rights Djankov",	
     "Antidirector rights Porta",
                 "Ownership concentration",
                 "Government effectiveness",
     "Market capitalization","Corruption perception index",	"Ln (GDP)",	
                   	"oil gas mining",	"automobiles",	"chemicals",	"electricity",	"engineering ", "financial",
	"foods",	"technical",	"real estate investment",	"telecom",	"transport",	"goods",	"media",	"pharmaceuticals","China",	"Hong Kong",	"India",	
    "Indonesia",	"Korea", "Malaysia",	"Philippines",	"Singapore", "Taiwan",	"Thailand","EEMEA"	,
           "Latin America"]

df.dropna(inplace=True) #drop nan values from the dataset

print(df[corp_gov_vars].corr()) #elements of the CLSA index 
plt.matshow(df[corp_gov_vars].corr()) #show the matrix 
#print(df[firm_vars].corr()) #firm related factors 
#plt.matshow(df[firm_vars].corr())
print(df[country_vars_cg].corr())
plt.matshow(df[country_vars_cg].corr())

print(df[country_vars_econ].corr())
plt.matshow(df[country_vars_econ].corr())

print("") 
print(df.isna().any()) #completeness of the dataset 


print(df[all].describe())  #description of variables 

predictors=df[all] #the explanatory variables 
#predictors1=df[all] #only country-related predictory variables 
#predictors2=df[firm_vars] #only firm-related predictory variables 
#predictors3=df[corp_gov_vars] #only the components of the index (CLSA) and the index itself 
predicted=df[dependent] #the dependent variable 


X2 = sm.add_constant(predictors) #adding a constant 
est = sm.OLS(predicted, X2) #OLS method 
est2 = est.fit() #fitting the model 
print(est2.summary()) #summary of regression results 

