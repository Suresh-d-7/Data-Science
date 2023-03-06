#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf

data = yf.download("^NSEI", start="2022-03-06", end="2023-03-06")
y = data['Adj Close']
model = sm.tsa.ARMA(y, order=(2,1))
results = model.fit()
prediction = results.forecast()[0]
print("The predicted price for the next day is:", prediction)


# In[47]:


def predict_next_day_price():
    data = yf.download("^NSEI", period="1y")
    y = data['Adj Close']
    model = sm.tsa.ARMA(y,order=(2,1))
    results = model.fit()
    prediction = results.forecast()[0]
    return prediction


# In[46]:


next_day_price = predict_next_day_price()
print("The predicted price for the next day is:", next_day_price)


# In[ ]:




