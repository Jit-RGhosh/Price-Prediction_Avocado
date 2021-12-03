#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd #data processing and I/O operations
import numpy as np #Linear Algebra
import seaborn as sns# Seaborn for plotting and styling
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split #Spliting Datasheet
import xgboost as xgb
from sklearn.metrics import r2_score


# # Import the dataset.

# In[112]:


avcdo = pd.read_csv('E:/Python/Python - Decodr/DecodR_Class/October - 2021/Projects/Avocado_Project/Avocado.csv')


# In[113]:


avcdo


# ## Finding Detail information about dataset.

# In[27]:


avcdo.info()


# In[28]:


avcdo.shape


# In[29]:


avcdo.columns


# ## Finding Null Value

# In[30]:


avcdo.isnull().sum()


# In[31]:


avcdo.head(5)


# In[32]:


avcdo.tail(5)


# # Data Cleaning - Remove irrelevant features, handling missing values, converting into correct data format, etc. (if required)

# In[69]:


avcdo.describe() # Get basic Data Description of given data


# ### As we can see that the average Price level is around 1.40 and 50% quantile also known as median which indicate Price around 1.37. So we can conclude that Chances outlier is less. Precisely we can ignore Outlier considering practical implementation

# ## Removing Unnecessery Column

# In[34]:


avcdo.drop('Unnamed: 0', axis = 1)


# ## Univariate Analysis

# In[35]:


plt.scatter(avcdo['year'], avcdo['AveragePrice'])
plt.xlabel('Year')
plt.ylabel('Average Price')


# ### Based on above Plot price of Avacado varies year on year however maxumum increase in the Year of 2016

# In[37]:


avcdo.AveragePrice.hist()
plt.plot()


# In[45]:


plt.scatter(avcdo['year'], avcdo['Total Volume'])
plt.xlabel('Year')
plt.ylabel('Total Volume')


# ## Bivariate Analysis

# In[48]:


plt.scatter(avcdo['AveragePrice'], avcdo['Total Volume'])
plt.xlabel('AveragePrice')
plt.ylabel('Total Volume')


# ### This Scatter plot clearly shown that price mostly varies between 0.5 to 1.5. It also prominent that at price range near 1, Sales volume is high compare to any other price range

# ## Multivariate Analysis

# In[47]:


sns.pairplot(avcdo)


# ## Feature Engineering - Label Encoding, One Hot Encoding, Feature Scaling. (if required)

# ### Data Modification for related prediction and Model Design

# In[55]:


df_mini1 = avcdo[['Total Volume', '4046']]
df_mini2 = avcdo[['Total Volume', '4225']]
df_mini3 = avcdo[['Total Volume', '4770']]


# In[56]:


df_mini1.head()


# In[57]:


df_corr1 = df_mini1.corr()
df_corr2 = df_mini2.corr()
df_corr3 = df_mini3.corr()


# In[58]:


df_corr1


# In[59]:


df_corr2


# In[60]:


df_corr3


# # Now Finding Correlation between each type of labeled Avocados and Total Volume of Avocados

# In[61]:


fig1 = plt.figure()
fig1, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(20,5))
axes[0].set_title('Correlation b/w Total Vol & 4046 Labeled Avocados')
axes[1].set_title('Correlation b/w Total Vol & 4225 Labeled Avocados')
axes[2].set_title('Correlation b/w Total Vol & 4770 Labeled Avocados')
sns.heatmap(df_corr1, ax = axes[0], annot = True, cmap = 'coolwarm')
sns.heatmap(df_corr2, ax = axes[1], annot = True, cmap = 'coolwarm')
sns.heatmap(df_corr3, ax = axes[2], annot = True, cmap = 'coolwarm')


# In[63]:


fig2 = plt.figure(figsize = (80,40))
axes4 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
sns.barplot(x = avcdo['region'], y = avcdo['AveragePrice'], ax = axes4)
axes4.set_xlabel('Region')
axes4.set_ylabel('Average Price')


# In[74]:


df = avcdo.drop(['Date', 'year'], axis = 1)


# In[75]:


df.head()


# In[76]:


target_df = df['AveragePrice']


# In[77]:


df = df.drop('AveragePrice', axis = 1)


# In[78]:


target_df.head()


# In[79]:


df.head()


# ### Categorical Variables Encloding, Encoding with Frequency Encoder

# In[83]:


encodr = OrdinalEncoder()


# In[90]:


encodr.fit([df['region']])
df['region']= encodr.fit_transform(df[['region']])


# In[91]:


df.head()


# ## Hot Encoading

# In[92]:


df['type'] = pd.get_dummies(df['type'], drop_first = True)


# In[93]:


df.head()


# In[94]:


X = df.values
y = target_df.values


# # Splitting the data into training and testing

# In[117]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[123]:


regressor = xgb.XGBRegressor(n_estimators = 800, max_depth = 10)
regressor.fit(X_train, y_train)


# In[119]:


y_pred = regressor.predict(X_test) #Prediction


# ## Evaluating a model

# In[126]:


r2_score(y_test, y_pred)


# In[127]:


fig10 = plt.figure(figsize = (10, 5))
axes6 = fig10.add_axes([0.1, 0.1, 0.8, 0.8])
sns.scatterplot(y_test, y_pred, linewidth = 1, ax = axes6)
plt.title('Scatter Plot between the true labels and predicted labels')
axes6.set_xlabel("True Average Price")
axes6.set_ylabel("Predicted Average Price")


# In[ ]:




