# -*- coding: utf-8 -*-
"""Untitled9.ipynb """

!pip install -U dataprep

# Commented out IPython magic to ensure Python compatibility.
################# Libraries ####################
import numpy as np
import pandas as pd
import math
from datetime import timedelta, datetime
from dataprep.clean import clean_country
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (20, 5)


df1 = pd.read_csv('sales_asia.csv', 
                  dtype={'week.year': str}, 
                  sep=';', 
                  decimal=',')

df1.head(15)

df1.info()

# Splitting 'week.year' column on '.' and creating 'week' and 'year' columns

df1['week'] = df1['week.year'].astype(str).str.split('.').str[0]
df1['year'] = df1['week.year'].astype(str).str.split('.').str[1]

# Converting year and week into date, using Monday as first day of the week

df1['date'] = pd.to_datetime(df1['year'].map(str) + df1['week'].map(str) + '-1', format='%Y%W-%w')

df1.head()

# Removing unnecesary columns

df2 = df1.drop(['week.year', 'week', 'year'], axis=1)

#Rename columns

df2.rename({'revenue': 'monetary'}, axis="columns", inplace=True)

df2.describe()

########## Period of time included in the data ##########
print(df2["date"].min())
print("Maximum Date :",df2["date"].max() )

df2['country'].unique()

clean_country(df2, "country")['country_clean'].unique()

df2["Clean Country"] = clean_country(df2, "country")['country_clean']

df2


#what are the unique number of customers##
df2['id'].nunique()

# Putting date into the index for plotting the time series

df2b = df2.set_index("date")
df2b.head()

plt.style.use('ggplot')
plt.title('Units sold per week')
plt.ylabel('units')
plt.xlabel('date');
df2b['units'].plot(figsize=(20,5), c='dodgerblue');

plt.style.use('ggplot')
plt.title('Revenue per week')
plt.ylabel('units')
plt.xlabel('date');
df2b['monetary'].plot(figsize=(20,5), c='dodgerblue');

# For greater visibility in the plots we convert the dates to monthly periods and we aggregate the units and revenue of the same period
df2c = df2b.to_period("M")

df2c.head()

plt.style.use('ggplot')
df2c['units'].groupby('date').agg(sum).plot(figsize=(20,5), c='dodgerblue')
plt.title('Units sold per month')
plt.ylabel('units')
plt.xlabel('date');

plt.style.use('ggplot')
df2c['monetary'].groupby('date').agg(sum).plot(figsize=(20,5), c='dodgerblue')
plt.title('Revenue per month')
plt.ylabel('revenue')
plt.xlabel('date');

print('Sales from {} to {}'.format(df2['date'].min(),
                                    df2['date'].max()))

#Let's focus on sales from last 365 days since most recent date

period = 365
date_N_days_ago = df2['date'].max() - timedelta(days=period)

date_N_days_ago

# We remove the rows with dates older than 365 days ago

df2 = df2[df2['date']> date_N_days_ago]

df2.reset_index(drop=True, inplace=True)

df2.head()

# There are customers with the same 'id' in several countries. This causes errors in the monetary values
# Let's create a unique 'id+' identifier that combines country code and customer id

df3 = df2.copy()

df3['id+'] = df3['country'].map(str) + df3['id'].map(str)

# We set the NOW date one day after the last sale

NOW = df3['date'].max() + timedelta(days=1)
NOW

#### We add a column, 'days_since_last_purchase', with the days between purchase date and the latest date

df3['days_since_purchase'] = df3['date'].apply(lambda x:(NOW - x).days)
df3.head()

df3["days_since_purchase"].min()

##################### Check records of particular ID ###################
df3[df3['id+']=='KR706854']

df4 = df3.groupby("id+")["days_since_purchase"].min()

df4

# Recency will be the minimum of 'days_since_last_purchase' for each customer
# Frequency will be the total number of orders in the period for each customer

aggr = {
    'days_since_purchase': lambda x:x.min(),
    'date': lambda x: len([d for d in x if d >= NOW - timedelta(days=period)])
}

rfm = df3.groupby(['id', 'id+', 'country']).agg(aggr).reset_index()
rfm.rename(columns={'days_since_purchase': 'recency',
                   'date': 'frequency'},
          inplace=True)

rfm

# We check customers with id 3790218 have different recency and frequency values per country

rfm[rfm['id']==3790218]

######## We get the revenue of the last 365 days per customer ##############

df3[df3['date'] >= NOW - timedelta(days=period)]\
    .groupby('id+')['monetary'].sum()

# We add the revenue from df3 of last period per customer to rfm dataframe

rfm['monetary'] = rfm['id+']\
    .apply(lambda x: df3[ (df3['id+'] == x) & (df3['date'] >= NOW - timedelta(days=period))]\
    .groupby(['id', 'country']).sum().iloc[0,0])

rfm

# Checking monetary value is correct by checking on our biggest customer

rfm[rfm['monetary']==rfm['monetary'].max()]

rfm.drop(['id+'], axis=1, inplace=True)

rfm[['recency', 'frequency', 'monetary']].quantile([.2, .4, .6, .8])

# We assign a rate between 1 and 5 depending on recency, monetary and frequency parameters
# We use the quintiles method, dividing every feature on groups that contain 20 % of the samples

quintiles = rfm[['recency', 'frequency', 'monetary']].quantile([.2, .4, .6, .8]).to_dict()
quintiles

# Assigning scores from 1 to 5
# Higher values are better for frequency and monetary, while lower values are better for recency

def r_score(x):
    if x <= quintiles['recency'][.2]:
        return 5
    elif x <= quintiles['recency'][.4]:
        return 4
    elif x <= quintiles['recency'][.6]:
        return 3
    elif x <= quintiles['recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5

# We asssign R, F and M scores to each customer

rfm['r'] = rfm['recency'].apply(lambda x: r_score(x))
rfm['f'] = rfm['frequency'].apply(lambda x: fm_score(x, 'frequency'))
rfm['m'] = rfm['monetary'].apply(lambda x: fm_score(x, 'monetary'))

rfm.head()

# Combine R, F and M scores to create a unique RFM score

rfm['rfm_score'] = rfm['r'].map(str) + rfm['f'].map(str) + rfm['m'].map(str)
rfm.head()

# With this rfm scores we would have 125 segments of customers
# To make a more simple segment map of 11 segments, we combine f and m scores, rounding them down
# fm = (f+m)/2

def truncate(x):
    return math.trunc(x)

rfm['fm'] = ((rfm['f'] + rfm['m'])/2).apply(lambda x: truncate(x))

rfm.head()

####### We create a segment map of only 11 segments based on only two scores: 'r' and 'fm'

segment_map = {
    r'22': 'hibernating',
    r'[1-2][1-2]': 'lost',
    r'15': 'can\'t lose',
    r'[1-2][3-5]': 'at risk',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'55': 'champions',
    r'[3-5][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists'
}

rfm['segment'] = rfm['r'].map(str) + rfm['fm'].map(str)
rfm['segment'] = rfm['segment'].replace(segment_map, regex=True)
rfm.head()

rfm["Clean Country"] = clean_country(rfm, "country")['country_clean']


rfm.to_csv('customerseg.csv') 
