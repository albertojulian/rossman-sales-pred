import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import boxcox, zscore
import os

# Path names to read the csv files
a = 'data'
b = 'train.csv'
c = 'store.csv'
train = os.path.join(a, b)
store = os.path.join(a, c)

# Read csv files to load the data
raw_train = pd.read_csv(train, dtype={'StateHoliday':'object'})
raw_store = pd.read_csv(store)

# Merge the two datasets to get a full view of the data
raw_train['Store'] = raw_train['Store'].fillna(0)
raw_train['Store'] = raw_train['Store'].astype(int)
raw_merged = pd.merge(raw_train, raw_store, on="Store", how='left')
cleaned = raw_merged.copy()

# Drop all rows without a store id as we don't have any info in that case
cleaned = cleaned[cleaned.loc[:, 'Store'] != 0]

# Drop the customers column as it makes the dataset too artificial
cleaned = cleaned.drop(columns=['Customers'])

# Change the Date column to Datetime so it can be used in the feature engineering and modeling parts
cleaned['Date'] =  pd.to_datetime(cleaned['Date'], format='%Y-%m-%d')

# Drop all rows with Null values to get the most basic model running
cleaned = cleaned.dropna(axis=0)

cleaned.to_csv(index=False)

