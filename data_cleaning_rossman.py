import numpy as np
import pandas as pd
import os

def clean_rossman(csv_tseries='train.csv', csv_store='store.csv', data_folder='data'):

    # Path names to read the csv files
    tseries_directory = os.path.join(data_folder, csv_tseries)
    store_directory = os.path.join(data_folder, csv_store)

    # Read csv files to load the data
    raw_tseries = pd.read_csv(tseries_directory, dtype={'StateHoliday':'object'})
    raw_store = pd.read_csv(store_directory)

    # Merge the two datasets to get a full view of the data
    raw_tseries['Store'] = raw_tseries['Store'].fillna(0)
    raw_tseries['Store'] = raw_tseries['Store'].astype(int)
    raw_merged = pd.merge(raw_tseries, raw_store, on="Store", how='left')
    cleaned = raw_merged.copy()

    # # Drop the customers column as it makes the dataset too artificial
    # cleaned = cleaned.drop(columns=['Customers'])

    # Replace missing values for Customers data with the median of the column
    cleaned['Customers'] = cleaned['Customers'].fillna(cleaned[cleaned['Customers'].notna()].loc[:,
                                                       'Customers'].median())

    # Drop all rows without a store id or a sales value as we don't have any info in that case
    cleaned = cleaned[cleaned.loc[:, 'Store'] != 0]
    cleaned = cleaned[cleaned.loc[:, 'Sales'] != 0]

    # Drop all the rows where the stores weren't open on that day. We assume there weren't any sales
    # as they all have a NaN value
    cleaned = cleaned[cleaned.loc[:, 'Open'] != 0]

    # # If the rows have a NaN value for the Sales as well as the Open column, we delete them as well
    # null_sales = cleaned[cleaned.loc[:, 'Sales'].isnull()]
    # null_sales_null_open = null_sales[null_sales.loc[:, 'Open'].isnull()]
    # cleaned = cleaned[~cleaned.index.isin(null_sales_null_open.index)]

    # When the Sales don't have a value, we delete them
    cleaned = cleaned[cleaned['Sales'].notna()]

    # Now we can drop the Open Column since we infer the store is open on days when there are sales
    cleaned = cleaned.drop(columns=['Open'])

    # Change the Date column to Datetime
    cleaned['Date'] =  pd.to_datetime(cleaned['Date'], format='%Y-%m-%d')

    # Fill in the missing values of the DayOfWeek column based on the Date
    cleaned['DayOfWeek'] = cleaned['DayOfWeek'].fillna(cleaned['Date'].dt.weekday + 1)

    # Drop the rows with missing values from the Promo column
    cleaned = cleaned[cleaned['Promo'].notna()]

    # Replace all the missing values from the StateHoliday with the most frequent value of the column
    # for date in set(cleaned.loc[:, 'Date']):
    #     #  get rows with this customer type
    #     mask = cleaned.loc[:, 'Date'] == date
    #
    #     #  fill the na's using the most frequent value at this date
    #     cleaned.loc[:, 'StateHoliday'].fillna(cleaned.loc[mask, 'StateHoliday'].value_counts().idxmax(), inplace=True)
    cleaned.loc[:, 'StateHoliday'].fillna(cleaned.loc[:, 'StateHoliday'].value_counts().idxmax(), inplace=True)

    # Replace all the missing values from the SchoolHoliday with the most frequent value of the column
    # for date in set(cleaned.loc[:, 'Date']):
    #     #  get rows with this customer type
    #     mask = cleaned.loc[:, 'Date'] == date
    #
    #     #  fill the na's using the most frequent value at this date
    #     cleaned.loc[:, 'StateHoliday'].fillna(cleaned.loc[mask, 'StateHoliday'].value_counts().idxmax(), inplace=True)
    cleaned.loc[:, 'SchoolHoliday'].fillna(cleaned.loc[:, 'SchoolHoliday'].value_counts().idxmax(), inplace=True)

    # Replace all the missing values from the CompetitionDistance column with the most frequent distance for that store
    for store in set(cleaned.loc[:, 'Store']):
        #  get rows with this store id
        mask = cleaned.loc[:, 'Store'] == store

        #  fill the na's using the most frequent value at this store
        cleaned.loc[:, 'CompetitionDistance'].fillna(cleaned.loc[mask, 'CompetitionDistance'].value_counts().idxmax(),
                                                     inplace=True)

    # Next we change the missing values in the PromoInterval to a 'No Promo' string to be able to
    # use them in the feature engineering
    cleaned.loc[:, 'PromoInterval'].fillna('No Promo', inplace=True)

    # Now only columns with numbers in a categorical context remain so they get mapped to a zero value to be able to use
    # them in the feature engineering of those columns, so a zero means no competition or no promo in those columns
    cleaned.fillna(0, inplace=True)

    # # Write the cleaned DataFrame to a csv file in the current directory
    # cwd = os.getcwd()
    # path = os.path.join(cwd, "cleaned.csv")
    # cleaned.to_csv(path, index=False)

    # Delete the fifth percentiles of the Customers and Sales columns
    pct_cust = np.percentile(cleaned.loc[:, 'Customers'], 98)
    cleaned = cleaned.loc[cleaned.loc[:, 'Customers'] < pct_cust]
    pct_sales = np.percentile(cleaned.loc[:, 'Sales'], 98)
    cleaned = cleaned.loc[cleaned.loc[:, 'Sales'] < pct_sales]

    # Change all the float columns to int type
    columns_to_int = ['DayOfWeek', 'Sales', 'Customers', 'Promo', 'SchoolHoliday', 'CompetitionDistance',
       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       'Promo2SinceWeek', 'Promo2SinceYear']
    cleaned.loc[:, columns_to_int] = cleaned.loc[:, columns_to_int].astype(int)

    return cleaned

if __name__ == "__main__":
    cleaned = clean_rossman()