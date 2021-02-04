import category_encoders as ce
import pandas as pd

# load the cleaned dataframe
def feat_eng(rossman_featured, train_data=True, te_store=None, te_week=None, te_day=None) :
    """
    
    This function perform some feature engineering tasks having as input a pandas dataframe. 
    
    - From the Date column, 3 news features (year, day of the month and week of the year) are created and
     when necessary they have their type changed.
     
    - A new feature is obtained dividing the Sales per Customer per Store.
    
    - One hot encoding and target encoding techniques are applied to some categorical features.
    
    - Some features are dropped.
    
    ...
    
    Attributes
    ----------
    rossman_featured : pandas.core.frame.DataFrame
         
    """        
    # create a new features from Date column
    rossman_featured['Year'] = rossman_featured['Date'].dt.year                       # year
    #rossman_featured['Month'] = rossman_featured['Date'].dt.month                    # month
    rossman_featured['Day'] = rossman_featured['Date'].dt.day                         # day of the month
    rossman_featured['WeekofYear'] = rossman_featured['Date'].dt.isocalendar().week   # week of the year

    # convert Week of the Year to integer
    rossman_featured['WeekofYear'] = rossman_featured['WeekofYear'].astype(int)

    # apply one hot encoding to some features
    rossman_featured = pd.get_dummies(data=rossman_featured,
                                     columns=['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'DayOfWeek'],
                                     prefix=['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'DayOfWeek'],
                                     prefix_sep='_')

    if train_data == True:
        # apply target encoding to the feature Store
        te_store = ce.TargetEncoder(cols=['Store'])
        rossman_featured['Store_target'] = te_store.fit_transform(rossman_featured['Store'], rossman_featured['Sales'])

        # apply target encoding to the feature WeekofYear
        te_week = ce.TargetEncoder(cols=['WeekofYear'])
        rossman_featured['WeekofYear_target'] = te_week.fit_transform(rossman_featured['WeekofYear'], rossman_featured['Sales'])

        # apply target encoding to the feature day of the month
        te_day = ce.TargetEncoder(cols=['Day'])
        rossman_featured['Day_target'] = te_day.fit_transform(rossman_featured['Day'], rossman_featured['Sales'])
    else:
        # apply target encoding to the feature Store
        rossman_featured['Store_target'] = te_store.transform(rossman_featured['Store'])

        # apply target encoding to the feature WeekofYear
        rossman_featured['WeekofYear_target'] = te_week.transform(rossman_featured['WeekofYear'])

        # apply target encoding to the feature day of the month
        rossman_featured['Day_target'] = te_day.transform(rossman_featured['Day'])


    # # create new feature dividing sales per customers and store
    # rossman_featured['Sales_Cust_Store']=  rossman_featured['Sales'] / (rossman_featured['Customers'] * rossman_featured['Store'])

    # remove chosen features
    rossman_featured = rossman_featured.drop(['Date',
                                            'Store',
                                            'Year',
                                            'WeekofYear',
                                            'Day',
                                            'Customers',
                                            'CompetitionOpenSinceMonth',
                                            'CompetitionOpenSinceYear',
                                            'Promo2SinceWeek',
                                            'Promo2SinceYear'],
                                             axis=1)
    return rossman_featured, te_store, te_week, te_day