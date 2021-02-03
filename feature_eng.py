import category_encoders as ce
import pandas as pd

# load the cleaned dataframe
def feat_eng(rossman_featured):

    # create a new features from Date column
    rossman_featured['Year'] = rossman_featured['Date'].dt.year                       # year
    #rossman_featured['Month'] = rossman_featured['Date'].dt.month                     # month
    rossman_featured['Day'] = rossman_featured['Date'].dt.day                         # day of the month
    rossman_featured['WeekofYear'] = rossman_featured['Date'].dt.isocalendar().week   # week of the year

    # convert Week of the Year to integer
    rossman_featured['WeekofYear'] = rossman_featured['WeekofYear'].astype(int)

    # apply one hot encoding to some features
    rossman_featured = pd.get_dummies(data=rossman_featured,
                                     columns=['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'DayOfWeek'],
                                     prefix=['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'DayOfWeek'],
                                     prefix_sep='_')

    # apply target enconding to the feature Store
    ce_te  = ce.TargetEncoder(cols=['Store'])
    rossman_featured['Store_target'] = ce_te.fit_transform(rossman_featured['Store'], rossman_featured['Sales'])

    # apply target enconding to the feature WeekofYear
    ce_te  = ce.TargetEncoder(cols=['WeekofYear'])
    rossman_featured['WeekofYear_target'] = ce_te.fit_transform(rossman_featured['WeekofYear'], rossman_featured['Sales'])

    # apply target enconding to the feature day of the month
    ce_te  = ce.TargetEncoder(cols=['Day'])
    rossman_featured['Day_target'] = ce_te.fit_transform(rossman_featured['Day'], rossman_featured['Sales'])

    # create new feature dividing sales per customers and store
    rossman_featured['Sales_Cust_Store']=  rossman_featured['Sales'] / (rossman_featured['Customers'] * rossman_featured['Store'])

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
    return rossman_featured