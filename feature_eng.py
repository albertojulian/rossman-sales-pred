import category_encoders as ce
from category_encoders import TargetEncoder

# read the cleaned file
#rossman_cleaned = pd.read_csv("cleaned.csv", low_memory=False)
rossman_cleaned = clean_rossman(csv_tseries='train.csv', csv_store='store.csv', data_folder='data')

# create a new features from Date column
rossman_cleaned['Year'] = rossman_cleaned['Date'].dt.year                       # year
#rossman_cleaned['Month'] = rossman_cleaned['Date'].dt.month                     # month 
rossman_cleaned['Day'] = rossman_cleaned['Date'].dt.day                         # day of the month 
rossman_cleaned['WeekofYear'] = rossman_cleaned['Date'].dt.isocalendar().week   # week of the year

# convert Week of the Year to integer
rossman_cleaned['WeekofYear'] = rossman_cleaned['WeekofYear'].astype(int)

# apply one hot encoding to some features
rossman_cleaned = pd.get_dummies(data=rossman_cleaned,
                                 columns=['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'DayOfWeek'],
                                 prefix=['StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'DayOfWeek'],
                                 prefix_sep='_')

# apply target enconding to the feature Store
ce_te  = ce.TargetEncoder(cols=['Store'])
rossman_cleaned['Store_target'] = ce_te.fit_transform(rossman_cleaned['Store'], rossman_cleaned['Sales'])

# apply target enconding to the feature WeekofYear
ce_te  = ce.TargetEncoder(cols=['WeekofYear'])
rossman_cleaned['WeekofYear_target'] = ce_te.fit_transform(rossman_cleaned['WeekofYear'], rossman_cleaned['Sales'])

# apply target enconding to the feature day of the month
ce_te  = ce.TargetEncoder(cols=['Day'])
rossman_cleaned['Day_target'] = ce_te.fit_transform(rossman_cleaned['Day'], rossman_cleaned['Sales'])

# create new feature dividing sales per customers and store
rossman_cleaned['Sales_Cust_Store']=  rossman_cleaned['Sales'] / (rossman_cleaned['Customers'] * rossman_cleaned['Store'])

# remove chosen features
rossman_cleaned = rossman_cleaned.drop(['Date',
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