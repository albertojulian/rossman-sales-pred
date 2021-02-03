import category_encoders as ce
from category_encoders import TargetEncoder

# read the cleaned file
rossman_cleaned = pd.read_csv("cleaned.csv", low_memory=False)

# create a new features from Date column
rossman_cleaned['Year'] = rossman_cleaned['Date'].dt.year                       # year
rossman_cleaned['Month'] = rossman_cleaned['Date'].dt.month                     # month 
rossman_cleaned['Day'] = rossman_cleaned['Date'].dt.day                         # day of the month 
rossman_cleaned['WeekofYear'] = rossman_cleaned['Date'].dt.isocalendar().week   # week of the year

# apply one hot encoding to some features
rossman_cleaned = pd.get_dummies(data=rossman_cleaned,
                                 columns=['StateHoliday','StoreType', 'Assortment'],
                                 prefix=['StateHoliday','StoreType', 'Assortment'],
                                 prefix_sep='_')

# apply target enconding to the feature Store
ce_te  = ce.TargetEncoder(cols=['Store'])
rossman_cleaned['Store_target'] = ce_te.fit_transform(rossman_cleaned['Store'], rossman_cleaned['Sales'])

# create new feature dividing sales per customers and store
rossman_cleaned['Sales_Cust_Store']=  rossman_cleaned['Sales'] / (rossman_cleaned['Customers'] * rossman_cleaned['Store'])

# remove chosen features
rossman_cleaned = rossman_cleaned.drop(['Date',
                                        'Store',       
                                        'Year'],
                                         axis=1)

