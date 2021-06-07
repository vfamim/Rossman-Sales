import pickle
import inflection
import pandas as pd 
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):
        self.home_path='/media/vfamim2/MEUS PROJETOS DS 2/Rossman/Rossman_sales/'
        self.competition_distance_scaler = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.year = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.competition_time_month = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week = pickle.load(open(self.home_path + 'parameter/competition_time_week.pkl_scaler', 'rb'))
        self.store_type = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))
        
    def data_cleaning(self, df1):
        # list of columns name
        col_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo','StateHoliday', 'SchoolHoliday', 'StoreType',
                    'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 
                    'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
        # loop for snakecase pattern
        snakecase = lambda x: inflection.underscore(x)
        # setting list for new columns 
        col_new = list(map(snakecase, col_old))
        # change names
        df1.columns = col_new
        # fill null values
        df1.competition_distance = df1.competition_distance.fillna(200000)
        # change type to date
        df1['date'] = pd.to_datetime(df1['date'])
        # competition open since month function  
        comp_open_month = lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month']
        df1['competition_open_since_month'] = df1.apply(comp_open_month, axis=1)
        # competition open since year function 
        comp_open_year = lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year']
        df1['competition_open_since_year'] = df1.apply(comp_open_year, axis=1)
        promo2_sw = lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week']
        df1['promo2_since_week'] = df1.apply(promo2_sw, axis=1)
        promo2_sy = lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year']
        df1['promo2_since_year'] = df1.apply(promo2_sy, axis=1)
        # list with months names
        month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # list with number sequence
        number_seq = np.arange(1,13).tolist()
        # converting two lists into a dictionary
        promo_month = dict(zip(number_seq, month_list))
        # converting null values into 0
        df1['promo_interval'].fillna(0, inplace=True)
        # columns with month names
        df1['month_map'] = df1['date'].dt.month.map(promo_month)
        # function to detect promo times
        is_promo_function = lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map']\
            in x['promo_interval'].split(',') else 0
        # new column to promo
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(
            is_promo_function, axis=1)
        # list of columns who need to change type
        change_type = [
            'competition_open_since_month', 'competition_open_since_year',
            'promo2_since_week', 'promo2_since_year'
            ]
        # looping change type
        for x in change_type:
            df1[x] = df1[x].astype(int)
            
        return df1
    
    def feature_engineering(self, df2):
        # year
        df2['year'] = df2.date.dt.year
        # month
        df2['month'] = df2.date.dt.month
        # day 
        df2['day'] = df2.date.dt.day
        # week of year
        df2['week_of_year'] = df2.date.dt.isocalendar().week
        # year week
        df2['year_week'] = df2.date.dt.strftime('%Y-%W')
                # competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],\
            month=x['competition_open_since_month'], day=1), axis=1)
        # competition time month column
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)
        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta( days=7))
        # time of promotion
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)      
        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(
            lambda x: 'public holiday' if x == 'a' else 'easter holiday' \
                if x == 'b' else 'christmas'\
                    if x == 'c' else 'regular day')       
        # closed stores has no sales in that day.
        df2 = df2[df2['open'] != 0]        
        # selecting columns to drop
        cols_drop = ['open', 'promo_interval', 'month_map']
        # drop columns
        df3 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    def data_preparation(self, df5):
        
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)
        # year
        df5['year'] = self.year.fit_transform(df5[['year']].values)    
        # competition time month
        df5['competition_time_month'] = self.competition_time_month.fit_transform(df5[['competition_time_month']].values)
        # promo time week
        df5['promo_time_week'] = self.promo_time_week.fit_transform(df5[['promo_time_week']].values)
        # state holiday - one hot encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])
        # store type - label encoding
        df5.store_type = self.store_type.fit_transform(df5.store_type)
        # assortment - ordinal encoding
        assortment_dict = {'basic' : 1, 'extra': 2, 'extended' : 3}
        df5.assortment = df5.assortment.map(assortment_dict)
        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))
        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))
        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))
        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))
        # columns selection
        cols_selected_boruta = [
            'store',
            'promo',
            'store_type',
            'assortment',
            'competition_distance',
            'competition_open_since_month',
            'competition_open_since_year',
            'promo2',
            'promo2_since_week',
            'promo2_since_year',
            'competition_time_month',
            'promo_time_week',
            'day_of_week_sin',
            'day_of_week_cos',
            'month_sin',
            'month_cos',
            'day_sin',
            'day_cos',
            'week_of_year_sin',
            'week_of_year_cos']
        
        return df5[cols_selected_boruta]

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')