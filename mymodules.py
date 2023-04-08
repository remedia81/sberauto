import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin


class drop_col_2(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        columns_for_drop = [
            'session_id',
            'client_id',
            'visit_date',
            'visit_time',
            'device_model',
            'device_brand',
            'device_screen_resolution',
            'visit_datetime'
        ]

        x = x.drop(columns_for_drop, axis=1)
        return x[x.columns]


class drop_na_utm(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        set_organic_trafic = {'organic', 'referral', '(none)'}
        utm_col = ['utm_source', 'utm_campaign', 'utm_adcontent', 'utm_keyword']
        for col in utm_col:
            df.loc[(df.utm_medium.isin(set_organic_trafic)) & (df[col].isna()), col] = \
                df[df.utm_medium.isin(set_organic_trafic)][col].describe()['top']
            df.loc[(df.utm_medium.isin(set_organic_trafic) == False) & (df[col].isna()), col] = \
                df[df.utm_medium.isin(set_organic_trafic) == False][col].describe()['top']
        return df[df.columns]


class device_num_pixel(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        df['device_num_pixel'] = df.device_screen_resolution.apply(lambda x: np.nan if x == '(not set)' else
        int(x.split('x')[0]) * int(x.split('x')[1]))
        top_value = df['device_num_pixel'].median()
        df['device_num_pixel'] = df['device_num_pixel'].fillna(top_value)
        return df[df.columns]


class drop_na_dev(TransformerMixin):
    def __init__(self, dict_res_cat, dict_cat_brand, dict_brand_os):
        self.dict_res_cat = dict_res_cat
        self.dict_cat_brand = dict_cat_brand
        self.dict_brand_os = dict_brand_os

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        # Формирование множества уникальных знашений разрашений экранов, для которых заданы непустые значения
        # атрибута device_brand
        set_brand_notna = set(df[(df.device_brand.notna()) & (df.device_brand != '') & (
                df.device_brand != '(not set)')].device_num_pixel.unique())

        # Формирование множества уникальных знашений разрашений экранов, для которых установлены пустые значения
        # атрибута device_brand
        set_brand_isna = set(
            df[(df.device_brand.isna()) | (df.device_brand == '') | (
                    df.device_brand == '(not set)')].device_num_pixel.unique())

        # Формирование множества уникальных значений разрешений экрана, которые входят одновременно в множество
        # set_brand_notna
        # и set_brand_isna, то есть получим множество разрешений, по которым можно восстановить данные
        set_intersec = set_brand_notna & set_brand_isna

        # Заполнение пропусков значений атрибута device_brand для записей датасета, у которых значение атрибута
        # device_num_pixel содержится в множестве set_intersec
        df.loc[((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                df.device_num_pixel.isin(set_intersec) == True), 'device_brand'] = df[
            ((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                    df.device_num_pixel.isin(set_intersec) == True)].apply(
            lambda x: self.dict_res_cat[x.device_num_pixel], axis=1)

        # Заполнение пропусков
        df.loc[((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                (df.device_os.notna()) & (df.device_os != '') & (df.device_os != '(not set)')), 'device_brand'] = df[
            ((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                    (df.device_os.notna()) & (df.device_os != '') & (df.device_os != '(not set)'))].apply(
            lambda x: self.dict_cat_brand[x.device_category][x.device_os], axis=1)

        # Заполнение пропусков
        df.loc[((df.device_os.isna()) | (df.device_os == '') | (df.device_os == '(not set)')) &
               ((df.device_brand.notna()) & (df.device_brand != '') & (df.device_brand != '(not set)')), 'device_os'] =\
            df[((
                    df.device_os.isna()) | (
                        df.device_os == '') | (
                        df.device_os == '(not set)')) & ((
                                                             df.device_brand.notna()) & (
                                                                 df.device_brand != '') & (
                                                                 df.device_brand != '(not set)'))].apply(
                lambda x: self.dict_brand_os[x.device_category][x.device_brand], axis=1)
        df.loc[(df.device_os.isna()) | (df.device_os == '') | (df.device_os == '(not set)'), 'device_os'] = \
            df.device_os.describe()['top']
        df.loc[(df.device_brand.notna()) | (df.device_brand != '') | (df.device_brand != '(not set)'), 'device_brand'] = \
            df.device_brand.describe()['top']
        return df[df.columns]


class clear_outliers(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        df_visit = x.groupby('client_id').aggregate(visit_number=('visit_number', 'max'))
        self.borders_max = df_visit.visit_number.mean() + (3 * df_visit.visit_number.std())
        return self

    def transform(self, df, y=None):
        # Формирование множества клиентов, количество визитов, которых выходит за установленные границы количества
        # визитов
        df.loc[df.visit_number > self.borders_max, 'visit_number'] = round(self.borders_max)
        return df[df.columns]


class new_features_session(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        # Создаем новый атрибут 'visit_datetime' типа timestamp
        df.loc[:, 'visit_datetime'] = df.apply(lambda x: pd.to_datetime(str(x.visit_date) + ' ' + str(x.visit_time)),
                                               axis=1)
        # Создаем новый атрибут 'visit_month', который будет хранить информацию о месяце визита
        df['visit_month'] = df.visit_datetime.dt.month
        # Создаем новый атрибут 'visit_day', который будет хранить информацию о дне визита
        df['visit_day'] = df.visit_datetime.dt.day
        # Создаем новый атрибут 'visit_hour', который будет хранить информацию о часе дня визита
        df['visit_hour'] = df.visit_datetime.dt.hour
        # Преобразование атрибута 'visit_hour' в тип переменной float
        df[['visit_hour', 'visit_day', 'visit_month']] = df[['visit_hour', 'visit_day', 'visit_month']].astype(np.int8)
        # Создаем новый атрибут 'day_name', который будет хранить информацию о наименовании дня недели
        df.loc[:, 'day_name'] = df.loc[:, 'visit_datetime'].apply(lambda x: x.day_name())
        return df[df.columns]


class features_hit(TransformerMixin):
    def __init__(self, df_dict):
        self.df_features = df_dict

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        list_col = ['visit_number', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                    'device_category', 'device_os', 'device_num_pixel', 'device_browser', 'geo_country', 'geo_city']
        df = df.merge(self.df_features, on=list_col, how="left")
        return df[df.columns]


class category_transform(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        categorycal_features = ['hit_referer', 'model', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                                'utm_keyword', 'device_category', 'device_os', 'device_brand', 'device_browser',
                                'geo_country',
                                'geo_city', 'day_name'
                                ]
        df[categorycal_features] = df[categorycal_features].astype('category')
        return df[df.columns]