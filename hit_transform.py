import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


class dropna_hit(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        # Формирование справочника номеров событий и соответствующих им средних значений времен совершения событий
        dict_num_time = {}
        for num in range(1, x.hit_number.max() + 1):
            dict_num_time[num] = x[(x.hit_number == num) & (x.hit_time.notna())].hit_time.mean()
        # Заполнение пропусков значений атрибута hit_time
        x.loc[x.hit_time.isna(), 'hit_time'] = x[x.hit_time.isna()].apply(lambda i: dict_num_time[i.hit_number], axis=1)
        # Замена пропусков значений атрибута hit_referer наиболее частым значением
        x.loc[x.hit_referer.isna(), 'hit_referer'] = x[x.hit_referer.notna()].hit_referer.describe()['top']
        # Замена пропусков значений атрибута event_label наиболее частым значением
        x.loc[x.event_label.isna(), 'event_label'] = x[x.event_label.notna()].event_label.describe()['top']
        return x


class clear_outliers_hit_number(TransformerMixin):
    def __init__(self, target_action):
        self.target_action = target_action

    def fit(self, x, y=None):
        x_gr = x[x.event_action.isin(self.target_action)].groupby('session_id').aggregate({'hit_number': 'min'})
        iqr = x_gr['hit_number'].quantile(0.75) - x_gr['hit_number'].quantile(0.25)
        boundaries = ((x_gr['hit_number'].quantile(0.25) - (1.5 * iqr)), (x_gr['hit_number'].quantile(0.75) +
                                                                          (1.5 * iqr)))
        self.boundaries = boundaries
        return self

    def transform(self, x, y=None):
        boundaries = self.boundaries
        # Сформируем новый датасет без выбросов в номерах событий
        x = x[x.hit_number <= boundaries[1]]
        return x


class hit_time_coef(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x['hit_time_coef'] = x.apply(lambda i: i.hit_time / i.hit_number, axis=1)
        return x


class clear_outliers_hit_time_coef(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        iqr = x['hit_time_coef'].quantile(0.75) - x['hit_time_coef'].quantile(0.25)
        boundaries = ((x['hit_time_coef'].quantile(0.25) - (1.5 * iqr)), (x['hit_time_coef'].quantile(0.75) +
                                                                          (1.5 * iqr)))
        self.boundaries = boundaries
        return self

    def transform(self, x, y=None):
        boundaries = self.boundaries
        # Так как выбросы на макcимальной стороне, то заменим значения выбросы атрибута hit_time_coef на максимальную
        # границу borders_time
        x.loc[x.hit_time_coef > boundaries[1], 'hit_time_coef'] = boundaries[1]
        return x


class features_new_hit(TransformerMixin):
    def __init__(self, target_action):
        self.target_action = target_action

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        # Формирование справочника номеров событий и соответствующих им средних значений времен совершения событий
        x['event_value'] = np.where(x['event_action'].isin(self.target_action) == False, 0, 1)
        for item in x.index:
            string_path = x.loc[item, 'hit_page_path']
            if len(string_path.split('/')) > 2:
                if (string_path.find('/cars/') != -1) and (string_path.find('/all/') == -1):
                    num_position = string_path.find('/cars/') + len('/cars/')
                    new_string = string_path[num_position:len(string_path)]
                    if new_string.find('/') != -1:
                        new_string = new_string[0:new_string.find('/')]
                        if new_string.find('?') != -1:
                            new_string = new_string[0:new_string.find('?')]
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                        else:
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                    else:
                        if new_string.find('?') != -1:
                            new_string = new_string[:new_string.find('?')]
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                        else:
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                elif (string_path.find('/cars/') != -1) and (string_path.find('/all/') != -1):
                    num_position = string_path.find('/all/') + len('/all/')
                    new_string = string_path[num_position:len(string_path)]
                    if new_string.find('/') != -1:
                        new_string = new_string[:new_string.find('/')]
                        if new_string.find('?') != -1:
                            new_string = new_string[:new_string.find('?')]
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                        else:
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                    else:
                        if new_string.find('?') != -1:
                            new_string = new_string[:new_string.find('?')]
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                        else:
                            if (new_string != '') and not any(x.isdigit() for x in new_string):
                                model_name = new_string
                            else:
                                model_name = np.nan
                else:
                    model_name = np.nan
            else:
                model_name = np.nan
            if str(model_name) != 'nan':
                model_position = string_path.find(model_name) + len(model_name) + 1
                new_string = string_path[model_position:len(string_path)]
                if new_string.find('/') != -1:
                    new_string = new_string[:new_string.find('/')]
                    if new_string.find('?') != -1:
                        type_model = new_string[:new_string.find('?')]
                    elif (new_string.find('&') != -1) or (new_string.find('=') != -1):
                        type_model = np.nan
                    else:
                        type_model = new_string
                elif new_string.find('?') != -1:
                    new_string = new_string[:new_string.find('?')]
                    if (new_string.find('&') != -1) or (new_string.find('=') != -1):
                        type_model = np.nan
                    else:
                        type_model = new_string
                elif (new_string.find('&') != -1) or (new_string.find('=') != -1):
                    type_model = np.nan
                else:
                    type_model = new_string
                if type_model == '':
                    type_model = np.nan
                type_name = type_model
                if str(type_name) != 'nan':
                    full_model = str(model_name) + ' ' + str(type_name)
                else:
                    full_model = str(model_name)
            else:
                full_model = str(model_name)
            x.loc[item, 'model'] = full_model
        x.loc[x.model == 'https:'] = np.nan
        x.loc[x.model == 'lada_(vaz)'] = 'lada-vaz'
        x['model'] = x.model.fillna('(none)')
        return x


class transform_hit(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        df = df.groupby('session_id', as_index=False).aggregate(
            hit_referer=('hit_referer', lambda x: x.unique()[0]),
            hit_time_max=('hit_time', 'max'),
            hit_number_max=('hit_number', 'max'),
            hit_time_coef_mean=('hit_time_coef', 'mean'),
            model_count=('model', pd.Series.nunique),
            model=('model', lambda x: x.unique()[0]),
            event_value=('event_value', 'max')
        )
        df['model'] = df.model.fillna('(none)')
        return df[df.columns]


class drop_na_utm_trans(TransformerMixin):
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


class device_num_pixel_trans(TransformerMixin):
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


class drop_na_dev_trans(TransformerMixin):
    def __init__(self):
        pass

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
        dict_res_cat = {}
        for res_cat in set_intersec:
            dict_res_cat[res_cat] = df[(df.device_num_pixel == res_cat) &
                                       ((df.device_brand.notna()) & (df.device_brand != '') &
                                        (df.device_brand != '(not set)'))].device_brand.describe()['top']
        # Заполнение пропусков значений атрибута device_brand для записей датасета, у которых значение атрибута
        # device_num_pixel содержится в множестве set_intersec
        df.loc[((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                df.device_num_pixel.isin(set_intersec) == True), 'device_brand'] = df[
            ((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                    df.device_num_pixel.isin(set_intersec) == True)].apply(
            lambda x: dict_res_cat[x.device_num_pixel], axis=1)
        # Формируем список ОС, которые встречаются а качестве значений атрибута device_os оставшихся с пропусками
        # записей датасета
        list_os = []
        list_os = df[(df.device_brand.isna() | (df.device_brand == '') | (
                df.device_brand == '(not set)')) & ((df.device_os.notna()) & (
                df.device_os != '') & (df.device_os != '(not set)'))].device_os.unique()
        # Формирование справочника ОС и соответствующие им топовые значения брендов
        dict_cat_brand = {}
        for cat in df.device_category.unique():
            dict_os_brand = {}
            for os in list_os:
                dict_os_brand[os] = df[(df.device_category == cat) & (df.device_os == os) & (
                        (df.device_brand.notna()) & (df.device_brand != '') & (
                        df.device_brand != '(not set)'))].device_brand.describe()['top']
            dict_cat_brand[cat] = dict_os_brand
        # Заполнение пропусков
        df.loc[((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                (df.device_os.notna()) & (df.device_os != '') & (df.device_os != '(not set)')), 'device_brand'] = df[
            ((df.device_brand.isna()) | (df.device_brand == '') | (df.device_brand == '(not set)')) & (
                    (df.device_os.notna()) & (df.device_os != '') & (df.device_os != '(not set)'))].apply(
            lambda x: dict_cat_brand[x.device_category][x.device_os], axis=1)
        # Формирование множества уникальных значенией атрибута device_brand, у которых установлены пропуски для
        # атрибута device_os
        set_client_os_na = set(df[((df.device_os.isna()) | (df.device_os == '') | (df.device_os == '(not set)')) &
                                  (df.device_brand.notna()) & (df.device_brand != '') &
                                  (df.device_brand != '(not set)')].device_brand.unique())
        # Формирование справочника категорий, который соответствуют справочники брендов с топовыми значениями атрибутов
        # device_os
        dict_brand_os = {}
        for cat in df.device_category.unique():
            dict_band_cat_os = {}
            for brand in set_client_os_na:
                dict_band_cat_os[brand] = df[(df.device_brand == brand) & ((df.device_os.notna()) &
                                                                           (df.device_os != '') &
                                                                           (df.device_os != '(not set)'))].device_os.describe()['top']
            dict_brand_os[cat] = dict_band_cat_os
        # Заполнение пропусков
        df.loc[((df.device_os.isna()) | (df.device_os == '') | (df.device_os == '(not set)')) &
               ((df.device_brand.notna()) & (df.device_brand != '') & (df.device_brand != '(not set)')), 'device_os'] = \
            df[((
                    df.device_os.isna()) | (
                        df.device_os == '') | (
                        df.device_os == '(not set)')) & ((
                                                             df.device_brand.notna()) & (
                                                                 df.device_brand != '') & (
                                                                 df.device_brand != '(not set)'))].apply(
                lambda x: dict_brand_os[x.device_category][x.device_brand], axis=1)
        df.loc[(df.device_os.isna()) | (df.device_os == '') | (df.device_os == '(not set)'), 'device_os'] = \
            df.device_os.describe()['top']
        df.loc[(df.device_brand.notna()) | (df.device_brand != '') | (df.device_brand != '(not set)'), 'device_brand'] = \
            df.device_brand.describe()['top']
        joblib.dump({
            'dict_res_cat': dict_res_cat,
            'dict_cat_brand': dict_cat_brand,
            'dict_brand_os': dict_brand_os
        }, 'data/dicts.pkl')
        return df


class clear_outliers_session(TransformerMixin):
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


def main():
    df_session = pd.read_pickle('data/ga_sessions.pkl')
    df_hit = pd.read_pickle('data/ga_hits.pkl')

    target_action = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                     'sub_open_dialog_click', 'sub_custom_question_submit_click',
                     'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                     'sub_car_request_submit_click']
    pipe_hit = Pipeline(steps=[
        ('drop_hit', dropna_hit()),
        ('clear_hit_number', clear_outliers_hit_number(target_action=target_action)),
        ('add_coef', hit_time_coef()),
        ('clear_outliers', clear_outliers_hit_time_coef()),
        ('new_features', features_new_hit(target_action=target_action)),
        ('transform_hit', transform_hit())
    ])
    pipe_hit.fit(df_hit, y=None)
    df_hit = pipe_hit.transform(df_hit)
    df_for_model = df_session.merge(df_hit[['session_id', 'event_value']], on='session_id', how="inner").copy()
    df_for_model.to_pickle('data/df_ready_pycharm.pkl')
    del df_for_model
    pipe = Pipeline(steps=[
        ('drop_utm', drop_na_utm_trans()),
        ('pixel_num', device_num_pixel_trans()),
        ('drop_dev', drop_na_dev_trans()),
        ('clear_outliers', clear_outliers_session())
    ])
    df_for_dict = df_session.merge(df_hit, on='session_id', how="inner").copy()
    pipe.fit(df_for_dict, y=None)
    df_for_dict = pipe.transform(df_for_dict)

    list_col = ['visit_number', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                'device_category', 'device_os', 'device_num_pixel', 'device_browser', 'geo_country', 'geo_city']
    df_dict_gr = df_for_dict.groupby(list_col, as_index=False).agg(hit_referer=('hit_referer', lambda x: x.mode()[0]),
                                                                   hit_time_max_mean=('hit_time_max', 'mean'),
                                                                   hit_number_max_mean=('hit_number_max', 'median'),
                                                                   hit_time_coef_mean=('hit_time_coef_mean', 'mean'),
                                                                   model_count_mean=('model_count', 'median'),
                                                                   model=('model', lambda x: x.mode()[0]))
    df_dict_gr.to_pickle('data/df_dict_gr.pkl')


if __name__ == '__main__':
    main()
