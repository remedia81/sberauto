import joblib
import dill
import pandas as pd
import datetime as dt
import category_encoders as ce
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from mymodules import drop_na_utm
from mymodules import drop_na_dev
from mymodules import drop_col_2
from mymodules import device_num_pixel
from mymodules import clear_outliers
from mymodules import category_transform
from mymodules import new_features_session
from mymodules import features_hit

dicts = joblib.load('data/dicts.pkl')
df_features = pd.read_pickle('data/df_dict_gr.pkl')


def main():
    df = pd.read_pickle('data/df_ready_pycharm.pkl')

    print(df.shape)
    continuous_features = ['hit_time_max_mean', 'hit_number_max_mean', 'hit_time_coef_mean', 'model_count_mean',
                           'visit_number', 'device_num_pixel', 'visit_month', 'visit_day', 'visit_hour']
    nominal_features = ['hit_referer', 'model', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                        'utm_keyword', 'device_category', 'device_os', 'device_browser', 'geo_country',
                        'geo_city']
    ordinary_features = ['day_name']

    x = df.drop('event_value', axis=1)
    y = df['event_value']

    numerical_transformer = imbPipeline(steps=[
        ('imputer_1', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_ordinary = imbPipeline(steps=[
        ('imputer_2', SimpleImputer(strategy='most_frequent')),
        ('encoder_1', OrdinalEncoder())
    ])

    categorical_nominal = imbPipeline(steps=[
        ('imputer_3', SimpleImputer(strategy='most_frequent')),
        ('encoder_2', ce.cat_boost.CatBoostEncoder())
    ])


    preprocessor = ColumnTransformer(transformers=[
        ('categorical_ordinary', categorical_ordinary, ordinary_features),
        ('categorical_nominal', categorical_nominal, nominal_features),
        ('numerical', numerical_transformer, continuous_features)
    ])
    dict_res_cat = dicts['dict_res_cat']
    dict_cat_brand = dicts['dict_cat_brand']
    dict_brand_os = dicts['dict_brand_os']


    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(n_estimators=109, bootstrap=False, max_depth=44, random_state=36),
        KNeighborsClassifier(n_neighbors=5),
        Perceptron(penalty='elasticnet')
    )
    best_pipe = None
    best_scores = .0
    best_diff = .0
    for model in models:
        pipe = imbPipeline(steps=[
            ('drop_utm', drop_na_utm()),
            ('pixel_num', device_num_pixel()),
            ('drop_dev', drop_na_dev(dict_res_cat=dict_res_cat, dict_cat_brand=dict_cat_brand,
                                     dict_brand_os=dict_brand_os)),
            ('clear_outliers', clear_outliers()),
            ('add_features', new_features_session()),
            ('features_hit', features_hit(df_dict=df_features)),
            ('category_transform', category_transform()),
            ('filter_2', drop_col_2()),
            ('preprocessor', preprocessor),
            ('smote', SMOTE()),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, x, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() - score.std() > best_diff:
            best_scores = score.mean()
            best_pipe = pipe
            best_diff = score.mean() - score.std()

    best_pipe.fit(x, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_scores:.4f}')
    with open('data/model_3.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'event_value prediction',
                'author': 'Vladimir Melnichuk',
                'version': 1,
                'date': dt.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_scores
            }
        }, file)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    lr_probs = best_pipe.predict_proba(x_test)
    lr_probs = lr_probs[:, 1]  # сохраняем вероятности только для положительного исхода
    fpr, tpr, treshold = roc_curve(y_test, lr_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
