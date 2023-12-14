import pandas as pd
import numpy as np

import implicit
import lightgbm
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix

import re
import warnings
from pandarallel import pandarallel

from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares

from tqdm import tqdm
tqdm.pandas()
warnings.filterwarnings('ignore')
pandarallel.initialize(use_memory_fs=False, verbose=0)


def reduce_mem_usage(df, verbose=0):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
def prefilter_items(data, take_n_popular=5000, item_features=None):
    # уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.25].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # уберем самые НЕ популярные товары (их и так НЕ купят)
    top_not_popular = popularity[popularity['share_unique_users'] < 0.03].item_id.tolist()
    data = data[~data['item_id'].isin(top_not_popular)]

    # уберем не интересные для рекомендаций категории (department)
    if item_features is not None:
        dep_size = pd.DataFrame(item_features. \
                                groupby('department')['item_id'].nunique(). \
                                sort_values(ascending=False)).reset_index()

        dep_size.columns = ['department', 'n_items']
        not_popular_departments = dep_size[dep_size['n_items'] < 180].department.tolist()
        items_in_not_popular_departments = item_features[
            item_features['department'].isin(not_popular_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_not_popular_departments)]

    # уберем товары, которые не продавались за последние 12 месяцев
    data = data[data['week_no'] >= data['week_no'].max() - 12]

    # уберем слишком дорогие товары
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] < 35]

    # уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data = data[data['price'] > 1]

    # возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data

def add_features(data, 
                 train, 
                 recommender, 
                 item_features, 
                 user_features,
                 items_embeddings_df,
                 users_embeddings_df,
                 N=50):

    target = get_targets(data, train, recommender, N)
    user_features = get_new_user_features(data, user_features, users_embeddings_df)
    item_features = get_new_item_features(data, item_features, items_embeddings_df)
    item_features = data.merge(item_features, on='item_id', how='left')

    output_data = item_features.merge(user_features, on='user_id', how='left')

    # коэффициент количества покупок товаров в данной категории к среднему количеству покупок
    count_perchases = output_data.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'quantity': 'mean'}) \
        .reset_index().rename(columns={'quantity': 'count_purchases_week_dep'})

    mean_count_perch = output_data.groupby(['commodity_desc', 'week_no']).agg({'quantity': 'sum'}) \
        .reset_index().rename(columns=({'quantity': 'mean_count_purchases_week_dep'}))

    k = count_perchases.merge(mean_count_perch, on=['commodity_desc', 'week_no'], how='left')
    k['count_purchases_week_mean'] = k['count_purchases_week_dep'] / k['mean_count_purchases_week_dep']
    k = k[['user_id', 'commodity_desc', 'count_purchases_week_mean']]

    tmp = k.groupby(['user_id', 'commodity_desc']).agg({'count_purchases_week_mean': 'mean'}) \
        .reset_index()

    output_data = output_data.merge(tmp, on=['user_id', 'commodity_desc'], how='left')

    # вычислим коэффициент отношения суммы покупок товаров в данной категории к средней сумме
    count_perchases = output_data.groupby(['user_id', 'commodity_desc', 'week_no']).agg({'price': 'sum'}) \
        .reset_index().rename(columns={'price': 'price_week'})

    mean_count_perch = output_data.groupby(['commodity_desc', 'week_no']).agg({'price': 'sum'}) \
        .reset_index().rename(columns=({'price': 'mean_price_week'}))

    k = count_perchases.merge(mean_count_perch, on=['commodity_desc', 'week_no'], how='left')
    k['sum_purchases_week_mean'] = k['price_week'] / k['mean_price_week']
    k = k[['user_id', 'commodity_desc', 'sum_purchases_week_mean']]

    tmp = k.groupby(['user_id', 'commodity_desc']).agg({'sum_purchases_week_mean': 'mean'}).reset_index()

    output_data = output_data.merge(tmp, on=['user_id', 'commodity_desc'], how='left')
    output_data = output_data.merge(target, on=['item_id', 'user_id'], how='left')

    # заполним пустые ячейки нулями
    output_data = output_data.fillna(0)

    return output_data

def get_targets(data, train, recommender, N):
    """Подготовка обучающего датасета, разбиение на X и y"""

    users = pd.DataFrame(data['user_id'].unique())

    users.columns = ['user_id']

    train_users = train['user_id'].unique()
    users = users[users['user_id'].isin(train_users)]

    # составим рекомендации на основе собственных покупок
    users['candidates'] = users['user_id'].parallel_apply(
        lambda x: recommender.get_own_recommendations(x, N=N)
    )

    s = users.parallel_apply(
        lambda x: pd.Series(x['candidates']), axis=1
    ).stack().reset_index(level=1, drop=True)

    s.name = 'item_id'

    users = users.drop('candidates', axis=1).join(s)

    users['flag'] = 1

    targets = data[['user_id', 'item_id']].copy()
    targets.head(3)

    targets['target'] = 1

    targets = users.merge(targets, on=['user_id', 'item_id'], how='left')

    targets['target'].fillna(0, inplace=True)
    targets.drop('flag', axis=1, inplace=True)

    return targets

def get_new_user_features(data, user_feats, users_embeddings_df):
    """Создает новые признаки для пользователей"""

    data['price'] = data['sales_value'] / data['quantity']
    new_user_features = user_feats.merge(data, on='user_id', how='left')

    # создадим эмбеддинги
    user_feats = user_feats.merge(users_embeddings_df, how='left')

    # стандартное время продажи
    trans_time = new_user_features.groupby('user_id')['trans_time'].mean().reset_index()
    trans_time.rename(columns={'trans_time': 'mean_time'}, inplace=True)
    trans_time = trans_time.astype(np.float32)
    user_feats = user_feats.merge(trans_time, how='left')

    # возраст
    user_feats['age'] = user_feats['age_desc'].replace(
        {'65+': 65, '45-54': 50, '25-34': 30, '35-44': 40, '19-24': 22, '55-64': 60}
    )
    user_feats = user_feats.drop('age_desc', axis=1)

    # доход
    user_feats['income'] = user_feats['income_desc'].replace(
        {'35-49K': 42,
         '50-74K': 62,
         '25-34K': 30,
         '75-99K': 87,
         'Under 15K': 15,
         '100-124K': 112,
         '15-24K': 20,
         '125-149K': 137,
         '150-174K': 162,
         '250K+': 250,
         '175-199K': 187,
         '200-249K': 225}
    )
    user_feats = user_feats.drop('income_desc', axis=1)

    # дети
    user_feats['children'] = 0
    user_feats.loc[(user_feats['kid_category_desc'] == '1'), 'children'] = 1
    user_feats.loc[(user_feats['kid_category_desc'] == '2'), 'children'] = 2
    user_feats.loc[(user_feats['kid_category_desc'] == '3'), 'children'] = 3
    user_feats = user_feats.drop('kid_category_desc', axis=1)

    # вычислим средний чек и средний чек в неделю
    basket = new_user_features.groupby(['user_id'])['price'].sum().reset_index()

    baskets = new_user_features.groupby('user_id')['basket_id'].count().reset_index()
    baskets.rename(columns={'basket_id': 'baskets'}, inplace=True)

    avr_basket = basket.merge(baskets)

    avr_basket['avr_bask'] = avr_basket.price / avr_basket.baskets
    avr_basket['sum_per_week'] = avr_basket.price / new_user_features.week_no.nunique()

    avr_basket = avr_basket.drop(['price', 'baskets'], axis=1)
    user_feats = user_feats.merge(avr_basket, how='left')

    return user_feats

def get_new_item_features(data, item_feats, items_embeddings_df):
    """Новые признаки для продуктов"""
    new_feats = item_feats.merge(data, on='item_id', how='left')

    # создадим эмбеддинги
    item_feats = item_feats.merge(items_embeddings_df, how='left')

    # обработаем manufacturer
    not_popular_manufacturer = item_feats.manufacturer.value_counts()[item_feats.manufacturer.value_counts() < 40].index
    item_feats.loc[item_feats.manufacturer.isin(not_popular_manufacturer), 'manufacturer'] = 999999999
    item_feats.manufacturer = item_feats.manufacturer.astype('object')

    # обработаем discount
    mean_disc = new_feats.groupby('item_id')['coupon_disc'].mean().reset_index().sort_values('coupon_disc')
    item_feats = item_feats.merge(mean_disc, on='item_id', how='left')

    # вычислим среднее количество продаж товара в категории в неделю
    items_in_dept = new_feats.groupby('department')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_dept.rename(columns={'item_id': 'items_in_department'}, inplace=True)

    sales_count_per_dept = new_feats.groupby(['department'])['quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    sales_count_per_dept.rename(columns={'quantity': 'sales_count_per_dep'}, inplace=True)

    items_in_dept = items_in_dept.merge(sales_count_per_dept, on='department')
    items_in_dept['qnt_of_sales_per_item_per_dep_per_week'] = (
            items_in_dept['sales_count_per_dep'] /
            items_in_dept['items_in_department'] /
            new_feats['week_no'].nunique()
    )
    items_in_dept = items_in_dept.drop(['items_in_department'], axis=1)
    item_feats = item_feats.merge(items_in_dept, on=['department'], how='left')

    # вычислим количество продаж и среднее количество продаж товара
    item_qnt = new_feats.groupby(['item_id'])['quantity'].count().reset_index()
    item_qnt.rename(columns={'quantity': 'quantity_of_sales'}, inplace=True)

    item_qnt['sales_count_per_week'] = item_qnt['quantity_of_sales'] / new_feats['week_no'].nunique()
    item_feats = item_feats.merge(item_qnt, on='item_id', how='left')

    # обработаем sub_commodity_desc
    items_in_dept = new_feats.groupby('sub_commodity_desc')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_dept.rename(columns={'item_id': 'items_in_sub_commodity_desc'}, inplace=True)

    sales_count_per_dept = new_feats.groupby(['sub_commodity_desc'])[
        'quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    sales_count_per_dept.rename(columns={'quantity': 'qnt_of_sales_per_sub_commodity_desc'}, inplace=True)

    items_in_dept = items_in_dept.merge(sales_count_per_dept, on='sub_commodity_desc')
    items_in_dept['qnt_of_sales_per_item_per_sub_commodity_desc_per_week'] = (
            items_in_dept['qnt_of_sales_per_sub_commodity_desc'] /
            items_in_dept['items_in_sub_commodity_desc'] /
            new_feats['week_no'].nunique()
    )
    items_in_dept = items_in_dept.drop(['items_in_sub_commodity_desc'], axis=1)
    item_feats = item_feats.merge(items_in_dept, on=['sub_commodity_desc'], how='left')

    return item_feats

def get_important_features(model, X_train, y_train):
    """Возвращает важные фичи"""

    model.fit(X_train, y_train)
    feat = list(zip(X_train.columns.tolist(), model.feature_importances_))
    feat = pd.DataFrame(feat, columns=['feature', 'value'])
    important_features = feat.loc[feat.value > 0, 'feature'].tolist()
    
    return important_features

def get_final_recs(X_test, test_preds_proba, val_2, data_train, item_features):
    """Возвращает финальный список рекомендованных товаров"""

    X_test['predict_proba'] = test_preds_proba

    X_test.sort_values(['user_id', 'predict_proba'], ascending=False, inplace=True)
    recs = X_test.groupby('user_id')['item_id']
    recs_lst = []
    
    for user, preds in recs:
        recs_lst.append({'user_id': user, 'recommendations': preds.tolist()})

    recs_lst = pd.DataFrame(recs_lst)

    res_2 = val_2.groupby('user_id')['item_id'].unique().reset_index()
    res_2.columns = ['user_id', 'actual']

    final_recommendations = res_2.merge(recs_lst, how='left')
    final_recommendations['recommendations'] = final_recommendations['recommendations'].fillna(0)

    price = data_train.groupby('item_id')['price'].mean().reset_index()

    popularity_recs = get_popularity_recommendations(data_train, n=500)
    popularity_recs_lst = []
    [popularity_recs_lst.append(item) for item in popularity_recs if price \
        .loc[price['item_id'] == item]['price'].values > 1]

    # progress_apply
    final_recommendations['recommendations'] = final_recommendations.parallel_apply \
        (lambda x: postfilter_items(x, 
                                    item_info=item_features, 
                                    data_train=data_train, 
                                    price=price,
                                    list_pop_rec=popularity_recs_lst, 
                                    N=5), 
         axis=1)

    return final_recommendations

def get_popularity_recommendations(data, n=5):
    """ Топ-n популярных товаров"""

    popular_items = data.groupby('item_id')['quantity'].count().reset_index()
    popular_items.sort_values('quantity', ascending=False, inplace=True)
    popular_items = popular_items[popular_items['item_id'] != 999999]
    
    return popular_items.head(n).item_id.tolist()

def postfilter_items(row, item_info, data_train, price, list_pop_rec, N=5):
    """ Выполняет пост-фильтрацию товаров
    Входные параметры
    -----
    row - строка датасета
    item_info: pd.DataFrame - датафрейм с информацией о товарах
    data_train: pd.DataFrame - обучающий датафрейм
    """
    
    rec = row['recommendations']
    purch_goods = data_train.loc[data_train['user_id'] == row['user_id']]['item_id'].unique()

    if rec == 0:
        rec = list_pop_rec

    # список уникальных рекомендаций
    unique_recs = []
    [unique_recs.append(item) for item in rec if item not in unique_recs]

    # товары должны быть дороже 1
    price_recommendations = []
    [price_recommendations.append(item) for item in unique_recs if price \
        .loc[price['item_id'] == item]['price'].values > 1]

    # один товар должен быть дороже 7
    expensive_items = []
    [expensive_items.append(item) for item in price_recommendations if price. \
        loc[price['item_id'] == item]['price'].values > 7]
    
    if len(expensive_items) == 0:
        [expensive_items.append(item) for item in list_pop_rec if price. \
            loc[price['item_id'] == item]['price'].values > 7]

    # товар, который пользователь ещё не покупал
    new_items = []
    [new_items.append(item) for item in price_recommendations if item not in purch_goods]

    # промежуточный итог
    recommendations = []
    recommendations.append(expensive_items[0] if len(expensive_items) > 0 else list_pop_rec[0])
    recommendations += new_items
    recommendations = cat_filter(recommendations, item_info=item_info)[0:3]
    recommendations += price_recommendations
    output_recommendations = cat_filter(recommendations, item_info=item_info)
    
    recs_count = len(output_recommendations)
    if recs_count < N:
        output_recommendations.extend(list_pop_rec[:N - recs_count])
    else:
        output_recommendations = output_recommendations[:N]

    assert len(output_recommendations) == N, 'Кол-во рекомендаций != {}'.format(N)

    return output_recommendations

def cat_filter(list_recommendations, item_info):
    """Получение списка товаров из уникальных категорий"""

    output_recommendations = []

    used_cats = []

    for item in list_recommendations:
        cat = item_info.loc[item_info['item_id'] == item, 'sub_commodity_desc'].values[0]

        if cat not in used_cats:
            output_recommendations.append(item)
            used_cats.append(cat)

    return output_recommendations
