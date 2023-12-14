import pandas as pd
import numpy as np
import implicit as implicit
import lightgbm
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix
import re
import warnings
from pandarallel import pandarallel
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from scipy.sparse import csr_matrix
import time
from multiprocessing import cpu_count

tqdm.pandas()
warnings.filterwarnings('ignore')
pandarallel.initialize(use_memory_fs=False, verbose=0)



class MainRecommender:
    '''Рекомендации, которые можно получить из ALS'''

    def __init__(self, data, weighting=True, n_factors=20, regularization=0.001, iterations=15, verbose=0):
        self.verbose = verbose # выводить ли данные о прогрессе выполнения и ошибках
        self.check_implicit_version()

        # топ покупок каждого полоьзователя
        self.top_purch = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purch.sort_values('quantity', ascending=False, inplace=True)
        self.top_purch = self.top_purch[self.top_purch['item_id'] != 999999]

        # топ покупок по всему датасету
        self.overall_top_purch = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purch.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purch = self.overall_top_purch[self.overall_top_purch['item_id'] != 999999]
        self.overall_top_purch = self.overall_top_purch.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix, 
                              n_factors=n_factors, 
                              regularization=regularization, 
                              iterations=iterations)
        self.own_rec = self.fit_own_recommender(self.user_item_matrix)

        self.item_factors = self.model.item_factors
        self.user_factors = self.model.user_factors

        self.items_embeddings_df, self.users_embeddings_df = self.get_embeddings()

        
    def num_threads(self):
        return int(cpu_count()/2)

    
    def is_implicit_version_good(self):
        return int(implicit.__version__.split('.')[1]) < 5

        
    def check_implicit_version(self):
        is_good = self.is_implicit_version_good()
        
        if not is_good:
            print(f'Текущая версия implicit = {implicit.__version__}. ', end='')
            print(f'Может упасть ядро Python, если матрица будет слишком большой.', end='')
            print(f'Стоит установить старую версию implicit командой: pip install implicit==0.4.4')

    
    def get_embeddings(self):
        items_embeddings = self.item_factors
        items_embeddings_df = pd.DataFrame(items_embeddings)
        items_embeddings_df.reset_index(inplace=True)
        items_embeddings_df['item_id'] = items_embeddings_df['index'].parallel_apply(lambda x: self.id_to_itemid[x])
        items_embeddings_df = items_embeddings_df.drop('index', axis=1)

        users_embeddings = self.user_factors
        users_embeddings_df = pd.DataFrame(users_embeddings)
        users_embeddings_df.reset_index(inplace=True)
        users_embeddings_df['user_id'] = users_embeddings_df['index'].parallel_apply(lambda x: self.id_to_userid[x])
        users_embeddings_df = users_embeddings_df.drop('index', axis=1)

        return items_embeddings_df, users_embeddings_df


    def _prepare_matrix(self, data):
        '''Готовит user-item матрицу'''
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity', 
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    
    def _prepare_dicts(self, user_item_matrix):
        '''Подготавливает вспомогательные словари'''

        user_ids = user_item_matrix.index.values
        item_ids = user_item_matrix.columns.values

        matrix_user_ids = np.arange(len(user_ids))
        matrix_item_ids = np.arange(len(item_ids))

        id_to_item_id = dict(zip(matrix_item_ids, item_ids))
        id_to_user_id = dict(zip(matrix_user_ids, user_ids))

        itemid_to_id = dict(zip(item_ids, matrix_item_ids))
        userid_to_id = dict(zip(user_ids, matrix_user_ids))

        return id_to_item_id, id_to_user_id, itemid_to_id, userid_to_id

    
    def fit_own_recommender(self, user_item_matrix):
        '''Обучает модель, которая рекомендует товары среди купленных товаров'''

        own_rec = ItemItemRecommender(K=1, num_threads=self.num_threads())
        if self.is_implicit_version_good():
            own_rec.fit(csr_matrix(user_item_matrix).T.tocsr())
        else:
            own_rec.fit(csr_matrix(user_item_matrix).T.tocsr(), verbose=int(self.verbose)) # для implicit >= 0.5.0

        return own_rec

    
    def fit(self, user_item_matrix, n_factors=20, regularization=0.001, iterations=15):
        '''Обучает ALS'''

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=self.num_threads())
        if self.is_implicit_version_good():
            model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=bool(self.verbose))
        else:
            model.fit(csr_matrix(user_item_matrix).tocsr() , show_progress=bool(self.verbose)) # для implicit >= 0.5.0

        return model


    def _update_dict(self, user_id):
        '''Обновляет данные словарей user'''

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})


    def _get_similar_item(self, item_id):
        '''Находит товар, похожий на item_id'''

        # если товар похож на себя, то рекомендуем 2 товара
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  
        top_rec = recs[1][0] # берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]


    def _extend_with_top_popular(self, recs, N=5):
        '''Если кол-во рекомендаций < N, то дополняем их топ-популярными'''

        if len(recs) < N:
            recs.extend(self.overall_top_purch[:N])
            recs = recs[:N]

        return recs


    def _get_recommendations(self, user, model, N=5):
        '''Рекомендации через стандартные библиотеки implicit'''
        
        self._update_dict(user_id=user)
        try:
            filter_items = False # для implicit >= 0.5.0. Иначе будет падать ядро
            if self.is_implicit_version_good(): # для implicit < 0.5.0
                filter_items=[self.itemid_to_id[999999]]
                
            res = [self.id_to_itemid[rec[0]] for rec in
                   model.recommend(userid=self.userid_to_id[user],
                                   user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                   N=N,
                                   filter_already_liked_items=False,
                                   filter_items=filter_items,
                                   recalculate_user=True)]
            res = self._extend_with_top_popular(res, N=N)

        except:
            print('error implicit version!')
            res = self.overall_top_purch[:N]

        return res


    def get_als_recommendations(self, user, N=5):
        '''Рекомендации через стардартные библиотеки implicit'''
        return self._get_recommendations(user, model=self.model, N=N)


    def get_own_recommendations(self, user, N=5):
        '''Рекомендуем товары среди тех, которые юзер уже купил'''
        return self._get_recommendations(user, model=self.own_rec, N=N)


    def get_similar_items_recommendation(self, user, N=5):
        '''Рекомендуем товары, похожие на топ-N купленных юзером товаров'''

        top_users_purch = self.top_purch[self.top_purch['user_id'] == user].head(N)

        res = top_users_purch['item_id'].parallel_apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


    def get_similar_users_recommendation(self, user, N=5):
        '''Рекомендуем топ-N товаров, среди купленных похожими пользователями'''

        result = []

        self._update_dict(user_id=user)
        # находим топ-N похожих пользователей
        try:
            similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
            similar_users = [rec_usr[0] for rec_usr in similar_users]
            similar_users = similar_users[1:]

            for usr in similar_users:
                result.extend(self.get_own_recommendations(self.id_to_userid[usr], N=1))

            result = self._extend_with_top_popular(result, N=N)
        except:
            result = self.overall_top_purch[:N]

        assert len(result) == N, 'Количество рекомендаций != {}'.format(N)
        return result
