import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        
        # your_code
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        
        id_to_itemid, id_to_userid, itemid_to_id, userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        res = []
        
        topN_bought = self.user_item_matrix.loc[[user]].T.drop([999999]).sort_values(by=user, ascending=False).\
        head(N).index.tolist()
        
        for top in topN_bought:
            closest_items = [ id_to_itemid[row_id] for row_id, score in self.model.similar_items(itemid_to_id[top], N=2)]
            res.append(closest_items[1])
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        id_to_itemid, id_to_userid, itemid_to_id, userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        res = []
        
        similar_users = self.model.similar_users(userid_to_id[user], N=N+1)
        similar_users = [id_to_userid[row_id] for row_id, score in similar_users[1:]]
        
        for sim_user in similar_users:
            res.append(self.user_item_matrix.loc[[sim_user]].T.drop([999999]).sort_values(by=sim_user, ascending=False).\
                       head(1).index.item())
        

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res