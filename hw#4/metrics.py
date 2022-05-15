'''
Metrics
'''
import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
#   сделать в домашней работе
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    
    recommend_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    
    flags = np.isin(recommend_list, bought_list)
    
    precision = np.dot(flags, prices_recommended).sum() / prices_recommended.sum()
    return precision

def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)
    

def recall_at_k(recommended_list, bought_list, k=5):
    #    сделать дома
    return recall(recommended_list[:k], bought_list)

def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    # сделать дома
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    
    flags = np.isin(bought_list, recommended_list)
    
    prices_relevant = np.dot(flags, np.array(prices_recommended))
    prices_bought = np.array(prices_bought)
    return prices_relevant.sum() / prices_bought.sum()

def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    
    amount_relevant = len(relevant_indexes)
            
    sum_ = sum([precision_at_k(recommended_list, bought_list, k=index_relevant+1) for index_relevant in relevant_indexes])
    return sum_/amount_relevant

def map_k(recommended_list, bought_list, k=5):
    # сделать дома
    map_k = list()
    
    recommended_list = np.array(recommended_list, dtype=object)
    bought_list = np.array(bought_list, dtype=object)
    
    for i in range(recommended_list.shape[0]):
        map_k.append(ap_k(recommended_list[i], bought_list[i], k=k))
    
    map_k = np.array(map_k)
    result = map_k.mean()
    
    return result

def reciprocal_rank(recommended_list, bought_list, k=1):
    # сделать дома
    rrank = list()
    
    recommended_list = np.array(recommended_list, dtype=object)
    bought_list = np.array(bought_list, dtype=object)
    
    for i in range(recommended_list.shape[0]):
        flags = np.isin(bought_list[i], recommended_list[i, :k])
        
        if len(np.flatnonzero(flags)) == 0:
            rrank.append(0)
        else:
            ku = np.flatnonzero(flags)[0] + 1
            rrank.append(1/ku)
    
    rrank = np.array(rrank)
    return rrank.mean()