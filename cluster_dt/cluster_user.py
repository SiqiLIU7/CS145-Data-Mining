"""Do PCA on user favor, cluster users"""

import logging
import pickle

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

from util import movie_genres_dict, movie_year_dict, user_favor_dict
import const

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
logger = logging.getLogger(__name__)


def get_user_favors_after_pca(range_ctr_debug=10000000, n_components=8):
    """ get transformed user favors
        return uids, transformed_user_favors
    """
    

    uids = []
    user_favors = []

    for uid, fav in user_favor_dict.items():
        range_ctr_debug -= 1
        if range_ctr_debug == 0:
            break

        uids.append(uid)
        user_favors.append(fav)

    logger.info('Doing PCA on %d * %d user favors' % (len(user_favors), len(user_favors[0])))

    pca = PCA(n_components)
    new_favors = pca.fit_transform(user_favors)

    logger.info('PCA component variance: %s' % pca.explained_variance_ratio_)
    logger.info('PCA component variance sum: %s' % np.sum(pca.explained_variance_ratio_))

    with open('tmp_pca_uids.pickle', 'wb') as f:
        pickle.dump(uids, f)
    
    with open('tmp_pca_favors.pickle', 'wb') as f:
        pickle.dump(new_favors, f)

    return uids, new_favors
    
    


def cluster_user(uids, user_favors, EPS=0.5, MIN_SAMPLES=2):
    """cluster users and save pickle"""
    logger.info('clustering users by user favor')

    # clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(user_favors)
    clustering = KMeans(n_clusters=1000, random_state=0, n_jobs=-1, verbose=1).fit(user_favors)
    logger.info('run %d iters' % clustering.n_iter_)
    logger.info('inertia %s' % clustering.inertia_)

    user_cluster_ids = clustering.labels_
    user_cluster_dict = {}
    for uid, cid in zip(uids, user_cluster_ids):
        user_cluster_dict[uid] = cid
    
    # count cluster
    noise_cnt = 0
    cluster_size_dict = {}
    for uid in uids:
        if user_cluster_dict[uid] == -1:
            noise_cnt += 1
        else:
            if user_cluster_dict[uid] not in cluster_size_dict:
                cluster_size_dict[user_cluster_dict[uid]] = 0
            cluster_size_dict[user_cluster_dict[uid]] += 1
    logger.info('found %s clusters %s' % (len(cluster_size_dict), cluster_size_dict))
    logger.info('found %d noises' % (noise_cnt))

    with open(const.USER_CLUSTER_PICKLE, 'wb') as f:
        pickle.dump(user_cluster_dict, f)

    logger.info('user_cluster_dict saved to %s' % const.USER_CLUSTER_PICKLE)



if __name__ == '__main__':
    # print(len(user_favor_dict))
    # print(user_favor_dict[2])

    """set range_ctr_debug to fast test"""
    # uids, new_user_favors = get_user_favors_after_pca(range_ctr_debug=1000000)


    """uncomment to cluster users"""
    with open('tmp_pca_uids.pickle', 'rb') as f:
        uids = pickle.load(f)
    with open('tmp_pca_favors.pickle', 'rb') as f:
        new_user_favors = pickle.load(f)
    cluster_user(uids, new_user_favors)


    ## only for debug
    # with open(const.USER_CLUSTER_PICKLE, 'rb') as f:
    #     user_cluster_dict = pickle.load(f)

    # print(user_cluster_dict)
