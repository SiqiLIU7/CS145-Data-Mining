import logging
import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from util import movie_genres_dict, movie_year_dict, \
                    user_favor_dict, user_movie_rating_dict
from util import movie_feature_dict, test_dict, val_dict

import const



logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
logger = logging.getLogger(__name__)


with open(const.USER_CLUSTER_PICKLE, 'rb') as f:
        user_cluster_dict = pickle.load(f)

def get_cluster_feature_labels():
    logger.info('extracting cluster features and labels')
    cluster_features = {}
    cluster_labels = {}
    for uid, movie_rating in user_movie_rating_dict.items():
        cid = user_cluster_dict[uid]
        if cid not in cluster_features:
            cluster_features[cid] = []
            cluster_labels[cid] = []
        for mid, rating in movie_rating.items():
            movie_f = movie_feature_dict[mid]
            cluster_features[cid].append(movie_f)
            cluster_labels[cid].append(rating)
    
    return cluster_features, cluster_labels


def train_cluster_clfs(cluster_features, cluster_labels):
    cluster_clf_dict = {}
    cluster_num = len(cluster_features)
    clf_cnt = 0
    for cid, features in cluster_features.items():
        clf_cnt += 1
        logger.info('Training %d cluster clf, total %d' % (clf_cnt, cluster_num))
        labels = cluster_labels[cid]

        """uncomment to use decision tree"""
        # if len(labels) > 100:
        #     clf = RandomForestClassifier(criterion='entropy')
        # else:
        #     clf = tree.DecisionTreeClassifier(criterion='entropy')

        """uncomment to use svm"""
        # clf = svm.SVC(kernel='rbf')

        """uncomment to use adaBoost"""
        clf = AdaBoostClassifier(DecisionTreeClassifier())
    
        clf = clf.fit(features, labels)
        cluster_clf_dict[cid] = clf
        
    logger.info('Finished training %d clfs' % cluster_num)
    with open(const.CLUSTER_CLF_PICKLE, 'wb') as f:
        pickle.dump(cluster_clf_dict, f)
    logger.info('Saved '+const.CLUSTER_CLF_PICKLE)


def get_val_score():
    res_list = predict_uid_mids(val_dict['uids'], val_dict['movie_ids'])
    bingo = 0
    for p, t in zip(res_list, val_dict['ratings']):
        if p == t:
            bingo += 1
    return bingo/len(res_list)
        

def save_test_csv(filename='result.csv'):
    res_list = predict_uid_mids(test_dict['uids'], test_dict['movie_ids'])
    with open(filename, 'w') as f:
        f.write('Id,rating\n')
        for id, pre in enumerate(res_list):
            f.write('%s,%d\n' % (id, pre))
    logger.info('Prediction result saved to '+ filename)


def predict_uid_mids(uids, movie_ids):
    with open(const.CLUSTER_CLF_PICKLE, 'rb') as f:
        cluster_clf_dict = pickle.load(f)
    res = {}
    test_id = 0
    cluster_features = {}
    cluster_test_ids = {}
    for uid, mid in zip(uids, movie_ids):
        test_id += 1
        print('test id #%d' % test_id)
        cid = user_cluster_dict[uid]
        if cid not in cluster_features:
            cluster_features[cid] = []
            cluster_test_ids[cid] = []
        cluster_features[cid].append(movie_feature_dict[mid])
        cluster_test_ids[cid].append(test_id)

    for cid, features in cluster_features.items():
        clf = cluster_clf_dict[cid]
        preds = clf.predict(features)
        for tid, p in zip(cluster_test_ids[cid], preds):
            res[tid] = p
    
    res_list = []
    for id in range(1, test_id+1):
        res_list.append(res[id])
    return res_list


# make sure init_pickle and cluster_user have been executed
if __name__ == '__main__':
    cluster_features, cluster_labels = get_cluster_feature_labels()
    train_cluster_clfs(cluster_features, cluster_labels)

    print(get_val_score())

    # save_test_csv()