import logging
import os
import pickle
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import get_tag_weight
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn import preprocessing
import graphviz
from sklearn.tree.export import export_text

import csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
# filename="./log.txt",
# filemode='a')

logger = logging.getLogger(__name__)

DATASET_DIR = 'uclacs145fall2019'

movie_genres = {}
genres_id_dict = {}


def init_genres_info(filename='movies.csv'):
    """ initialize {movie_id: genre_id_list} dict
        and {genre_name: genre_id} dict"""
    fp = os.path.join(DATASET_DIR, filename)
    logger.info('Initializing info from ' + fp)
    genre_id_cnt = 0
    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            movie_id = int(line[0])
            genres = line[-1].split('|')
            for g in genres:
                if g not in genres_id_dict:
                    genres_id_dict[g] = genre_id_cnt
                    genre_id_cnt += 1
            movie_genres[movie_id] = [genres_id_dict[g] for g in genres]

    logger.info('Finished %s initialization' % fp)


def get_training_set(filename='train_ratings_binary.csv'):
    """ return x, y
        x: feature vectore list
            [[uid, binary_genre1, binary_genre2, ...], ...]
        y: label list
            [rating1, rating2, ...]
    """
    fp = os.path.join(DATASET_DIR, filename)
    logger.info('Extracting training vectors from ' + fp)

    genre_cnt = len(genres_id_dict)
    features = []
    labels = []

    movie2tag = np.load("movie2tag_matrix.npy")
    # pca = PCA(n_components=100)
    # pca.fit(movie2tag)
    # movie2tag_PCA = pca.transform(movie2tag)

    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            rating = int(line[2])

            # genre_vec = [0]*genre_cnt
            # for g in movie_genres[movie_id]:
            #     genre_vec[g - 1] = 1

            # features.append(genre_vec + list(movie2tag_PCA[movie_id]))
            features.append(list(movie2tag[movie_id]))
            labels.append(rating)

    np.save("features_noGenre_tag1000.npy", np.array(features))
    np.save("labels.npy", np.array(labels))

    # features = np.load("features.npy")
    # labels = np.load("labels.npy")

    logger.info('Finished training vector extraction')
    return features


def get_testing_set(filename='test_ratings.csv'):
    """ return x
        x: feature vectore list
            [[binary_genre1, binary_genre2, ...], ...]
    """
    fp = os.path.join(DATASET_DIR, filename)
    logger.info('Extracting testing vectors from ' + fp)

    genre_cnt = len(genres_id_dict)
    features = []

    movie2tag = np.load("movie2tag_matrix.npy")
    logger.info('Finish loading movie2tag_matrix')
    pca = PCA(n_components=20)
    pca.fit(movie2tag)
    movie2tag_PCA = pca.transform(movie2tag)

    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))

            genre_vec = [0] * genre_cnt
            for g in movie_genres[movie_id]:
                genre_vec[g] = 1
            if movie_id >= len(movie2tag_PCA):
                features.append(genre_vec + [0] * 20)
            else:
                features.append(genre_vec + list(movie2tag_PCA[movie_id]))

    # np.save("test_features.npy", np.array(features))

    # features = np.load("features.npy")
    # labels = np.load("labels.npy")

    logger.info('Finished testing vector extraction')
    return features


# def train_SVM(features, labels, model_save_path):
#     logger.info('Training SVM classifier')
#     clf = svm.SVC(kernel='rbf')
#     clf.fit(features, labels)
#     logger.info('Finished training')
#     with open(model_save_path,'wb') as f:
#         pickle.dump(clf, f)
#     logger.info('SVM classifier save at '+model_save_path)

def train_DecisionTree(features, labels, model_save_path):
    logger.info('Training Decision Tree classifier')
    clf = tree.DecisionTreeClassifier()
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 6, 2), random_state=1)
    # scores = cross_val_score(clf, features, labels, cv=5)
    clf.fit(features, labels)

    # Visualizing tree
    r = export_text(clf)
    print("visualizing: ", r)

    logger.info('Finished training')
    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)
    logger.info('Decision Tree classifier save at ' + model_save_path)


def validation(features_val, labels_val, model_save_path):
    logger.info('Begin validation')
    with open(model_save_path, 'rb') as f:
        clf = pickle.load(f)

    score_val = clf.score(features_val, labels_val)
    logger.info('Finish validation')
    logger.info('Mean accuracy of SVM classifier on validation set: ' + str(score_val))


def test(X_test, model_save_path):
    logger.info('Begin testing')
    file = open(model_save_path, 'rb')
    clf = pickle.load(file)
    output = clf.predict(X_test)
    with open("submission.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "rating"])
        for id, y in enumerate(output):
            writer.writerow([id, y])

    logger.info('Finish testing')


if __name__ == '__main__':
    model_path = 'decision_tree_v.model'
    init_genres_info()
    features, labels = get_training_set()

    features, labels = shuffle(features, labels, random_state=0)
    # sel = VarianceThreshold(threshold=.2)
    # sel.fit_transform(features)
    # features = features + 4
    # features_new = SelectKBest(chi2, k=20).fit_transform(features, labels)
    # features_std = preprocessing.scale(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

    train_DecisionTree(X_train, y_train, model_path)
    features_val, labels_val = get_training_set('val_ratings_binary.csv')
    # validation(X_test, y_test, model_path)
    validation(features_val, labels_val, model_path)

    features_test, _ = get_training_set("test_ratings.csv")
    # features_std = preprocessing.scale(features_test)
    test(features_test, model_path)


