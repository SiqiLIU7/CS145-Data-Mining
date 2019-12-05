import logging
import os
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn import tree

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
logger = logging.getLogger(__name__)

DATASET_DIR = 'uclacs145fall2019'
MAX_MOVIE_GENRE_CNT = 10

popularity = {}
user_genre_popularity = {}
movie_genres = {}


def init_movie2genres(filename='movies.csv'):
    fp = os.path.join(DATASET_DIR, filename)
    logger.info('Initializing movie_id-genre_list dict from %s' % fp)
    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            movie_id = int(line[0])
            genres = line[-1].split('|')
            movie_genres[movie_id] = genres    
    logger.info('Finished initialization from %s' % fp)


def cal_popularity(filename='train_ratings_binary.csv'):
    """ calculate the popularity of each movie among all users,
        calculate each user's preference for each genre
    """
    logger.info('Calculating movie popularity')
    fp = os.path.join(DATASET_DIR, filename)
    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            rating = int(line[2])
            # popularity among all users
            if movie_id not in popularity:
                popularity[movie_id] = 0
            popularity[movie_id] += 1 if rating == 1 else -1
            # each user's preference
            if uid not in user_genre_popularity:
                user_genre_popularity[uid] = {}
            genres = movie_genres.get(movie_id, [])
            for g in genres:
                if g not in user_genre_popularity[uid]:
                    user_genre_popularity[uid][g] = 0
                user_genre_popularity[uid][g] += 1 if rating == 1 else -1
            
    logger.info('Calculated popularity for %d movies and %d users' % 
                    (len(popularity), len(user_genre_popularity)))   


def get_user_movie_feature(uid, movie_id):
    """ return [movie_populartity, user_genre_preference1, user_genre_preference2, ...]
    """
    movie_popularity = popularity.get(movie_id, 0)
    user_popularity = []
    movie_gs = movie_genres.get(movie_id, [])
    for movie_g in movie_gs:
        gp = user_genre_popularity.get(uid, {})
        user_popularity.append(gp.get(movie_g, 0))
    # padding zero
    user_popularity = user_popularity + [0]*(MAX_MOVIE_GENRE_CNT-len(user_popularity))
    # sort by abs
    user_popularity = sorted(user_popularity, key=abs, reverse=True)
    return [movie_popularity] + user_popularity


def get_training_set(filename='train_ratings_binary.csv'):
    fp = os.path.join(DATASET_DIR, filename)
    logger.info('Extracting training vectors from '+fp)

    features = []
    labels = []

    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            rating = int(line[2])
            f = get_user_movie_feature(uid, movie_id)
            features.append(get_user_movie_feature(uid, movie_id))
            labels.append(rating)

    logger.info('Finished training vector extraction')
    return features, labels
    


def train_decision_tree(model_save, features, labels):
    logger.info('Training decision tree classifier')
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(features, labels)
    logger.info('Finished training')
    with open(model_save,'wb') as f:
        pickle.dump(clf, f)
    logger.info('Classifier save at '+model_save)
    


def save_pred_res(clf, resfile, testfile='test_ratings.csv'):
    test_fp = os.path.join(DATASET_DIR, testfile)
    logger.info('Predicting ' + test_fp)
    features = []
    with open(test_fp, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            features.append(get_user_movie_feature(uid, movie_id))
            
    res = clf.predict(features)
    with open(resfile, 'w') as f:
        f.write('Id,rating\n')
        for id, pre in enumerate(res):
            f.write('%s,%d\n' % (id, pre))
    logger.info('Prediction result saved to '+ resfile)

    

if __name__ == '__main__':
    model_save = 'dt.model'

    """ uncomment to train classifier """
    init_movie2genres()
    cal_popularity()
    train_features, train_labels = get_training_set()
    print('training set size: %d ' % len(train_features))
    train_decision_tree(model_save, train_features, train_labels)


    """ uncomment to predict """
    with open(model_save, 'rb') as f:
        clf = pickle.load(f)
    save_pred_res(clf, 'dt_result.csv')


    """ uncomment to val_test """
    val_features, val_labels = get_training_set('val_ratings_binary.csv')
    print(clf.score(val_features, val_labels))



    """ tmp cross validation """
    # model_tmp_save = 'tmp_dt_crossval.model'
    # init_movie2genres()
    # cal_popularity()
    # train_features, train_labels = get_training_set()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     train_features, train_labels, test_size=0.1, random_state=0)
    # train_decision_tree(model_tmp_save, train_features, train_labels)
    # with open(model_tmp_save, 'rb') as f:
    #     clf = pickle.load(f)
    # print(clf.score(X_test, y_test))

    