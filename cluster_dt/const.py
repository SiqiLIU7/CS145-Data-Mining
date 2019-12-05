import os

GENRE_NUM = 20
USER_NUM = 138493
MOVIE_NUM = 131262
SYS_TAG_NUM = 1128


DATASET_DIR = '../uclacs145fall2019'
MOVIES_CSV = os.path.join(DATASET_DIR, 'movies.csv')
TRAIN_CSV = os.path.join(DATASET_DIR, 'train_ratings_binary.csv')
TEST_CSV = os.path.join(DATASET_DIR, 'test_ratings.csv')
VAL_CSV = os.path.join(DATASET_DIR, 'val_ratings_binary.csv')


TAG_RELEVANCE_CSV = os.path.join(DATASET_DIR, 'genome-scores.csv')


MOVIE_GENRES_PICKLE = 'movie_genres.pickle'
MOVIE_YEAR_PICKLE = 'movie_year.pickle'
MOVIE_FEATURE_PICKLE = 'movie_feature.pickle'

USER_FAVOR_PICKLE = 'user_genre_favor.pickle'
USER_MOVIE_RATING_PICKLE = 'user_movie_rating.pickle'  # from training data

USER_CLUSTER_PICKLE = 'user_cluster.pickle'

CLUSTER_CLF_PICKLE = 'cluster_clf.pickle'

MOVIE_TAGS_PICKLE = 'movie_sys_tags.pickle' #{mid: [(tid1, relevance), (tid2, r2), ...]}


TEST_DATA_PICKLE = 'test_data.pickle' # {'uids': [...], 'movie_ids': [...]}
VAL_DATA_PICKLE = 'val_data.pickle' # {'uids': [...], 'movie_ids': [...], 'ratings': [...]}
