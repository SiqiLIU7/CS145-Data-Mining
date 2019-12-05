import logging
import pickle

import const

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
logger = logging.getLogger(__name__)

with open(const.MOVIE_GENRES_PICKLE, 'rb') as f:
    movie_genres_dict = pickle.load(f)

with open(const.MOVIE_YEAR_PICKLE, 'rb') as f:
    movie_year_dict = pickle.load(f)

with open(const.USER_FAVOR_PICKLE, 'rb') as f:
    user_favor_dict = pickle.load(f)

with open(const.USER_MOVIE_RATING_PICKLE, 'rb') as f:
    user_movie_rating_dict = pickle.load(f)

with open(const.MOVIE_FEATURE_PICKLE, 'rb') as f:
    movie_feature_dict = pickle.load(f)

with open(const.TEST_DATA_PICKLE, 'rb') as f:
    test_dict = pickle.load(f)

with open(const.VAL_DATA_PICKLE, 'rb') as f:
    val_dict = pickle.load(f)

logger.info('loaded pickle files')



# for debugging
if __name__ == '__main__':
    print('movie_genres_dict[1]:')
    print(movie_genres_dict[1])
    print('movie_year_dict[1]:')
    print(movie_year_dict[1])
    
    print('user_favor_dict:')
    for i in range(1, 10):
        print(user_favor_dict[i])
    
    print('movie_feature_dict')
    for i in range(1, 10):
        print(movie_feature_dict[i])
    