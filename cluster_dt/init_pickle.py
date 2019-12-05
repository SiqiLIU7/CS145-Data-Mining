import logging
import os
import pickle
import math

import const

from sklearn.decomposition import PCA
import numpy as np


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
logger = logging.getLogger(__name__)


FAVOR_TOTAL_THRESHOLD = 3


def init_movie_info():
    """ save dict pickle, key: movie_id, val: list of genre ids
        save dict pickle, key: movie_id, val: movie year
    """
    logger.info('Initializing movie info')

    movie_genres_dict = {}
    movie_year_dict = {}

    genre_id = {}
    genre_id_cnt = 0
    with open(const.MOVIES_CSV, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            movie_id = int(line[0])

            # get genre info
            genres = line[-1].split('|')
            for g in genres:
                if g not in genre_id:
                    genre_id[g] = genre_id_cnt
                    genre_id_cnt += 1
            movie_genres_dict[movie_id] = [genre_id[g] for g in genres]

            # get year info
            try:
                year = int(line[-2].strip('"').strip()[-5:-1])
            except Exception:
                logger.error('failed to parse year from '+line[-2])
                year = 0
            movie_year_dict[movie_id] = year


    with open(const.MOVIE_GENRES_PICKLE, 'wb') as f:
        pickle.dump(movie_genres_dict, f)
    with open(const.MOVIE_YEAR_PICKLE, 'wb') as f:
        pickle.dump(movie_year_dict, f)
    
    logger.info('Saved %s, %s' % 
                (const.MOVIE_GENRES_PICKLE, 
                 const.MOVIE_YEAR_PICKLE))
    


def init_user_favor():
    """ save dict pickle, key: user_id, val: user favor vector
        key: user_id
        val: [genre1_favor, genre2_favor, ... genre20_favor]
              `favor` is (#like-#dislike}/(#like+#dislike)

        save dict pickle, key: user_id, val: {movie_id: rating} dict
    """
    logger.info('Initializing user favor')
    
    with open(const.MOVIE_GENRES_PICKLE, 'rb') as f:
        movie_genres_dict = pickle.load(f)
    with open(const.MOVIE_TAGS_PICKLE, 'rb') as f:
        movie_tags_dict = pickle.load(f)

    user_favor_dict = {}
    user_genre_favor_dict = {}
    user_genre_totoal_view_dict = {}
    user_tag_favor_dict = {}
    user_tag_totoal_view_dict = {}
    user_movie_rating_dict = {}
    # count user genere favor
    with open(const.TRAIN_CSV, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            rating = int(line[2])

            print('generating favor for uid %d' % uid)

            if uid not in user_favor_dict:
                user_favor_dict[uid] = []
                user_movie_rating_dict[uid] = {}
                user_genre_favor_dict[uid]      = [0]*(const.GENRE_NUM)
                user_genre_totoal_view_dict[uid]= [0]*(const.GENRE_NUM)
                user_tag_favor_dict[uid]        = [0]*(const.SYS_TAG_NUM)
                user_tag_totoal_view_dict[uid]  = [0]*(const.SYS_TAG_NUM)

            user_movie_rating_dict[uid][movie_id] = rating
            
            genres = movie_genres_dict.get(movie_id, [])
            tag_rel_list = movie_tags_dict.get(movie_id, [])
            
            for g in genres:
                user_genre_totoal_view_dict[uid][g] += 1
                user_genre_favor_dict[uid][g] += 1 if rating == 1 else -1
            for tid, rel in tag_rel_list:
                user_tag_totoal_view_dict[uid][tid-1] += rel
                user_tag_favor_dict[uid][tid-1] += rel if rating == 1 else -rel
    
    # normalize user favor
    for uid in user_favor_dict:
        print('normalizing user favor for uid %d' % uid)
        norm_user_favors = []
        # append genre favor
        for fav, total in zip(user_genre_favor_dict[uid], user_genre_totoal_view_dict[uid]):
            if total <= FAVOR_TOTAL_THRESHOLD:
                norm_user_favors.append(0)
            else:
                norm_user_favors.append(fav/total)
        # append tag favor
        for fav, total in zip(user_tag_favor_dict[uid], user_tag_totoal_view_dict[uid]):
            if total <= FAVOR_TOTAL_THRESHOLD:
                norm_user_favors.append(0)
            else:
                norm_user_favors.append(fav/total)
        user_favor_dict[uid] = norm_user_favors
    
    with open(const.USER_FAVOR_PICKLE, 'wb') as f:
        pickle.dump(user_favor_dict, f)
    logger.info('Saved %s' % const.USER_FAVOR_PICKLE)

    with open(const.USER_MOVIE_RATING_PICKLE, 'wb') as f:
        pickle.dump(user_movie_rating_dict, f)
    logger.info('Saved %s' % const.USER_MOVIE_RATING_PICKLE)


def init_test_info():
    logger.info('Initializing test data')
    test_dict = {'uids': [], 'movie_ids': []}
    with open(const.TEST_CSV, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            test_dict['uids'].append(uid)
            test_dict['movie_ids'].append(movie_id)
    with open(const.TEST_DATA_PICKLE, 'wb') as f:
        pickle.dump(test_dict, f)


def init_validation_info():
    logger.info('Initializing validation data')
    val_dict = {'uids': [], 'movie_ids': [], 'ratings': []}
    with open(const.VAL_CSV, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            rating = int(line[2])
            val_dict['uids'].append(uid)
            val_dict['movie_ids'].append(movie_id)
            val_dict['ratings'].append(rating)
    with open(const.VAL_DATA_PICKLE, 'wb') as f:
        pickle.dump(val_dict, f)


def init_sys_tag_info():
    logger.info('Initializing sys tag data')
    movie_tags_dict = {} #{mid: [(tid1, relevance), (tid2, r2), ...]}
    with open(const.TAG_RELEVANCE_CSV, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            mid = int(line[0])
            tid = int(line[1])
            rel = float(line[2])
            if mid not in movie_tags_dict:
                movie_tags_dict[mid] = []
            movie_tags_dict[mid].append((tid, rel))
    with open(const.MOVIE_TAGS_PICKLE, 'wb') as f:
        pickle.dump(movie_tags_dict, f)



def gen_movie_pca_feature(n_components=30):
    with open(const.MOVIE_GENRES_PICKLE, 'rb') as f:
        movie_genres_dict = pickle.load(f)
    with open(const.MOVIE_YEAR_PICKLE, 'rb') as f:
        movie_year_dict = pickle.load(f)
    with open(const.MOVIE_TAGS_PICKLE, 'rb') as f:
        movie_tags_dict = pickle.load(f)

    movie_feature_list = []
    movie_ids = list(range(1, const.MOVIE_NUM+1))
    for mid in movie_ids:
        # year as feature TODO
        
        # genre as 0/1 feature
        g_features = [0]*const.GENRE_NUM
        for g in movie_genres_dict.get(mid, []):
            g_features[g] = 1
        # tag as 0-1 feature
        t_features = [0]*const.SYS_TAG_NUM
        for tid, rel in movie_tags_dict.get(mid, []):
            t_features[tid-1] = rel
        
        movie_feature_list.append(g_features+t_features)

    logger.info('Doing PCA on movie feature')

    pca = PCA(n_components)
    new_features = pca.fit_transform(movie_feature_list)

    logger.info('PCA component variance: %s' % pca.explained_variance_ratio_)
    logger.info('PCA component variance sum: %s' % np.sum(pca.explained_variance_ratio_))

    movie_new_feature_dict = {}
    for mid, new_f in zip(movie_ids, new_features):
        movie_new_feature_dict[mid] = new_f

    with open(const.MOVIE_FEATURE_PICKLE, 'wb') as f:
        pickle.dump(movie_new_feature_dict, f)
    


if __name__ == '__main__':
    # init_sys_tag_info()
    # init_movie_info()
    init_user_favor()
    # init_test_info()
    # init_validation_info()
    # gen_movie_pca_feature()
    pass
