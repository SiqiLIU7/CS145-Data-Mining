import logging
import os
import random

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-10s %(levelname)-6s %(message)s')
logger = logging.getLogger(__name__)

DATASET_DIR = 'uclacs145fall2019'

popularity = {}

def cal_popularity(filename='train_ratings_binary.csv'):
    """ calculate the popularity of each movie,
        return dict, key: movie id, val: popularity (0-1)
    """
    logger.info('Calculating movie popularity')
    fp = os.path.join(DATASET_DIR, filename)
    viewed_dict = {}
    liked_dict = {}
    with open(fp, 'r') as f:
        # skip csv header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            uid = int(line[0])
            movie_id = int(float(line[1]))
            rating = int(line[2])
            if movie_id not in viewed_dict:
                viewed_dict[movie_id] = 0
                liked_dict[movie_id] = 0
            viewed_dict[movie_id] += 1
            if rating == 1: 
                liked_dict[movie_id] += 1
    
    for movie_id, viewed_cnt in viewed_dict.items():
        popularity[movie_id] = liked_dict[movie_id]/viewed_cnt

    logger.info('Calculated popularity for %d movies' % len(popularity))      


def test_val(filename='val_ratings_binary.csv'):
    fp = os.path.join(DATASET_DIR, filename)
    bingo = 0
    total = 0
    in_dict = 0
    with open(fp, 'r') as f:
        f.readline()
        for line in f:
            total += 1
            line = line.strip().split(',')
            movie_id = int(float(line[1]))
            guess = random.randint(0, 1)
            if movie_id in popularity:
                in_dict += 1
                if popularity[movie_id] > 0.5:
                    guess = 1
                elif popularity[movie_id] < 0.5:
                    guess = 0
            if guess == int(line[2]):
                bingo += 1
    
    print('total: %d' % total)
    print('indic: %d' % in_dict)
    print('bingo: %d' % bingo)
    print('score: %s' % (bingo/total))


def save_pred_res(testfile='test_ratings.csv', resfile='decision_tree_res.csv'):
    logger.info('Predicting ' + testfile)
    res = []
    test_fp = os.path.join(DATASET_DIR, testfile)
    with open(test_fp, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            movie_id = int(float(line[1]))
            guess = random.randint(0, 1)
            if movie_id in popularity:
                if popularity[movie_id] > 0.5:
                    guess = 1
                elif popularity[movie_id] < 0.5:
                    guess = 0
            res.append(guess)
    
    with open(resfile, 'w') as f:
        f.write('Id,rating\n')
        for id, pre in enumerate(res):
            f.write('%s,%d\n' % (id, pre))
    logger.info('Prediction result saved to '+ resfile)

    
                    


if __name__ == '__main__':
    cal_popularity()
    save_pred_res()
    