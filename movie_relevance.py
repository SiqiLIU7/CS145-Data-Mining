import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

# movie_tag_relevance_df = pd.read_csv('uclacs145fall2019/genome-scores.csv', dtype={'movideId': np.int32, 'tagId': np.int32, 'relevance': np.float32})
# movie_tag_relevance_df.head()
#
# movie_size = len(set(movie_tag_relevance_df['movieId'].values))
# tag_size = len(set(movie_tag_relevance_df['tagId'].values))
# assert movie_tag_relevance_df.shape[0] == movie_size * tag_size
# #
# # movie_id = sorted(list(set(movie_tag_relevance_df['movieId'].values)))
# # movie_id2idx = {}
# # for idx, id in enumerate(movie_id):
# #     movie_id2idx[id] = idx
# #
# # tag_id = sorted(list(set(movie_tag_relevance_df['tagId'].values)))
# # tag_id2idx = {}
# # for idx, id in enumerate(tag_id):
# #     tag_id2idx[id] = idx
# #
# # movie_fea = [[0] * (tag_size) for _ in range(movie_size)]
#
# movieId_max = max(movie_tag_relevance_df['movieId'].values)
# tagId_max = max(movie_tag_relevance_df['tagId'].values)
# movie2tag = np.zeros((movieId_max, tagId_max))
#
# for i in range(movie_tag_relevance_df.shape[0]):
#     try:
#         if i % 500000 == 0: print(i)
#         movie_id = movie_tag_relevance_df.iloc[i, 0] - 1
#         tag_id = movie_tag_relevance_df.iloc[i, 1] - 1
#         relevance = movie_tag_relevance_df.iloc[i, 2]
#         movie2tag[movie_id][tag_id] = relevance
#         #movie_fea[movie_id2idx[movie_id]][tag_id2idx[tag_id]] = relevance
#     except:
#         print('error at: ', i)
#
# np.save("movie2tag_maxtrix.npy", movie2tag)
matrix = np.load("movie2tag_maxtrix.npy")

pca = PCA(n_components=100)
pca.fit(matrix)
X_new = pca.transform(matrix)
print(X_new)

print("finish")
# movie_tag_relevance_df.iloc[1, 1]
# tmp = pd.DataFrame(movie_fea)
# tmp.shape
