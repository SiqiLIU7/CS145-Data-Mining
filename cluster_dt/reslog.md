feature: `user_favors.append(fav[20:])`
    DB SCAN

        DECISION TREE
            EPS=0.005, MIN_SAMPLES=2   random forest threshold=200  3600 clusters               0.6636
            EPS=0.003, MIN_SAMPLES=2   random forest threshold=200  5000 clusters 1811 noises   0.6628


    KMEANS

        DECISION TREE

            n_clusters=1000, random_state=0, n_jobs=-1, verbose=1
            run 5 iters
            inertia 17.376749735480626
            random forest threshold=200
            0.6664


            n_clusters=1000, random_state=0, n_jobs=-1, verbose=1
            random forest threshold=100
            0.66628


        SVM
            kernel='rbf'
            too slow

        
        `AdaBoostClassifier(DecisionTreeClassifier())`
            


DEBUG: 
change `user_tag_favor_dict[uid][tid-1] += 1 if rating == 1 else -1`
to `user_tag_favor_dict[uid][tid-1] += rel if rating == 1 else -rel`

todo: recluster user with all favor
        feature: `user_favors.append(fav)`  **cluster with both genre and tag info**

        KMEANS  n_clusters=1000, random_state=0, n_jobs=-1, verbose=1

            DECISION TREE and random forest
                0.68


            ada boost
                

                
                