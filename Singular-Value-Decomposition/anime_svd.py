import pandas as pd
import numpy as np

'''
Makes the order of ratings make more sense. 
Haven't seen is 0, and negative ratings are now numerically negative.
'''
def adjust_matrix(np_matrix):
    np_matrix[np_matrix == -1] = 0
    np_matrix[np_matrix == 1] = -5
    np_matrix[np_matrix == 2] = -4
    np_matrix[np_matrix == 3] = -3
    np_matrix[np_matrix == 4] = -2
    np_matrix[np_matrix == 5] = -1
    np_matrix[np_matrix == 6] = 1
    np_matrix[np_matrix == 7] = 2
    np_matrix[np_matrix == 8] = 3
    np_matrix[np_matrix == 9] = 4
    np_matrix[np_matrix == 10] = 5
    return np_matrix

'''
Constructs a matrix that maps user_ids to their ratings of different animes
'''
def make_matrix(ratings, animes):
    ratings_sample = ratings.iloc[:len(ratings)//70]
    np_matrix = ratings_sample.pivot_table(index='user_id', columns='anime_id', values='rating', aggfunc = "mean")#.to_numpy() #make_data(ratings,animes)
    
    #get mapping column index --> anime_id
    anime_ids = list(np_matrix)

    #finally turn np_matrix into an NP matrix
    np_matrix = np_matrix.to_numpy()
    #turn all the non-initialize ratings into negative ones
    np_matrix[np.isnan(np_matrix)] = -1

    return np_matrix, anime_ids

'''
gets recommendations for a user user_num from file names
'''
def getRecs(user_num, ratings_file, anime_file):
    #get all data
    ratings = pd.read_csv(ratings_file)
    animes = pd.read_csv(anime_file)

    #make matrix
    np_matrix, anime_ids = aSVD.make_matrix(ratings, animes) #in the np array, users ind = user -1, anime_id = anime_ids[anime_indice]

    #conduct SVD
    U,S,Vt = np.linalg.svd(np_matrix,full_matrices=False)

    #find which types that user likes
    user_preferences = np.argsort(U[user_num - 1, :])[-3:] #get the max 3 types

    reccomendations = []
    #find which animes they like that fit in that type
    for pref_ind in user_preferences:
        #sort by the best in the category
        sorted_recs = np.argsort(-Vt[pref_ind, :])
    
        #check which animes they haven't seen yet
        for sorted_rec in sorted_recs:
            if np_matrix[user_num - 1, sorted_rec] == -1:
                top_anime_id = anime_ids[sorted_rec]
                anime_ind = animes.index[animes["anime_id"] == top_anime_id].tolist()[0]
                reccomendations.append(animes[["name"]].loc[anime_ind])
                break

    #return three of these movies
    return reccomendations


