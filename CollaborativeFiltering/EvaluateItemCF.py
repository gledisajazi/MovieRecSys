# -*- coding: utf-8 -*-
"""
Created on 2016

@author: Gledis
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData

def LoadMovieLensData():
    ml = MovieLens()
    print("Duke u ngarkuar vleresimet e filmave...")
    data = ml.loadMovieLensLatestSmall()
    print("\nDuke llogaritur popullaritein qe te matim diversitetin...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

ml, data, rankings = LoadMovieLensData()

evalData = EvaluationData(data, rankings)

# Trajnim me setin leave-One-Out
trainSet = evalData.GetLOOCVTrainSet()
sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

leftOutTestSet = evalData.GetLOOCVTestSet()

# ndertojme lista dictionary per ciftet (int(movieID), predictedrating)
topN = defaultdict(list)
k = 10
for uiid in range(trainSet.n_users):
    userRatings = trainSet.ur[uiid]
   
    kNeighbors = heapq.nlargest(k, similarRatings, key=lambda t: t[1])
    
    candidates = defaultdict(float)
   for itemID, rating in kNeighbors:
       similarityRow = simsMatrix[itemID]
       for innerID, score in enumerate(similarityRow):
           candidates[innerID] += score * (rating / 5.0)
     
    # dictionary me filmat qe perdoruesi i ka pare
    watched = {}
    for itemID, rating in trainSet.ur[uiid]:
        watched[itemID] = 1
        
    # marrim filmat me te vleresuar per perdoruesit e ngjashem
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            topN[int(trainSet.to_raw_uid(uiid))].append( (int(movieID), 0.0) )
            pos += 1
            if (pos > 40):
                break
    

print("HR", RecommenderMetrics.HitRate(topN, leftOutTestSet))   


