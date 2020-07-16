# -*- coding: utf-8 -*-
"""
Created on 2019

@author: Gledis
"""

from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Duke u ngarkuar vleresimet e filmave...")
    data = ml.loadMovieLensLatestSmall()
    print("\nDuke llogaritur popullaritein qe te matim diversitetin...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)


(ml, evaluationData, rankings) = LoadMovieLensData()

print("Duke kerkuar per parametrat me te mire...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

# RMSE me e mire
print("RMSE: ", gs.best_score['rmse'])

# parametrat qe dhane RMSE me te mire
print(gs.best_params['rmse'])


evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = 20, lr_all = 0.005, n_factors = 50)
evaluator.AddAlgorithm(SVDtuned, "SVD e permiresuar")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD e papermiresuar")


#Random = NormalPredictor()
#evaluator.AddAlgorithm(Random, "Random")


evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
