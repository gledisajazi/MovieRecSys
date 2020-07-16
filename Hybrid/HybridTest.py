# -*- coding: utf-8 -*-
"""
Created on 2019

@author: Gledis
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator
from surprise import SVD, SVDpp
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


evaluator = Evaluator(evaluationData, rankings)


SimpleRBM = RBMAlgorithm(epochs=40)

ContentKNN = ContentKNNAlgorithm()
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")


Hybrid = HybridAlgorithm([SimpleRBM, ContentKNN, SVDPlusPlus], [0.3, 0.4, 0.3])


evaluator.AddAlgorithm(Hybrid, "Hybrid")


evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
