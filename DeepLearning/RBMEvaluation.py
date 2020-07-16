# -*- coding: utf-8 -*-
"""
Created on 2019

@author: Gledis
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator

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


RBM = RBMAlgorithm(epochs=20)
evaluator.AddAlgorithm(RBM, "RBM")


#Random = NormalPredictor()
#evaluator.AddAlgorithm(Random, "Random")


evaluator.Evaluate(True)
evaluator.SampleTopNRecs(ml)
