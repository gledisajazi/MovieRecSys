# -*- coding: utf-8 -*-
"""
Created on 2019

@author: Gledis
"""
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Llogatimi saktësinë
        if (verbose):
            print("Duke llogaritur saktësinë...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)
        
        if (doTopN):
        
            if (verbose):
                print("Duke llogaritur top-N me leave-one-out...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())        

            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())
            # llogaritimin top-10 rekomadimet per cdo user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Duke llogaritur hit-rate dhe metrikat e renditjes...")

            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   

            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)

            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
        

            if (verbose):
                print("Duke llogaritur rekomandime me të gjithë setin e të dhënave...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Duke analizuar mbulimin, diversitetin dhe risinë ...")

            metrics["Mbulimi"] = RecommenderMetrics.UserCoverage(  topNPredicted, 
                                                                   evaluationData.GetFullTrainSet().n_users, 
                                                                   ratingThreshold=4.0)

            metrics["Diversiteti"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.GetSimilarities())
            

            metrics["Risia"] = RecommenderMetrics.Novelty(topNPredicted, 
                                                            evaluationData.GetPopularityRankings())
        
        if (verbose):
            print("Analiza u përfundua.")
    
        return metrics
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
    
    