# -*- coding: utf-8 -*-
"""
Created on 2019

@author: Gledis
"""
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

       
        print("\n")
        
        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algoritmi", "RMSE", "MAE", "HR", "cHR", "ARHR", "Mbulimi", "Diversiteti", "Risia"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Mbulimi"], metrics["Diversiteti"], metrics["Risia"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        print("\nLegjenda:\n")
        print("RMSE:      Root Mean Squared Error. Vlerat më të ulëta nënkuptojnë saktësi më të lartë .")
        print("MAE:       Mean Absolute Error. Vlerat më të ulëta nënkuptojnë saktësi më të lartë .")
        if (doTopN):
            print("HR:        Hit Rate. Sa shpesh mund të rekomandojmë filmin e lënë jashtë setit të trajnimit. Sa më i lartë aq më i mirë.")
            print("cHR:       Cumulative Hit Rate; hit rate, por për vlerësimet sipër një pragu të caktuar. Sa më i lartë aq më i mirë.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate që merr parasysh rendin e filmit në listën top-N. Sa më i lartë aq më i mirë." )
            print("Mbulimi:   Përqindja e përdoruesve për të cilët ekziston një rekomandim sipër një pragu të caktuar. Sa më i lartë aq më i mirë.")
            print("Diversiteti: 1-S, ku S është mesatarja e ngjashmërisë midis çdo çifti të kombinuar për një përdorues të dhënë.")
            print("           Numri i lartë tregon diversitet më të madh.")
            print("Risia:     Mesatarja e popullaritetit të filmave të rekomanduar. Numri i lartë tregon risi më të lartë.")
        
    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        
        for algo in self.algorithms:
            print("\nDuke përdorur rekomanduesin ", algo.GetName())
            
            print("\nDuke ndërtuar modelin e rekomandimit...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            print("Duke llogaritur rekomandimet...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        
            predictions = algo.GetAlgorithm().test(testSet)
            
            recommendations = []
            
            print ("\nRekomandohet:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
                

            
            
    
    