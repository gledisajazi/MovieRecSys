from MovieLens import MovieLens
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics

ml = MovieLens()

print("Duke ngarkuar vlerësimet e filmave...")
data = ml.loadMovieLensLatestSmall()

print("\nDuke llogaritur popullaritetin për të gjetur risinë më vonë...")
rankings = ml.getPopularityRanks()

print("\nDuke llogaritur ngjashmërinë midis filmave në mënyrë që të llogaritim diversitetin ...")
fullTrainSet = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)

print("\nNdertojmë modelin e rekomandimit...")
trainSet, testSet = train_test_split(data, test_size=.25, random_state=1)

algo = SVD(random_state=10)
algo.fit(trainSet)

print("\nDuke llogaritur rekomandimet...")
predictions = algo.test(testSet)

print("\nDuke llogrritur saktësinë e modelit...")
print("RMSE: ", RecommenderMetrics.RMSE(predictions))
print("MAE: ", RecommenderMetrics.MAE(predictions))

print("\Llogaritim rekomandimet Top-10...")

# Leme jashte nje vleresim per cdo user per ta perdorur si testim
LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for trainSet, testSet in LOOCV.split(data):
    print("Llogaritim rekomandimet me leave-one-out...")

    # Trajnohet modeli
    algo.fit(trainSet)

    # Parashikohen vleresimet vetem per vleresimin e lene jashte
    print("Llogaritimin vlerësimet për setin left-out...")
    leftOutPredictions = algo.test(testSet)

    # Parashikime per te gjithe vleresimet qe nuk jane ne setin e trajnimit
    print("Parashikojmë të gjitha vlerësimet që mungojnë...")
    bigTestSet = trainSet.build_anti_testset()
    allPredictions = algo.test(bigTestSet)

   
    print("Llogaritim 10 rekomandimet krysore për përdorues...")
    topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n=10)

    
    print("\nHit Rate: ", RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions))

   
    print("\nrHR (Hit Rate by Rating value): ")
    RecommenderMetrics.RatingHitRate(topNPredicted, leftOutPredictions)

 
    print("\ncHR (Cumulative Hit Rate, vlerësimi >= 4): ", RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions, 4.0))

    print("\nARHR (Average Reciprocal Hit Rank): ", RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions))

print("\nLlogaritim të gjitha rekomandimet,gjithë setin...")
algo.fit(fullTrainSet)
bigTestSet = fullTrainSet.build_anti_testset()
allPredictions = algo.test(bigTestSet)
topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n=10)


print("\nMbulimi: ", RecommenderMetrics.UserCoverage(topNPredicted, fullTrainSet.n_users, ratingThreshold=4.0))

print("\nDiversiteti: ", RecommenderMetrics.Diversity(topNPredicted, simsAlgo))

print("\nNovelty (risia): ", RecommenderMetrics.Novelty(topNPredicted, rankings))
