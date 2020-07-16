from MovieLens import MovieLens
from surprise import SVD


def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []
    
    u = trainset.to_inner_uid(str(testSubject))
    
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset

# Perdorues arbitrat per testim (u zgjodh per arsye ngjashmerie me shijet e mia per filmat)
testSubject = 85

ml = MovieLens()

print("Duke ngarkuar vlerësimet e filmave...")
data = ml.loadMovieLensLatestSmall()

userRatings = ml.getUserRatings(testSubject)
loved = []
hated = []
for ratings in userRatings:
    if (float(ratings[1]) > 4.0):
        loved.append(ratings)
    if (float(ratings[1]) < 3.0):
        hated.append(ratings)

print("\nPërdoruesi ", testSubject, " pëlqen këta filma:")
for ratings in loved:
    print(ml.getMovieName(ratings[0]))
print("\n...dhe nuk pëlqen këta filma:")
for ratings in hated:
    print(ml.getMovieName(ratings[0]))

print("\nDuke ndërtuar modelin e rekomandimit...")
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

print("Duke llogaritur rekomandimet...")
testSet = BuildAntiTestSetForUser(testSubject, trainSet)
predictions = algo.test(testSet)

recommendations = []

print ("\nRekomandohen:")
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

recommendations.sort(key=lambda x: x[1], reverse=True)

for ratings in recommendations[:10]:
    print(ml.getMovieName(ratings[0]))


