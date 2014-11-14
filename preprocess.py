from json import loads
import numpy
import scipy
import sklearn
import datetime
#not sure i did these imports right yet, lets see if i get a chance to test

businesses_filename = 'data/yelp_academic_dataset_business.json'
checkin_filename    = 'data/yelp_academic_dataset_checkin.json'
review_filename     = 'data/yelp_academic_dataset_review.json'
tip_filename        = 'data/yelp_academic_dataset_tip.json'
user_filename       = 'data/yelp_academic_dataset_user.json'

testReviews = [] #a list of feature lists
trainingReviews = [] #a list of feature lists
testReviewActual = [] #list of integers
trainingReviewActual = [] #list of integers
users = {} #dict of userid:user object

#im starting with lists because they are what I know, 
#I figure later we can go through and convert each one into a numpy array

#get all the users
with open(user_filename, 'r') as user_file:
	for line in user_file:
		user = (loads(line))
		users[user['user_id']] = user

#right now just gets the features in the review, not the text
def extractReviewFeatures(review):
	features = []
	features.append(review['stars'])
	features.append(len(review['text']))
	userFeatures = getUserFeatures(review['user_id'])
	if userFeatures:
		for feature in userFeatures:
			#annoying we have to unpack this but i did this "quick and dirty" lol 
			features.append(feature)

	#get the date... a bit messy
	dateList = review['date'].split('-')
	dateObj = datetime.datetime(int(dateList[0]), int(dateList[1]), int(dateList[2]))
	epoch = datetime.datetime.utcfromtimestamp(1072915200) #so the number of seconds is smaller, we start from 2004 (yelp founding date)
	deltaTime = dateObj - epoch
	features.append(deltaTime.total_seconds())

#return list of more features
def getUserFeatures(userId):
	features = []
	user = users[userId]
	features.append(user['review_count'])
	features.append(user['average_stars'])
	features.append(user['votes']['useful'] + user['votes']['cool'] + user['votes']['funny'])

numTrainingExamples = 500
numTestExamples = 100
with open(review_filename, 'r') as review_file:
	n = 0
  	for line in review_file:
  		review = loads(line)
  		if n > numTrainingExamples + numTestExamples:
  			break
  		if n > numTrainingExamples:
  			testReviews.append(extractReviewFeatures(review))
  			testReviewActual.append(review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool'])
  		else:
  			trainingReviews.append(extractReviewFeatures(review))
  			trainingReviewActual.append(review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool'])
  		n += 1

#SOMETHING IS NOT QUITE RIGHT WITH THIS. SORRY I HAD TO RUN. GOTTA DEBUG A LITTLE BIT I GUESS...

print testReviews
print trainingReviews
print testReviewActual
print trainingReviewActual 
