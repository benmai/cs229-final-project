from json import loads
import pylab as pl
import numpy as np
import scipy
from sklearn import preprocessing
from sklearn import linear_model
import datetime
#not sure i did these imports right yet, lets see if i get a chance to test

businesses_filename = 'data/yelp_academic_dataset_business.json'
checkin_filename    = 'data/yelp_academic_dataset_checkin.json'
review_filename     = 'data/yelp_academic_dataset_review.json'
tip_filename        = 'data/yelp_academic_dataset_tip.json'
user_filename       = 'data/yelp_academic_dataset_user.json'

# List of lists of features
testReviews = []
# List of feature lists
trainingReviews = []
# List of integers
testReviewActual = []
# List of integers
trainingReviewActual = []
# Dict of userid:user object
users = {}

# Starting with lists because they are what I know, 
# I figure later we can go through and convert each one into a numpy array

def parseReviewDate(date):
	#get the date... a bit messy
	dateList = date.split('-')
	dateObj = datetime.datetime(int(dateList[0]), int(dateList[1]), int(dateList[2]))
	#so the number of seconds is smaller, we start from 2004 (yelp founding date)
	epoch = datetime.datetime.utcfromtimestamp(1072915200)
	deltaTime = dateObj - epoch	
	return deltaTime.total_seconds()

def parseYelpingSinceDate(date):
	dateList = date.split('-')
	dateObj = datetime.datetime(int(dateList[0]), int(dateList[1]), 1)
	epoch = datetime.datetime.utcfromtimestamp(1072915200)
	deltaTime = dateObj - epoch	
	return deltaTime.total_seconds()

# Return list of more features
def getUserFeatures(users, userId):
	features = []
	user = users[userId]
	features.append(user['review_count'])
	features.append(user['average_stars'])
	features.append(user['votes']['useful'] + user['votes']['cool'] + user['votes']['funny'])
	features.append(user['votes']['useful'])
	features.append(user['votes']['cool'])
	features.append(user['votes']['funny'])
	features.append(parseYelpingSinceDate(user['yelping_since']))
	# Number of friends
	features.append(len(user['friends']))
	# Compliment types listed on yelp site here: http://officialblog.yelp.com/2013/03/compliments-theyre-free-give-them.html
	compliment_types = ["profile", "funny", "cute", "plain", "writer", "list", "note", "photos", "hot", "more", "cool"];
	for compliment_type in compliment_types:
		num_compliments = user['compliments'][compliment_type] if compliment_type in user['compliments'] else 0
		features.append(num_compliments)
	return features

# Right now just gets the features in the review, not the text
def extractReviewFeatures(review):
	features = []
	features.append(review['stars'])
	features.append(len(review['text']))
	userFeatures = getUserFeatures(users, review['user_id'])
	features.extend(userFeatures)
	features.append(parseReviewDate(review['date']))
	return features

# Get all the users
with open(user_filename, 'r') as user_file:
	for line in user_file:
		user = (loads(line))
		users[user['user_id']] = user

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

# print 'testReviews:'
# print testReviews
# print 'trainingReviews:'
# print trainingReviews
# print 'testReviewActual:'
# print testReviewActual
# print 'trainingReviewActual:'
# print trainingReviewActual 

test_design_matrix = np.array(testReviews)
training_design_matrix = np.array(trainingReviews)

test_outputs = np.array(testReviewActual)
training_outputs = np.array(trainingReviewActual)

# print test_design_matrix
# print training_design_matrix


clf = linear_model.LinearRegression()
clf.fit(training_design_matrix, training_outputs)
