from json import loads
import pylab as pl
import numpy as np
import scipy
import random
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
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
#Label 1 or zero based on whether the review actually has more than 10 upvotes
testReviewActualClassification = []
#Label 1 or zero based on whether the review actually has more than 10 upvotes
trainingReviewActualClassification = []
#Buckets for how many upvotes the review has 
testReviewActualClassification = []
#Buckets for how many upvotes the review has
trainingReviewActualClassification = []

# Dict of userid:user object
users = {}
# Dict of businessid: business object
businesses = {}

testReviewsEvenDistribution = []
testReviewsEvenDistributionActual = []
testReviewsEvenDistributionActualClassification = []

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

def getBusinessFeatures(businesses, businessId):
	features = []
	business = businesses[businessId]
	features.append(business['stars'])
	features.append(business['review_count'])
	isOpen = 0
	if business['open'] == True:
		isOpen = 1
	features.append(isOpen)
	return features

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
	ngram_vectorizer = CountVectorizer(analyzer='char', )
	features = []
	features.append(review['stars'])
	features.append(len(review['text']))
	userFeatures = getUserFeatures(users, review['user_id'])
	features.extend(userFeatures)
	businessFeatures = getBusinessFeatures(businesses, review['business_id'])
	features.extend(businessFeatures)
	features.append(parseReviewDate(review['date']))
	features.append()
	for i in xrange(len(features)):
		features[i] = float(features[i])
	return features

# Get all the users
with open(user_filename, 'r') as user_file:
	for line in user_file:
		user = (loads(line))
		users[user['user_id']] = user
#and the businesses
with open(businesses_filename, 'r') as businesses_file:
	for line in businesses_file:
		business = (loads(line))
		businesses[business['business_id']] = business

numTrainingExamples = 100000
numTestExamples = 1000
cutoff = 20
numAboveCutoff = 0
with open(review_filename, 'r') as review_file:
	for line in review_file:
  		review = loads(line)
  		numUpvotes = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
  		if numUpvotes > cutoff:
  			numAboveCutoff += 1
print numAboveCutoff

with open(review_filename, 'r') as review_file:
	n = 0
  	for line in review_file:
  		review = loads(line)
  		numUpvotes = 0
  		if n > numTrainingExamples + numTestExamples:
  			break
  		if n > numTrainingExamples:
  			numUpvotes = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
  			if len(testReviews) < numTestExamples:
  				testReviews.append(extractReviewFeatures(review))
  				testReviewActual.append(numUpvotes)
  				if numUpvotes > cutoff:
  					testReviewActualClassification.append(1)
  				else:
  					testReviewActualClassification.append(0)
  			if n%2 == 1:
  				if numUpvotes > cutoff:
  					testReviewsEvenDistributionActualClassification.append(1)
  					testReviewsEvenDistribution.append(extractReviewFeatures(review))
  					testReviewsEvenDistributionActual.append(numUpvotes)
  					n+=1
  			else:
  				if numUpvotes < cutoff:
  					testReviewsEvenDistributionActualClassification.append(0)
  					testReviewsEvenDistribution.append(extractReviewFeatures(review))
  					testReviewsEvenDistributionActual.append(numUpvotes)
  					n+=1
  		else:
  			numUpvotes = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
  			trainingReviews.append(extractReviewFeatures(review))
  			trainingReviewActual.append(numUpvotes)
  			if numUpvotes > cutoff:
  				trainingReviewActualClassification.append(1)
  			else:
  				trainingReviewActualClassification.append(0)
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

modelScaler = preprocessing.StandardScaler().fit(training_design_matrix)
test_design_matrix_scaled = modelScaler.transform(test_design_matrix)
training_design_matrix_scaled = modelScaler.transform(training_design_matrix)

even_test_design_matrix = np.array(testReviewsEvenDistribution)
even_test_design_matrix_scaled = modelScaler.transform(even_test_design_matrix)

test_outputs = np.array(testReviewActual)
training_outputs = np.array(trainingReviewActual)

training_outputs_classification = np.array(trainingReviewActualClassification)
test_outputs_classification = np.array(testReviewActualClassification)
even_test_outputs = np.array(testReviewsEvenDistributionActual)
even_test_outputs_classification = np.array(testReviewsEvenDistributionActualClassification)

#bootstrapping
numModels = 40
bootstrapDatasetSize = 1000
totalTrainScore = 0
totalTestScore = 0
evenDistTestScore = 0
for i in xrange(numModels):
  	newModelSet = []
  	newModelSetActual = []
  	newModelSetActualClassification = []
  	usedNumbers = {}
  	j = 0
  	while j < bootstrapDatasetSize:
  		rand = random.randint(0, len(trainingReviewActual)-1)
  		while rand in usedNumbers:
  			rand = random.randint(0, len(trainingReviewActual)-1)
  		if j%2 == 1:
  			if trainingReviewActualClassification[rand] == 1:
  				newModelSet.append(trainingReviews[rand])
  				newModelSetActualClassification.append(1)
  				newModelSetActual.append(trainingReviewActual[rand])
  				j += 1
  				usedNumbers[rand] = 1
  		else:
  			if trainingReviewActualClassification[rand] == 0:
  				newModelSet.append(trainingReviews[rand])
  				newModelSetActualClassification.append(0)
  				newModelSetActual.append(trainingReviewActual[rand])
  				j += 1
  				usedNumbers[rand] = 1
  	scaler = preprocessing.StandardScaler().fit(np.array(newModelSet))
	trainMatrix = scaler.transform(np.array(newModelSet))
	trainOutput = np.array(newModelSetActual)
	trainOutputClassification = np.array(newModelSetActualClassification)
	temporaryTestset = scaler.transform(test_design_matrix)
	temporaryEvenTestset = scaler.transform(even_test_design_matrix)
	clf = linear_model.LinearRegression()
	totalTrainScore += clf.fit(trainMatrix, trainOutputClassification).score(trainMatrix, trainOutputClassification)
	totalTestScore += clf.fit(trainMatrix, trainOutputClassification).score(temporaryTestset, test_outputs_classification)
	evenDistTestScore += clf.fit(trainMatrix, trainOutputClassification).score(temporaryEvenTestset, even_test_outputs_classification)

print "Average training score is " + str(totalTrainScore/numModels)
print "Average test score is " + str(totalTestScore/numModels)
print "Average test score on even distribution is " + str(evenDistTestScore/numModels)

print "Linear Regression training score"
clf = linear_model.LinearRegression()
print clf.fit(training_design_matrix_scaled, training_outputs).score(training_design_matrix_scaled, training_outputs)

print "\nLinear Regression test score"
print clf.fit(training_design_matrix_scaled, training_outputs).score(test_design_matrix_scaled, test_outputs)
print "\nLinear Regression score on evenly distributed test"
print clf.fit(training_design_matrix_scaled,training_outputs).score(even_test_design_matrix_scaled, even_test_outputs)

print "\nLogistic Regression training score"
logisticRegression = linear_model.LogisticRegression()
print logisticRegression.fit(training_design_matrix_scaled, training_outputs_classification).score(training_design_matrix_scaled, training_outputs_classification)

print "\nLogistic regression test score"
print logisticRegression.fit(training_design_matrix_scaled, training_outputs_classification).score(test_design_matrix_scaled, test_outputs_classification)
print "\nLogistic regression score on evenly distributed test"
print logisticRegression.fit(training_design_matrix_scaled,training_outputs_classification).score(even_test_design_matrix_scaled, even_test_outputs_classification)

print "\nNaive Bayes training score"
naiveBayes = GaussianNB()
print naiveBayes.fit(training_design_matrix_scaled, training_outputs_classification).score(training_design_matrix_scaled, training_outputs_classification)
print "\nNB score on evenly distributed test"
print naiveBayes.fit(training_design_matrix_scaled,training_outputs_classification).score(even_test_design_matrix_scaled, even_test_outputs_classification)

print "\nNaive Bayes test score"
print naiveBayes.fit(training_design_matrix_scaled, training_outputs_classification).score(test_design_matrix_scaled, test_outputs_classification)

print "\nSVM (SVC) training scorez"
supportVecMachine = svm.SVC(tol=1e-2, max_iter=3000)
#print supportVecMachine.fit(training_design_matrix_scaled,training_outputs_classification).score(training_design_matrix_scaled, training_outputs_classification)

print "\nSVM (SVC) test score"
#print supportVecMachine.fit(training_design_matrix_scaled,training_outputs_classification).score(test_design_matrix_scaled, test_outputs_classification)
print "\nSVM score on evenly distributed test"
#print supportVecMachine.fit(training_design_matrix_scaled,training_outputs_classification).score(even_test_design_matrix_scaled, even_test_outputs_classification)
#
#TODO:
#new scoring method
#get the text features
#
