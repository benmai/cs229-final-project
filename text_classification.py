from json import loads
import numpy as np
import scipy
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def getTextFeatures(num_training_reviews, num_test_reviews, cutoff):
	review_filename     = 'data/yelp_academic_dataset_review.json'
	review_texts_positive = []
	review_texts_negative = []

	with open(review_filename, 'r') as review_file:
	  reviews_read = 0
	  for line in review_file:
	    review = loads(line)
	    text = review['text']
	    votes = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
	    if votes > cutoff:
	      review_texts_positive.append(text)
	    else:
	      review_texts_negative.append(text)

	positive_sample = random.sample(review_texts_positive, (num_training_reviews + num_test_reviews)/2)
	negative_sample = random.sample(review_texts_negative, (num_training_reviews + num_test_reviews)/2)

	training_reviews = positive_sample[:num_training_reviews/2] + negative_sample[:num_training_reviews/2]
	test_reviews = positive_sample[num_training_reviews/2:] + negative_sample[num_training_reviews/2:]

	training_classifications = np.array([1 for i in xrange(num_training_reviews/2)] + [0 for i in xrange(num_training_reviews/2)])
	test_classifications = np.array([1 for i in xrange(num_test_reviews/2)] + [0 for i in xrange(num_test_reviews/2)])

	vectorizer = CountVectorizer(min_df=1)

	X = vectorizer.fit_transform(training_reviews + test_reviews)

	training_review_design = X[:num_training_reviews]
	test_review_design = X[num_training_reviews:]

	return (training_review_design, test_review_design, training_classifications, test_classifications)

if __name__ == '__main__':
  n_iter = 10
  training_score = 0
  test_score = 0
  for i in xrange(n_iter):
    (training_review_design, test_review_design, training_classifications, test_classifications) = getTextFeatures(30000, 250, 20)
    naive_bayes = MultinomialNB()
    model = naive_bayes.fit(training_review_design, training_classifications)
    training_score += model.score(training_review_design, training_classifications)
    print training_score
    test_score += model.score(test_review_design, test_classifications)
    print test_score
  training_score /= n_iter
  test_score /= n_iter

  print "\nNaive Bayes average training score"
  print training_score
  print "\nNaive Bayes average test score"
  print test_score
