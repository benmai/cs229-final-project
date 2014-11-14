from json import loads

businesses_filename = 'data/yelp_academic_dataset_business.json'
checkin_filename    = 'data/yelp_academic_dataset_checkin.json'
review_filename     = 'data/yelp_academic_dataset_review.json'
tip_filename        = 'data/yelp_academic_dataset_tip.json'
user_filename       = 'data/yelp_academic_dataset_user.json'

with open(review_filename, 'r') as review_file:
  for line in review_file:
  	# json.loads(line) will be a python dictionary of the review on the given line
  	review = json.loads(line)
  	# access each attribute in review like this
  	business_id = review['business_id']
