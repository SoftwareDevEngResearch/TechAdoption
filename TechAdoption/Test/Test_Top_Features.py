''' Test_Top_Features.py '''


''' def test_kepler_loc():
    p1 = jupiter(two_days_ago)
    p2 = jupiter(yesterday)
    exp = jupiter(today)
    obs = kepler_loc(p1, p2, 1, 1)
	assert exp == obs
	
	# use pytest
'''

def test_format_dataset():		
	''' Make sure data is being loaded correctly using dummy dataset'''
	exp_features = [[0]*6, [1]*6, [2]*6, [3]*6, [4]*6, [5]*6, [6]*6,[7]*6,[8]*6]
	exp_labels = [0 1 2 3 4 5 6 7 8]
	exp_feature_list = ['b', 'c', 'd', 'e', 'f', 'g']

	data = r'G:\My Drive\Classes\ME 599 - Software Development\TechAdoption\TechAdoption\Test\Test_Data.csv'
	return obs_features, obs_labels, obs_feature_list = test_format_dataset(data)
	assert exp_features == obs_features
	assert exp_lables == obs_labels
	assert exp_feature_list == obs_feature_list

def test_split_train_test():
	''' Make sure the shape of the test and train data sets are the same '''
	return train_features, train_labels, test_features, test_labels = split_train_test(obs_features, obs_labels, 0.25, 42)
	assert train_features.shape[1] == test_features.shape[1]
	assert train_labels.shape[1] == test_labels.shape[1]

def test_create_random_forest(trees, randomstate, maxfeatures, train_features, train_labels):
	# Instantiate model with 1000 decision trees, randomstate = 42, jobs = 2, maxfeatures = float(1/3)
	rf = RandomForestRegressor(n_estimators = trees, random_state = randomstate, max_features = maxfeatures)
	# Train the model on training data
	rf.fit(train_features, train_labels)
	return rf

def test_predict_test_data(rf, test_features):
	''' Make sure the length of predictions matches the length of test_labels ''' 
	predictions = rf.predict(test_features)
	return predictions
	assert len(predictions) == len(test_labels)
	
def test_evaluate_fit(predictions, test_labels):
	''' Make sure math works out '''
	# Calculate the absolute errors
	errors = abs(predictions - test_labels)
	# Print out the mean absolute error (mae)
	print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
	# Calculate mean absolute percentage error (MAPE)
	mape = 100 * (errors / test_labels)
	# Calculate and display accuracy
	accuracy = 100 - np.mean(mape)
	print('Accuracy:', round(accuracy, 2), '%.')
	rsquared = r2_score(test_labels, predictions)
	return errors, accuracy, rsquared
	
def list_top_features(rf, feature_list):	
	''' Make sure number of features is correct '''
	# Get numerical feature importances
	importances = list(rf.feature_importances_)
	# List of tuples with variable and importance
	feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
	# Sort the feature importances by most important first
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	# Print out the feature and importances 
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
	return importances
	

testsize = 0.25;	randomstate = 42;	trees = 1000;	maxfeatures = float(1/3)
filename = r'G:\My Drive\Classes\ME 599 - Software Development\TechAdoption\TechAdoption\Test\Test_Data.csv'

features, labels, feature_list = format_dataset(filename)
train_features, train_labels, test_features, test_labels = split_train_test(features, labels, testsize, randomstate)
rf = create_random_forest(trees, randomstate, maxfeatures, train_features, train_labels)
predictions = predict_test_data(rf, test_features)
errors, accuracy, rsquared = evaluate_fit(predictions, test_labels)
importances = list_top_features(rf, feature_list)



	
	
	
	
''' Make sure data set is loaded in correctly'''
# load test csv into dataframe and make sure dataframe is the same as fixed dataframe

''' Make sure the shape of the features is what is expected '''
print('The shape of our features is:', features.shape)
#The shape of our features is: (348, 9)


''' Make sure that the Features length and Labels length match '''
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

'''Training Features Shape: (261, 14)
Training Labels Shape: (261,)
Testing Features Shape: (87, 14)
Testing Labels Shape: (87,)'''