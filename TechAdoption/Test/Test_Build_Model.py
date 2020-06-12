""" Created by: Erin Peiffer, June 2020 

"""
from ..Build_Model import (format_magpi, format_dataset, split_train_test, create_random_forest, predict_test_data, 
evaluate_fit, list_top_features, plot_top_features, plot_predicted_actual, plot_tree)
import pandas as pd
import numpy as np


""" df for format_dataset, split_train_test """
df = pd.DataFrame()
df['Devices'] = ['a','b','c','d','e','f','g','h','i']
cols = ['b','c','d','e','f','g']
for col in cols:
	df[col] = [1,2,3,4,5,6,7,8,9]

def test_format_dataset1():		
	"""Make sure features are loaded correctly using dummy dataset"""
	exp_features = [[1]*6, [2]*6, [3]*6, [4]*6, [5]*6, [6]*6,[7]*6,[8]*6,[9]*6]
	exp_features = np.array(exp_features)
	obs_features, obs_labels, obs_feature_list = format_dataset(df)
	assert np.all(exp_features == obs_features)
	
def test_format_dataset2():		
	""" Make sure labels are loaded correctly using dummy dataset"""
	exp_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	obs_features, obs_labels, obs_feature_list = format_dataset(df)
	assert np.all(exp_labels == obs_labels)

def test_format_dataset3():		
	""" Make sure feature list is loaded correctly using dummy dataset"""
	exp_feature_list = ['b', 'c', 'd', 'e', 'f', 'g']
	obs_features, obs_labels, obs_feature_list = format_dataset(df)
	assert np.all(exp_feature_list == obs_feature_list)

def test_split_train_test2():
	""" Make sure the shape of the train and test labels (y) data sets are the same """
	obs_features = df[df.columns[0]]
	obs_labels = df[df.columns[1:6]]
	train_features, train_labels, test_features, test_labels = split_train_test(obs_features, obs_labels, 0.25, 42)
	assert train_labels.shape[1] == test_labels.shape[1]

def test_predict_test_data():
	""" Make sure the length of predictions matches the length of test_labels """
	trees = 1000
	random_state = 42
	maxfeatures = float(1/3)
	features =  [[1]*6, [2]*6, [3]*6, [4]*6, [5]*6, [6]*6,[7]*6,[8]*6,[9]*6]
	features = np.array(features)
	labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	rf = create_random_forest(trees, random_state, maxfeatures, features, labels)
	predictions = predict_test_data(rf, features)
	assert len(predictions) == len(labels)

def test_evaluate_fit1():
	""" Test errors for perfect prediction """
	predictions = np.array([1,1,1,1,1])
	actual = np.array([1,1,1,1,1])
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_errors = 0
	exp_accuracy = 100
	assert np.all(exp_errors == obs_errors)
	
def test_evaluate_fit2():
	""" Test errors for worst fit prediction """
	predictions = np.array([0,0,0,0,0])
	actual = np.array([1,1,1,1,1])
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	print('OBS = ',obs_errors)
	exp_errors = [1,1,1,1,1]
	assert np.all(exp_errors == obs_errors)

def test_evaluate_fit3():
	""" Test accuracy for perfect prediction """
	predictions = np.array([1,1,1,1,1])
	actual = np.array([1,1,1,1,1])
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_accuracy = 100
	assert exp_accuracy == obs_accuracy

def test_evaluate_fit4():
	""" Test accuracy for worst fit prediction """
	predictions = np.array([0,0,0,0,0])
	actual = np.array([1,1,1,1,1])
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_accuracy = 0
	assert exp_accuracy == obs_accuracy

	
def test_list_top_features():	
	""" Make sure importances are sorted in descending order """
	trees = 1000
	random_state = 42
	maxfeatures = float(1/3)
	features =  [[1]*6, [2]*6, [3]*6, [4]*6, [5]*6, [6]*6,[7]*6,[8]*6,[9]*6]
	features = np.array(features)
	labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	rf = create_random_forest(trees, random_state, maxfeatures, features, labels)
	features_list = list(df.columns)	
	importances = list_top_features(rf,features_list)
	assert importances[0] >= importances[-1]



