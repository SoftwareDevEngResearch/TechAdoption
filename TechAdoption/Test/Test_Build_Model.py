""" Created by: Erin Peiffer, 12 May 2020 

"""
from Build_Model import (format_magpi, format_dataset, split_train_test, create_random_forest, predict_test_data, 
evaluate_fit, list_top_features, plot_top_features, plot_predicted_actual, plot_tree)
import pytest
import pandas as pd
import csv


def test_format_magpi1():
	''' Make sure that number of columns are correct '''
	data = 'testMagpi.csv'
	exp_df = 6
	questions = 2
	options = 2
	obs_df, obs_options = format_magpi(data,options,questions)
	assert exp_df == len(obs_df.columns)

def test_format_magpi2():
	''' Make sure that the number of options/devices are correct '''
	data = 'testMagpi.csv'
	questions = 2
	exp_options = 2
	obs_df, obs_options = format_magpi(data,options,questions)
	assert exp_options == len(obs_options)	

def test_format_dataset1():		
	''' Make sure features are loaded correctly using dummy dataset'''
	exp_features = [[0]*6, [1]*6, [2]*6, [3]*6, [4]*6, [5]*6, [6]*6,[7]*6,[8]*6]
	data = 'testdata.csv'
	obs_features, obs_labels, obs_feature_list = test_format_dataset(data)
	assert exp_features == obs_features
	
def test_format_dataset2():		
	''' Make sure labels are loaded correctly using dummy dataset'''
	exp_labels = [0 1 2 3 4 5 6 7 8]
	data = 'testdata.csv'
	obs_features, obs_labels, obs_feature_list = test_format_dataset(data)
	assert exp_lables == obs_labels

def test_format_dataset3():		
	''' Make sure feature list is loaded correctly using dummy dataset'''
	exp_feature_list = ['b', 'c', 'd', 'e', 'f', 'g']
	data = 'testdata.csv'
	obs_features, obs_labels, obs_feature_list = test_format_dataset(data)
	assert exp_feature_list == obs_feature_list

def test_split_train_test1():
	''' Make sure the shape of the train and test features (x) data sets are the same '''
	data = 'testdata.csv'
	df = pd.read_csv(data)
	obs_features = df[df.columns[0]]
	obs_labels = df[df.columns[1:6]]
	train_features, train_labels, test_features, test_labels = split_train_test(obs_features, obs_labels, 0.25, 42)
	assert train_features.shape[1] == test_features.shape[1]

def test_split_train_test2():
	''' Make sure the shape of the train and test labels (y) data sets are the same '''
	data = 'testdata.csv'
	df = pd.read_csv(data)
	obs_features = df[df.columns[0]]
	obs_labels = df[df.columns[1:6]]
	train_features, train_labels, test_features, test_labels = split_train_test(obs_features, obs_labels, 0.25, 42)
	assert train_labels.shape[1] == test_labels.shape[1]

''' Not sure how to test the below function '''
def test_create_random_forest():
	trees = 1000
	random_state = 42
	maxfeatures = float(1/3)
	data = 'testdata.csv'
	df = pd.read_csv(data)
	features = df[df.columns[0]]
	labels = df[df.columns[1:6]]
	rf = RandomForestClassifier(n_estimators = trees, random_state = randomstate, max_features = maxfeatures)
	rf.fit(features, labels)
	return rf

def test_predict_test_data():
	''' Make sure the length of predictions matches the length of test_labels ''' 
	data = 'testdata.csv'
	df = pd.read_csv(data)
	labels = df[df.columns[1:6]]
	features = df[df.columns[0]]
	predictions = rf.predict(features)
	assert len(predictions) == len(labels)
	
def test_evaluate_fit1():
	''' Test errors for perfect prediction '''
	predictions = [1,1,1,1,1]
	actual = [1,1,1,1,1]
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_errors = 0
	exp_accuracy = 100
	assert exp_errors = obs_errors
	assert exp_accuracy = obs_accuracy
	
def test_evaluate_fit2():
	''' Test errors for worst fit prediction '''
	predictions = [0,0,0,0,0]
	actual = [1,1,1,1,1]
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_errors = len(predictions) # WHAT SHOULD THIS NUMBER BE??
	assert exp_errors = obs_errors

def test_evaluate_fit3():
	''' Test accuracy for perfect prediction '''
	predictions = [1,1,1,1,1]
	actual = [1,1,1,1,1]
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_accuracy = 100
	assert exp_accuracy = obs_accuracy

def test_evaluate_fit4():
	''' Test accuracy for worst fit prediction '''
	predictions = [0,0,0,0,0]
	actual = [1,1,1,1,1]
	obs_errors, obs_accuracy = evaluate_fit(predictions, actual)
	exp_accuracy = 0
	assert exp_accuracy = obs_accuracy

	
def test_list_top_features(rf, feature_list):	
	''' Make sure importances are sorted in descending order '''
	
	data = 'testdata.csv'
	df = pd.read_csv(data)
	features_list = list(df.columns)	
	importances = list_top_features(rf,features_list)
	assert importances[0] >= importances[-1]




