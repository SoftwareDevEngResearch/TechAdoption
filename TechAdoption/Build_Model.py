''' Erin Peiffer, 16 April 2020

	Code to identify most relevant factors in predicting adoption using decision trees
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydot
import csv
'''
import sys
import argparse

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser(
    description='This is a simple command-line program.'
    )
parser.add_argument('-i', '--input', required=True,
                    help='input file for data analysis'
                    )
args = parser.parse_args(sys.argv[1:])

# display a friendly message to the user
print("Hi there {}, it's nice to meet you!".format(args.name))'''

def format_magpi(file,num_devices,num_questions):
# use argparse to have someone insert question names for each question number
	df = pd.read_csv(file)
	cols_to_drop = ['Created By','Last Submitter','Record Uuid','Start Record Date','End Record Date','Last Submission Date',
	'gps_stamp_latitude','gps_stamp_longitude','gps_stamp_accuracy']
	df = df.drop(cols_to_drop,axis=1) #drop columns not containing survey data
	num_devices = 8
	# delete information before '.' e.g. cooking_devices.microwave --> microwave
	l = df.columns.tolist()
	devices = []
	# get list of devices
	for col in range(num_devices):
		text = l[col]
		head, sep, tail = text.partition('.')
		devices.append(tail)
	# get list of questions
	questions = []
	for col in range(num_devices,num_devices+num_questions):
		text = l[col]
		head, sep, tail = text.partition('.')
		questions.append(tail)
	df_new = pd.DataFrame()
	df_new['Devices'] = devices # row names
	for q in questions: # column names
		df_new[q] = ''
	for col in range(num_devices):
		for dev in devices:
		text = l[col]
		head,sep,tail = text.partition('.')
		if tail == dev:
			
		
	'''for col in l:
		head,sep,tail = text.partition('.')
		for device in devices:
			if tail == device'''
			
		
	
filename1 = r"G:\My Drive\Classes\ME 599 - Software Development\TechAdoption\TechAdoption\Magpi_Dummy_data.csv"
format_magpi(filename1, 8,12)

#def format_qualtrics(file):
	# pull from qualtrics? use argparse to locate user information? 
	

def format_dataset(data):		
	''' Loads dataset, converts categorical data into numerical data, and splits into predictors and output'''
	''' STILL NEED TO UPDATE WITH QUALTRICS CSV FILE FORMATTING '''
	
	df = pd.read_csv(data) 
	features = pd.DataFrame()
	for name in list(df):
		features[name],name = pd.factorize(df[name]) 	# Converts categorical data into numerical data
	labels = np.array(features['actual'])	# Labels = y, features = x
	features= features.drop('actual', axis = 1)
	feature_list = list(features.columns)
	features = np.array(features)
	return features, labels, feature_list

def split_train_test(features, labels, testsize, randomstate):	
	''' Splits data into train and test sets. Test size can be altered.'''
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = testsize, random_state = randomstate)
	return train_features, train_labels, test_features, test_labels

def create_random_forest(trees, randomstate, maxfeatures, train_features, train_labels):
	''' Random Forest '''
	# Instantiate model with 1000 decision trees, randomstate = 42, jobs = 2, maxfeatures = float(1/3)
	rf = RandomForestRegressor(n_estimators = trees, random_state = randomstate, max_features = maxfeatures)
	rf.fit(train_features, train_labels)
	return rf

def predict_test_data(rf, test_features):
	''' Predict '''
	predictions = rf.predict(test_features)
	return predictions
	
def evaluate_fit(predictions, test_labels):
	''' Evaluate fit '''
	errors = abs(predictions - test_labels)
	print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
	mape = 100 * (errors / test_labels)
	accuracy = 100 - np.mean(mape)
	print('Accuracy:', round(accuracy, 2), '%.')
	rsquared = r2_score(test_labels, predictions)
	return errors, accuracy, rsquared
	
def list_top_features(rf, feature_list):	
	''' List of Top Features '''
	importances = list(rf.feature_importances_)
	feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
	return importances

def plot_top_features(importances, feature_list):
	''' Plot Top Features '''
	x_values = list(range(len(importances)))
	plt.figure(figsize = (20,15), dpi = 100)
	importances_sorted, feature_list_sorted = (list(t) for t in zip(*sorted(zip(importances, feature_list))))
	plt.barh(x_values, importances_sorted, align = 'center', alpha = 0.8, color='orchid')
	plt.yticks(x_values, feature_list_sorted, fontsize = '33'); plt.xticks(fontsize = '30')
	plt.xlabel('Importance', fontsize = '33'); plt.ylabel('Variable'); plt.title('Variable Importances', fontsize = '35');
	plt.show()
	#plt.savefig('Variable_Importances.png', bbox_inches='tight', dpi = 500)

def plot_predicted_actual(test_labels, predictions, rsquared, x_loc, y_loc):
	''' Plots predicted versus actual '''
	plt.figure(dpi = 100)
	plt.plot(test_labels, predictions, 'k*')
	plt.plot([min(test_labels), max(test_labels)],[min(test_labels), max(test_labels)],'r-')
	plt.xticks(fontsize = 15); plt.yticks(fontsize = 15)
	plt.title('Cooking Methods', fontsize = 25); plt.xlabel('Actual Values', fontsize = 20); plt.ylabel('Predicted Values', fontsize = 20)
	plt.text(x_loc,y_loc, '$R^2$ ='+str(round(rsquared,3)), fontsize = 20) 
	plt.show()
	#plt.savefig('Prediced_Actual.png', bbox_inches='tight', dpi = 500)
	
def plot_tree(rf, feature_list):
	''' Visualize one tree '''		
	# Pull out one tree from the forest
	tree = rf.estimators_[5]
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	(graph, ) = pydot.graph_from_dot_file('tree.dot')
	graph.write_png('tree.png')
'''
def main():
	testsize = 0.25;	randomstate = 42;	trees = 1000;	maxfeatures = float(1/3);  x_loc = 2; y_loc = 1
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir="C:\Documents", title="Select File",  filetype=(("csv", "*.csv"),("all files","*.*")))
	filename = root.filename
	features, labels, feature_list = format_dataset(filename)
	train_features, train_labels, test_features, test_labels = split_train_test(features, labels, testsize, randomstate)
	rf = create_random_forest(trees, randomstate, maxfeatures, train_features, train_labels)
	predictions = predict_test_data(rf, test_features)
	errors, accuracy, rsquared = evaluate_fit(predictions, test_labels)
	importances = list_top_features(rf, feature_list)
	plot_top_features(importances, feature_list)
	plot_predicted_actual(test_labels, predictions, rsquared,x_loc, y_loc)
	plot_tree(rf, feature_list)

if __name__ == "__main__":
	main()
'''