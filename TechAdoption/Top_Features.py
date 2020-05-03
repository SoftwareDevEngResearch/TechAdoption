''' Erin Peiffer, 16 April 2020

	Code to identify most relevant factors in predicting adoption using decision trees
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydot
import csv

#class features(self):
#	def _init_(self):
	

def format_dataset(data):		
	''' Loads dataset, converts categorical data into numerical data, and splits into predictors and output'''
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
	# Train the model on training data
	rf.fit(train_features, train_labels)
	return rf

def predict_test_data(rf, test_features):
	''' Predict '''
	# Use the forest's predict method on the test data
	predictions = rf.predict(test_features)
	return predictions
	
def evaluate_fit(predictions, test_labels):
	''' Evaluate fit '''
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
	''' List of Top Features '''
	# Get numerical feature importances
	importances = list(rf.feature_importances_)
	# List of tuples with variable and importance
	feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
	# Sort the feature importances by most important first
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	# Print out the feature and importances 
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
	return importances

def plot_top_features(importances, feature_list):
	''' Plot Top Features '''
	# list of x locations for plotting
	x_values = list(range(len(importances)))
	# Make a bar chart
	plt.figure(figsize = (20,15), dpi = 100)
	plt.barh(x_values, importances, align = 'center', alpha = 0.8, color='orchid')
	# Tick labels for x axis
	plt.yticks(x_values, feature_list, fontsize = '33')
	# Axis labels and title
	plt.xlabel('Importance'); plt.ylabel('Variable'); plt.title('Variable Importances', fontsize = '35');
	plt.show()
	#plt.savefig('Variable_Importances.png', bbox_inches='tight', dpi = 500)

def plot_predicted_actual(test_labels, predictions, rsquared):
	''' Plots predicted versus actual '''
	plt.figure(dpi = 100)
	plt.plot(test_labels, predictions, 'k*')
	plt.plot([min(test_labels), max(test_labels)],[min(test_labels), max(test_labels)],'r-')
	plt.title('Cooking Methods', fontsize = 25); plt.xlabel('Actual Values', fontsize = 20); plt.ylabel('Predicted Values', fontsize = 20)
	plt.text(2,1, '$R^2$ ='+str(round(rsquared,3)), fontsize = 20) 
	plt.show()
	#plt.savefig('Prediced_Actual.png', bbox_inches='tight', dpi = 500)
	
def plot_tree(rf, feature_list):
	''' Visualize one tree '''		
	# Pull out one tree from the forest
	tree = rf.estimators_[5]
	# Pull out one tree from the forest
	tree = rf.estimators_[5]
	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	# Use dot file to create a graph
	(graph ) = pydot.graph_from_dot_file('tree.dot')
	# Write graph to a png file
	graph.write_png('tree.png')

def main():
	testsize = 0.25;	randomstate = 42;	trees = 1000;	maxfeatures = float(1/3)
	filename = r'G:\My Drive\Classes\ME 599 - Software Development\TechAdoption\TechAdoption\Dummy_data.csv'
	features, labels, feature_list = format_dataset(filename)
	train_features, train_labels, test_features, test_labels = split_train_test(features, labels, testsize, randomstate)
	rf = create_random_forest(trees, randomstate, maxfeatures, train_features, train_labels)
	predictions = predict_test_data(rf, test_features)
	errors, accuracy, rsquared = evaluate_fit(predictions, test_labels)
	importances = list_top_features(rf, feature_list)
	plot_top_features(importances, feature_list)
	plot_predicted_actual(test_labels, predictions, rsquared)
	#plot_tree(rf, feature_list)

if __name__ == "__main__":
	main()
