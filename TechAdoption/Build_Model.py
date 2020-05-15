""" Created by: Erin Peiffer, 12 May 2020

	Code to identify most relevant factors in predicting adoption using decision trees
"""
# TO DO: modify code so it allows for people to answer "no, I don't have 'improved' device" and reasons why
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import pandas as pd
import numpy as np
import argparse
import pydot
import csv


def format_magpi(file,num_devices,num_questions):
	""" Formats Magpi dataset to input into RFR 
	
	Parameters
    ----------
    file : csv
        Magpi data file to build model
    num_devices : int
        Number of devices included in dataset
	num_questions : int
		Number of questions asked for each device (should be the same for each device)

    Returns
    -------
    dataframe
        Formatted to have the first column list all of the devices used and their corresponding yes/no answer to each question
		
	"""		
	df = pd.read_csv(file)
	cols_to_drop = ['Created By','Last Submitter','Record Uuid','Start Record Date','End Record Date','Last Submission Date',
	'gps_stamp_latitude','gps_stamp_longitude','gps_stamp_accuracy']
	df = df.drop(cols_to_drop,axis=1) # drop columns not containing survey data

	# get list of devices
	l = df.columns.tolist()
	devices = []
	for col in range(num_devices):
		text = l[col]
		head, sep, tail = text.partition('.') 	# delete information before '.' e.g. cooking_devices.microwave --> microwave
		devices.append(tail)
		
	# get list of questions
	questions_list = []
	for col in range(num_devices,num_devices+num_questions):
		text = l[col]
		head, sep, tail = text.partition('.')
		questions_list.append(tail)
		
	# get one long list of relevant yes/no answers
	df_new = pd.DataFrame()
	dev_list = []	
	q_list = []
	df_devices = df.iloc[:,0:num_devices] # dataframe with just cooking_devices and yes/no
	for row in range(14):
		for col in range(num_devices):
			if df_devices.iloc[row, col] == 'Yes':
				dev_list.append(devices[col])
				for col1 in range((num_devices+(col*num_questions)),(num_devices+(col*num_questions)+num_questions)):
						q_list.append(df.iloc[row,col1]) # one long list of all questions (yes/no) for each device
	
	# add device and question data to new dataframe
	df_new['Devices'] = dev_list	
	for q in questions_list:
		df_new[q] = ''
	for row in range(len(dev_list)):
		for col in range(1,len(questions_list)+1):
			df_new.iloc[row,col] = q_list[col-1+(row*12)]
	return df_new
	

#def format_qualtrics(file, respondents):
	""" Formats Qualtrics dataset to input into RFR 
	"""
	
	'''df = pd.read_csv(file)
	cols_to_drop = ['StartDate','EndDate','Status','Progress','Duration (in seconds)','Finished',
	'RecordedDate','ResponseId','DistributionChannel','UserLanguage','Q1']
	df = df.drop(cols_to_drop, axis=1) # drop columns not containing survey data
	df.drop(df.index[[0,2]], axis = 0)
	
	dev_list = []
	for i in range(2,2+respondents):
		text = df.iloc[i,0]
		head,sep,tail = text.partition(',')
		dev_list.append(head)
		dev_list.append(tail)
		dev_list = list(filter(None, dev_list)) 
	print(dev_list)
	
	df.to_csv('test_qualtrics.csv', index = False)


input_file = r"G:\My Drive\Classes\ME 599 - Software Development\TechAdoption\TechAdoption\Qualtrics_Dummy_data.csv"
num_respondents = 10
format_qualtrics(input_file, num_respondents)'''
	
def format_dataset(data):		
	""" Loads dataset, converts categorical data into numerical data, and splits into predictors and output 
	
	Parameters
    ----------
    data : dataframe
        Formatted dataframe returned from format_magpi function

    Returns
    -------
    features : array
        Array of all the "x variables" (questions and responses to the questions)
	labels : array
		Array of all the "y variables" (devices)
	feature_list : list
		List with all the column names
		
	"""
	df = data
	features = pd.DataFrame()
	for name in list(df):
		features[name],name = pd.factorize(df[name]) 	# Converts categorical data into numerical data
	features += 1	# helps with math later so that accuracy calculation doesn't require dividing by 0
	labels = np.array(features['Devices'])	# Labels = y, features = x
	features= features.drop('Devices', axis = 1)
	feature_list = list(features.columns)
	features = np.array(features)
	return features, labels, feature_list

def split_train_test(features, labels, testsize, randomstate):	
	""" Splits data into train and test sets. Test size can be altered as an input. 
	
	Parameters
    ----------
    features : array
        Array of all the "x variables" (questions and responses to the questions), returned from format_dataset()
	labels : array
		Array of all the "y variables" (devices), returned from format_dataset()
	feature_list : list
		List with all the column names, returned from format_dataset()
	testsize : int
		Fraction of data that will be used to test the model. Default is 0.25
	randomstate : int
		Random state. Default is 42

    Returns
    -------
    train_features : list
        List of fraction of "x variables" used to build/train the model
	train_labels : list
		List of the fraction of "y variables" used to build/train the model
	test_features : list
		List of fraction of "x variables" used to test the model
	test_labels : list
		List of fraction of "y variables" used to test the model
	
	
	"""
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = testsize, random_state = randomstate)
	return train_features, train_labels, test_features, test_labels

def create_random_forest(trees, randomstate, maxfeatures, train_features, train_labels):
	""" Creates Random Forest Regression
	
	Parameters
    ----------
	trees : int
		Number of trees in random forest. Default is 1000
	randomstate : int
		Random state. Default is 42
	maxfeatures : int
		The number of features to consider when looking for the best split. Set to 1/3 per literature recommendations.
    train_features : list
        List of fraction of "x variables" used to build/train the model, returned from split_train_test()
	train_labels : list
		List of the fraction of "y variables" used to build/train the model, returned from split_train_test()
	
	Returns
    -------
	rf : model
		Random forest model built from train data
	
	"""
	# Instantiate model with (default) 1000 decision trees, randomstate = 42, jobs = 2, maxfeatures = float(1/3)
	rf = RandomForestRegressor(n_estimators = trees, random_state = randomstate, max_features = maxfeatures)
	rf.fit(train_features, train_labels)
	return rf

def predict_test_data(rf, test_features):
	""" Predicts outcomes from test data 
	
	Parameters
    ----------
	rf : model
		Random forest model built from train data, returned from create_random_forest()
	test_features : list
		List of fraction of "x variables" used to test the model, returned from split_train_test()
	
	Returns
    -------
	predictions : list
		List of predicted "y values" when using the test data (test_features) and the random forest model (rf)
	
	
	"""
	predictions = rf.predict(test_features)
	return predictions
	
def evaluate_fit(predictions, test_labels):
	""" Evaluates how predicted outcomes and actual outcomes compare with
	error (predicted-actual), accuracy, and r-squared values
	
	Parameters
    ----------
	predictions : list
		List of predicted "y values" when using the test data (test_features) and the random forest model (rf), returned from predict_test_data()
	test_labels : list
		List of fraction of "y variables" used to test the model, returned from split_train_test()
		
	Returns
    -------
	errors : list
		List of errors
	accuracy : float
		Percent accuracy of the model predicting the test dataset
	rsquared : float
		The r-squared value for test_labels versus predictions		
	
	"""
	errors = abs(predictions - test_labels)
	print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
	mape = 100 * (errors / test_labels)
	accuracy = 100 - np.mean(mape)
	print('Accuracy:', round(accuracy, 2), '%.')
	rsquared = r2_score(test_labels, predictions)
	return errors, accuracy, rsquared
	
def list_top_features(rf, feature_list):	
	""" Generates and prints list of top features influencing tech adoption
	
	Parameters
    ----------
	rf : model
		Random forest model built from train data, returned from create_random_forest()
	feature_list : list
		List with all the column names, returned from format_dataset()
	
	Returns
    -------
	importances : array
		List of the feature importances from 0 to 1 (least important to model to most important to model)
	
	"""
	importances = list(rf.feature_importances_)
	feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
	return importances

def plot_top_features(importances, feature_list,c):
	""" Plots Top Features in horizontal bar chart
	
	Parameters
    ----------
	importances : array
		List of the feature importances from 0 to 1 (least important to model to most important to model), returned from list_top_features()
	feature_list : list
		List with all the column names, returned from format_dataset()
	c : str
		Color of bar chart
	
	Returns
	-------
	'Plot_Variable_Importances.png' : png
		Horizontal bar chart of variable importances in descending order 
		
	
	"""
	x_values = list(range(len(importances)))
	plt.figure(figsize = (20,15), dpi = 300)
	importances_sorted, feature_list_sorted = (list(t) for t in zip(*sorted(zip(importances, feature_list))))
	plt.barh(x_values, importances_sorted, align = 'center', alpha = 0.8, color=c)
	plt.yticks(x_values, feature_list_sorted, fontsize = '33'); plt.xticks(fontsize = '30')
	plt.xlabel('Importance', fontsize = '33'); plt.title('Variable Importances', fontsize = '35');
	plt.savefig('Plot_Variable_Importances.png', bbox_inches='tight', dpi = 500)

def plot_predicted_actual(test_labels, predictions, rsquared):
	""" Plots predicted versus actual with rsquared and fitted line
	
	Parameters
    ----------
	predictions : list
		List of predicted "y values" when using the test data (test_features) and the random forest model (rf), returned from predict_test_data()
	test_labels : list
		List of fraction of "y variables" used to test the model, returned from split_train_test()
	rsquared : float
		The r-squared value for test_labels versus predictions, returned from evaluate_fit()
		
	Returns
	-------
	'Plot_Predicted_Actual.png' : png
		Scatter plot of predicted devices and actual devices with rsquared value
	
	"""
	plt.figure(dpi = 300)
	plt.plot(test_labels, predictions, 'k*')
	plt.plot([min(test_labels), max(test_labels)],[min(test_labels), max(test_labels)],'r-')
	plt.xticks(fontsize = 15); plt.yticks(fontsize = 15)
	plt.title('Cooking Methods', fontsize = 25); plt.xlabel('Actual Values', fontsize = 20); plt.ylabel('Predicted Values', fontsize = 20)
	x_loc = (max(test_labels) - 1)*0.75
	y_loc = (min(predictions)+1)*0.75
	plt.text(x_loc,y_loc, '$R^2$ ='+str(round(rsquared,3)), fontsize = 20) 
	plt.savefig('Plot_Predicted_Actual.png', bbox_inches='tight', dpi = 500)
	
def plot_tree(rf, feature_list):
	""" Visualize one decision tree 
	
	Parameters
    ----------
	rf : model
		Random forest model built from train data, returned from create_random_forest()
	feature_list : list
		List with all the column names, returned from format_dataset()
	
	Returns
    -------
	'Plot_tree.png' : png
		Figure of randomly generate decision tree for illustrative purposes
	
	"""	
	# Pull out one tree from the forest. Tree #5 randomly chosen
	tree = rf.estimators_[5]
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	(graph, ) = pydot.graph_from_dot_file('tree.dot')
	graph.write_png('Plot_tree.png')

def main():

	# constants 
	maxfeatures = float(1/3);
	
	# user inputs
	parser = argparse.ArgumentParser() 
	parser.add_argument('-nd', action='store', dest='num_devices', nargs='*', type=int, required=True, help='number of devices included in dataset')
	parser.add_argument('-nq', action='store', dest='num_questions', nargs='*', type=int, required=True, help='number of questions per device in dataset')
	parser.add_argument('-ts', action='store', dest='testsize', nargs='*', type=int, default=0.25, 
	help='proportion of the dataset used to train (build) the model, and proportion to test model? Default = 0.25 test, 0.75 train')
	parser.add_argument('-t', action='store', dest='trees', nargs='*', type=int, default=1000, help='number of trees in random forest regression. Default = 1000')
	parser.add_argument('-rs', action='store', dest='randomstate', nargs='*', type=int, default=42, help='random state. Default = 42')
	parser.add_argument('-c', action='store', dest='color', nargs='*', type=str, default='orchid', help='choose bar colors. Default = orchid. Documentation: https://matplotlib.org/2.0.2/api/colors_api.html')
	args = parser.parse_args()
	
	# input file
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir="C:\Documents", title="Select File",  filetype=(("csv", "*.csv"),("all files","*.*")))
	filename = root.filename
	root.destroy()
	
	# call functions
	df_new = format_magpi(filename,args.num_devices[0],args.num_questions[0])
	features, labels, feature_list = format_dataset(df_new)
	train_features, train_labels, test_features, test_labels = split_train_test(features, labels, args.testsize, args.randomstate)
	rf = create_random_forest(args.trees, args.randomstate, maxfeatures, train_features, train_labels)
	predictions = predict_test_data(rf, test_features)
	errors, accuracy, rsquared = evaluate_fit(predictions, test_labels)
	importances = list_top_features(rf, feature_list)
	plot_top_features(importances, feature_list,args.color)
	plot_predicted_actual(test_labels, predictions, rsquared)
	plot_tree(rf, feature_list)

if __name__ == "__main__":
	main()
