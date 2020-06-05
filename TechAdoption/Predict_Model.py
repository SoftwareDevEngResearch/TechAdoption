""" Created by: Erin Peiffer, 12 May 2020

	Code to predict rates of adoption using Top_Features identified
	
"""
from Build_Model import (format_magpi, format_dataset, split_train_test, create_random_forest, predict_test_data, 
evaluate_fit, list_top_features, plot_top_features, plot_predicted_actual, plot_tree)
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pydot
import csv

''' TO ADD: include uncertainty parameters for each factor '''

def main_test_model():
	""" Build RFR Inputs """
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
	
	# input file to build model
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir="C:\Documents", title="Select File",  filetype=(("csv", "*.csv"),("all files","*.*")))
	filename_build = root.filename
	root.destroy()
	
	# call functions to build model
	df_build = format_magpi(filename_build,args.num_devices[0],args.num_questions[0])
	features_build, labels_build, feature_list_build = format_dataset(df_build)
	train_features_build, train_labels_build, test_features_build, test_labels_build = split_train_test(features_build, labels_build, args.testsize, args.randomstate)
	rf = create_random_forest(args.trees, args.randomstate, maxfeatures, train_features_build, train_labels_build)

	""" Test RFR Inputs """
	
	# input file to test model
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir="C:\Documents", title="Select File",  filetype=(("csv", "*.csv"),("all files","*.*")))
	filename_test = root.filename
	root.destroy()
	
	df_test = format_magpi(filename_test, args.num_devices[0], args.num_questions[0])
	features_test, labels_test, feature_list_test = format_dataset(df_test)
	predictions = predict_test_data(rf, features_test)
	errors, accuracy, rsquared = evaluate_fit(predictions, labels_test)
	importances = list_top_features(rf, feature_list_test)
	plot_top_features(importances, feature_list_test, args.color)
	plot_predicted_actual(labels_test, predictions, rsquared)
	plot_tree(rf, feature_list_test)

if __name__ == "__main__":
	main_test_model()


