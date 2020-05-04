''' Erin Peiffer, 16 April 2020

	Code to predict rates of adoption using Top_Features identified
	
	'''
from Build_Model import (format_dataset, split_train_test, create_random_forest, predict_test_data, 
evaluate_fit, list_top_features, plot_top_features, plot_predicted_actual, plot_tree)
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydot
import csv

''' include uncertainty parameters for each factor '''

def main():
	''' Build RFR Inputs '''
	testsize = 0.25;	randomstate = 42;	trees = 1000;	maxfeatures = float(1/3)
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir="C:\Documents", title="Select File",  filetype=(("csv", "*.csv"),("all files","*.*")))
	filename_build = root.filename
	features_build, labels_build, feature_list_build = format_dataset(filename_build)
	train_features_build, train_labels_build, test_features_build, test_labels_build = split_train_test(features_build, labels_build, testsize, randomstate)
	rf = create_random_forest(trees, randomstate, maxfeatures, train_features_build, train_labels_build)

	''' Test RFR Inputs '''
	x_loc = 2; y_loc = 1
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir="C:\Documents", title="Select File",  filetype=(("csv", "*.csv"),("all files","*.*")))
	filename_test = root.filename
	features_test, labels_test, feature_list_test = format_dataset(filename_test)
	predictions = predict_test_data(rf, features_test)
	errors, accuracy, rsquared = evaluate_fit(predictions, labels_test)
	importances = list_top_features(rf, feature_list_test)
	plot_top_features(importances, feature_list_test)
	plot_predicted_actual(labels_test, predictions, rsquared, x_loc, y_loc)
	plot_tree(rf, feature_list_test)

if __name__ == "__main__":
	main()


