# coding: utf-8
import pandas as pd
import numpy as np
import json


def clean_data(source, destination):
	'''
	Takes in a path to a csv dataset and saves that dataset with appropriate
	features extracted, in csv format. Also fills in missing values using
	forward filling.

	In other words, text features get separated out into into binary 0 and 1
	in different columns.

	NOTE: Assumes final column consists of feature labels. All other columns
	will be modified if they are text fields.
	'''

	df = pd.read_csv(source)

	# Fill missing values using forward filling
	fill_method = 'ffill'
	df = df.fillna(method=fill_method)

	col_names = list(df.columns)

	# All but last column are data columns
	data_cols = col_names[:-1]
	orig_data = df[data_cols]
	target = df[col_names[-1]]

	# Extracts features from text fields
	cleaned_data = pd.get_dummies(orig_data)

	# Put extracted features together with target column
	cleaned_dataset = pd.concat([cleaned_data, target], axis=1)

	# Save as a csv
	cleaned_dataset.to_csv(destination, index=False)


def seperate_data_target(df):
	'''
	Splits DataFrame data into the attributes and target. Assumes last column
	is the feature column.
	'''

	col_names = list(df.columns)

	# All but last column are data columns
	attribute_cols = col_names[:-1]
	attributes = df[attribute_cols]
	target = df[col_names[-1]]

	return (attributes, target)

def split_data(df, percentage_training):
	'''
	Splits data into a training set and test set using the
	percentage provided. Assumes the classification is in the
	final column and uses all other columns as input
	attributes (vector X).
	'''

	data, target = seperate_data_target(df)

	n_examples, n_attributes = data.shape
	cutoff = int((percentage_training) * n_examples)

	training_set = (data[:cutoff].values, target[:cutoff].values)
	test_set = (data[cutoff:].values, target[cutoff:].values)

	return training_set, test_set


def shuffle_dataframe(df):
	'''
	Takes in a pandas DataFrame and shuffles around the rows and
	resets the indices to ensure that rows are randomized. Useful especially
	if the dataset is sequential in its nature (ex: population over
	a period of time)
	'''
	shuffled = df.sample(frac=1).reset_index(drop=True)
	return shuffled


def read_file(filename):
	'''
	Takes in a filename, returns a pandas DataFrame and shuffles the rows
	'''
	df = pd.read_csv(filename)
	return shuffle_dataframe(df)
