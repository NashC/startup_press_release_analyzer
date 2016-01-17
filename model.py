#spra_model.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pymongo import MongoClient
import math
import re
from scipy import sparse
from textblob import TextBlob
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import data_munging as dm
import matplotlib.pyplot as plt


def make_stop_words():
	'''
		Take in list of user-created stop words and join with Tfidf 'english' stop words.
		
		INPUT:
		- None

		OUTPUT:
		- New master list of stop words including user and model inputs
	'''
	new_stop_words = ['ha', "\'s", 'tt', 'ireach', "n\'t", 'wo', 'pv', 'tm', 'anite', 'rabichev', 'russell', '603', 'hana', 'atmel', 'radwin', 'se', 'doxee', 'lantto', 'publ', 'fpc1025', '855', 'il', '0344']
	#create temporary TfidfVectorizer object
	tfidf_temp = TfidfVectorizer(stop_words='english')
	#get Tfidf 'english' stop words from model
	stop_words = tfidf_temp.get_stop_words()
	#combine two lists of stop words
	result = list(stop_words) + new_stop_words
	return result


def print_top_words(nmf_fitted, tfidf_vectorizer, n_top_words):
	'''
		Print n most common words for each latent feature/topic in a corpus
		
		INPUT:
		- nmf_fitted: fitted NMF model object
		- tfidf_vectorizer: tfidf vectorizer model object
		- n_top_words: Number of top words to print

		OUTPUT:
		- Prints n top words for each latent topic
	'''
	feature_names = tfidf_vectorizer.get_feature_names()
	for topic_idx, topic in enumerate(nmf_fitted.components_):
		print("Topic #%d:" % topic_idx)
		print(" ".join([feature_names[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))
	pass


def tfidf(release_texts, max_features=None):
	#tfidf model
	custom_stop_words = make_stop_words()
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words, max_features=max_features)
	tfidf_sparse = tfidf_vectorizer.fit_transform(release_texts)

	#normalize row-wise so each row sums to one, and return sparse matrix
	tfidf_sparse = normalize(tfidf_sparse, axis=1, norm='l1')
	return tfidf_vectorizer, tfidf_sparse


def nmf(tfidf_sparse, n_components=8):
	#nmf model
	nmf = NMF(n_components=n_components, random_state=1)
	nmf.fit(tfidf_sparse)
	W = nmf.transform(tfidf_sparse)
	return nmf, W


def tfidf_nmf(release_texts, n_components=10, max_features=None):
	'''
		Creates and fits tfidf and NMF models.

		INPUT:
		- n_components: number of latent features for the NMF model to find
		- max_features: max number of features (vocabulary size) for the tfidf model to consider

		OUTPUT:
		- tfidf_vectorizer: tfidf model object
		- tfidf_sparse:tfidf sparse matrix
		- nmf: NMF model object
		- W: Feature matrix output from NMF factorization into W and H matrices
	'''
	#tfidf model
	custom_stop_words = make_stop_words()
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words, max_features=max_features)
	tfidf_sparse = tfidf_vectorizer.fit_transform(release_texts)

	#normalize row-wise so each row sums to one
	tfidf_sparse = normalize(tfidf_sparse, axis=1, norm='l1')

	#nmf model
	nmf = NMF(n_components=n_components, random_state=1)
	nmf.fit(tfidf_sparse)
	W = nmf.transform(tfidf_sparse)
	return tfidf_vectorizer, tfidf_sparse, nmf, W


def tfidf_traintestsplit(tfidf_sparse, test_size=0.2):
	'''
		Custom train_test_split for tfidf model. You can't do a typical five k-fold holding out 20 percent of the observations as the tfidf sparse matrix can't be broken apart and reassembled to acheive valid results. After studying a few academic papers online, I found you could set a random 20 percent of the values to zero for a five fold equivalent analysis. For each 'fold' you then have the train set be the matrix with the 20 percent of new zeros, and the test set is simply the original tfidf sparse matrix. We output these train and test matrices to another function to test the Mean Squared Error between them.

		INPUT:
		- tfidf_sparse: tfidf sparse matrix output after fit_transform
		- test_size: percent of observations to set to zero for the training set

		OUTPUT:
		- A_train: training tfidf sparse matrix with certain percent of entries set to zero
		- A_test: original unaltered tfidf sparse matrix
	'''
	A = tfidf_sparse
	train_size = 1. - test_size

	#create binary mask for training matrix to set test_size percent of entries to zero. Done by creating list of the appropriate number of 1's and 0's, then reshaping to size of tfidf sparse matrix.
	total_entries = A.shape[0] * A.shape[1]
	ones = np.ones(math.ceil(total_entries * train_size))
	zeros = np.zeros(math.floor(total_entries * test_size))
	r_temp = np.append(ones, zeros)

	#shuffle list to randomly disperse zeros throughout the mask and reshape
	np.random.shuffle(r_temp)
	R = np.reshape(r_temp, (A.shape))

	#convert mask into sparse matrix
	R = sparse.csr_matrix(R)

	#create training matrix by multiplying element-wise
	A_train = sparse.spmatrix.multiply(A,R)
	A_test = A
	return A_train, A_test


def grid_search_nmf_ncomponents(tfidf_sparse, folds, low, high, verbose=True):
	'''
		Custom grid search to find optimal value for number of latent features (n_components) to identify in the NMF model. This function outputs the Mean Squared Error for multiple values of n_components used to fit the NMF model. It also prints and appends the results to an external list.

		INPUT:
		- tfidf_sparse: tfidf sparse matrix
		- folds: number of train_test_split's to run for each n_components value
		- low: smallest value to begin the range of n_components value tests
		- high: highest value to end the range of n_components value tests
		- verbose: choose whether or not to print progress statements as test is running

		OUTPUT:
		- mse_export: list containing tuples of (n_components, MSE for that n_components value)
	'''
	tfidf_dense = tfidf_sparse.toarray()
	
	#temp variables to update as below for loops run
	mse_min = 99
	mse_min_ncomponents = -1
	mse_export = []
	
	#test entire range of n_components from low to high
	for i in xrange(low, high + 1):
		if verbose:
			print 'Fitting n_components = %d ...' %i
		mse_arr = []

		#test number of 'folds'
		for j in xrange(1, folds + 1):
			if verbose:
				print 'Testing fold # %d' %j
			
			#get A_train and A_test from above tfidf_traintestsplit() function
			test_size = 1./folds
			A_train, A_test = tfidf_traintestsplit(tfidf_sparse, test_size=test_size)
			
			#create and fit NMF model with A_train
			nmf_temp = NMF(n_components=i, random_state=1)
			nmf_temp.fit(A_train)
			
			#set W and H matrices from NMF
			W = nmf_temp.transform(A_train)
			H = nmf_temp.components_ 
			
			#create reconstructed tfidf predicted matrix from W and H
			tfidf_pred = np.dot(W, H)
			
			#get Mean Squared Error between the predicted and test matrices, append result
			mse_fold = mean_squared_error(A_test.toarray(), tfidf_pred)
			mse_arr.append(mse_fold)
		
		#get mean of all MSE's from each 'fold' in previous loop
		mse_temp = np.mean(mse_arr)

		#append (n_component, MSE) to mse_export array for later inspection and plotting
		mse_export.append((i, mse_temp))
		
		#set above variables if current MSE is the lowest yet
		if mse_temp < mse_min:
			mse_min = mse_temp
			mse_min_ncomponents = i
		if verbose:
			print 'MSE of n_components = %d: %.10f' %(i, mse_temp)
			print '-------------------------------'
	return mse_export


def nmf_component_plot(mse_arr, show=False):
	'''
		Create plot of NMF grid search results using matplotlib. Shows 1) MSE vs n_components and 2) Percent MSE Improvment vs n_components.
		
		INPUT:
		- mse_arr: List output from grid_search_nmf_ncomponents() above containing (MSE, n_components) tuple for entire range of n_components tested.
		- show: whether you want to show the plot or simply load the matplotlib objects for later display.

		OUTPUT:
		- returns nothing
		- Shows plot is show=True
	'''
	#create x values
	x = np.arange(1,len(mse_arr) + 1)

	#create y values for MSE plot
	y = [j[1] for j in mse_arr]
	
	#create y values for MSE percentage improvement subplot
	y_percent = []
	for i in xrange(len(mse_arr)-1):
		diff = abs(mse_arr[i+1][1] - mse_arr[i][1])
		y_percent.append(diff / mse_arr[i][1])
	y_percent = [y_percent[0]] + y_percent
	
	#make figure and axis objects, assign appropriate values to each axis
	f, axarr = plt.subplots(2, sharex=True, figsize=(12,12))
	axarr[0].plot(x, y)
	axarr[0].set_title('MSE vs NMF(n_components)')
	axarr[1].scatter(x, y_percent)
	axarr[1].set_title('MSE Percent Improvement per n_component unit increase')
	if show:
		plt.show()


def nmf_cross_val(tfidf_sparse, show=False):
	mse_arr = grid_search_nmf_ncomponents(tfidf_sparse, 5, 1, 15)
	nmf_component_plot(mse_arr, show=show)
	pass

if __name__ == '__main__':
	#import clean data from using data_munging.py program
	df_orig = dm.mongo_to_df('press', 'spra_master')
	df = dm.make_final_df(df_orig)
	release_texts = df['release_text']

	#create and fit both tf-idf and NMF models
	tfidf_vectorizer, tfidf_sparse = tfidf(release_texts, max_features=None)
	nmf, W = nmf(tfidf_sparse, n_components=8)

	#print top words from each latent topic found using the NMF model
	print_top_words(nmf, tfidf_vectorizer, n_top_words=15)

	#NMF cross val plot of MSE vs n_components 
	#nmf_cross_val(tfidf_sparse, show=True)
