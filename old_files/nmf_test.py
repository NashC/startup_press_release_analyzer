#nmf_test.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from nltk import tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
import math
import re
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

def tfidf_traintestsplit(tfidf_sparse, test_size=0.2):
	# A = tfidf_sparse.toarray()
	A = tfidf_sparse
	total_entries = A.shape[0] * A.shape[1]
	# print 'total_entries =', total_entries
	train_size = 1. - test_size
	# print 'train_size =', train_size
	ones = np.ones(math.ceil(total_entries * train_size))
	zeros = np.zeros(math.floor(total_entries * test_size))
	# print 'ones length =', len(ones)
	# print 'zeros length =', len(zeros)
	r_temp = np.append(ones, zeros)
	np.random.shuffle(r_temp)
	# print 'total entries = ', total_entries
	# print 'r_temp shape = ', r_temp.shape
	# print 'tfidf shape = ', A.shape
	R = np.reshape(r_temp, (A.shape))
	R_flip = np.logical_not(R)

	R = sparse.csr_matrix(R)
	R_flip = sparse.csr_matrix(R_flip)
	# print 'R.shape =', R.shape
	# print 'R_flip.shape = ', R_flip.shape

	A_train = sparse.spmatrix.multiply(A,R)
	A_test = A
	return A_train, A_test

def grid_search_nmf_ncomponents(tfidf, folds, low, high, export_array):
	tfidf_dense = tfidf.toarray()
	mse_min = 99
	mse_min_ncomponents = -1
	for i in xrange(low, high + 1):
		print 'Fitting n_components = %d ...' %i
		mse_arr = []
		for j in xrange(1, folds + 1):
			print 'Testing fold # %d' %j
			test_size = 1./folds
			A_train, A_test = tfidf_traintestsplit(tfidf, test_size=test_size)
			nmf_temp = NMF(n_components=i, random_state=1)
			nmf_temp.fit(A_train)
			W = nmf_temp.transform(A_train)
			H = nmf_temp.components_
			tfidf_pred = np.dot(W, H)
			mse_fold = mean_squared_error(A_test.toarray(), tfidf_pred)
			mse_arr.append(mse_fold)
		mse_temp = np.mean(mse_arr)
		export_array.append((i, mse_temp))
		if mse_temp < mse_min:
			mse_min = mse_temp
			mse_min_ncomponents = i
		print 'MSE of n_components = %d: %.10f' %(i, mse_temp)
		print '-------------------------------'
	pass

def gen_temp_matrix():
	q1 = np.random.randint(1,5, size=(50,1))
	q2 = np.random.randint(7,11, size=(50,1))
	q3 = np.random.randint(7,11, size=(50,1))
	q4 = np.random.randint(1,5, size=(50,1))
	top = np.hstack((q1,q2))
	bottom = np.hstack((q3,q4))
	result = np.vstack((top, bottom))
	return result

def nmf_test(A, folds, low, high, export_array):
	mse_min = 99
	mse_min_ncomponents = -1
	for i in xrange(low, high + 1):
		print 'Fitting n_components = %d ...' %i
		mse_arr = []
		for j in xrange(1, folds + 1):
			print 'Testing fold # %d' %j
			nmf_temp = NMF(n_components=i, random_state=1)
			nmf_temp.fit(A)
			W = nmf_temp.transform(A)
			H = nmf_temp.components_
			tfidf_pred = np.dot(W, H)
			mse_fold = mean_squared_error(A, tfidf_pred)
			mse_arr.append(mse_fold)
		mse_temp = np.mean(mse_arr)
		export_array.append((i, mse_temp))
		if mse_temp < mse_min:
			mse_min = mse_temp
			mse_min_ncomponents = i
		print 'MSE of n_components = %d: %.10f' %(i, mse_temp)
		print '-------------------------------'
	print "Optimal n_components = %d" %mse_min_ncomponents

H_test = np.array([[2,1,0],[0,0,1]])
W_test = gen_temp_matrix()
A_test = np.dot(W_test,H_test)
A_sparse = sparse.csr_matrix(A_test)

mse_test1 = []

# nmf_test(A_test, 5, 1, 10, mse_test1)
grid_search_nmf_ncomponents(A_sparse, 5, 1, 4, mse_test1)

