import argparse
import os.path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def prep_data(df):
	split_index = int(round(df.shape[0]*0.8))

	train_set = df.iloc[0:split_index]
	test_set = df.iloc[split_index:,]

	X_train = train_set.drop('Income', axis=1)
	y_train = train_set['Income']

	X_test = test_set.drop('Income', axis=1)
	y_test = test_set['Income']

	min_max_scaler = MinMaxScaler()

	train_data_knn = min_max_scaler.fit_transform(X_train)
	train_labels_knn = y_train
	test_data_knn = min_max_scaler.transform(X_test)
	test_labels_knn = y_test

	std_scaler = StandardScaler()

	train_data_std = std_scaler.fit_transform(X_train)
	train_labels_std = y_train
	test_data_std = std_scaler.transform(X_test)
	test_labels_std = y_test

	return (X_train, y_train, X_test, y_test,
			train_data_knn, train_labels_knn, test_data_knn, test_labels_knn, 
			train_data_std, train_labels_std, test_data_std, test_labels_std)

def baseline(x_train, y_train, x_test, y_test):
	print '\n\n-------------------Baseline-------------------'
	# Baseline
	baseline_clf = DummyClassifier(strategy = 'most_frequent').fit(X_train,y_train)
	print 'Accuracy When Guessing Most Frequent: {:.2f}'.format(baseline_clf.score(X_test, y_test))

	# Bayes Error Rate
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	print '\n\n-------------------Bayes error rate-------------------'
	print 'KNN accuracy with k=1 %s' % metrics.accuracy_score(y_test, y_pred)

def knn(shouldTune, x_train, y_train, x_test, y_test):
	print '\n\n-------------------KNN-------------------'
	if shouldTune:
		model = KNeighborsClassifier()
		params = {"n_neighbors": np.arange(1, 31, 1),"metric": ["euclidean", "cityblock"]}

		grid = RandomizedSearchCV(model, params)
		grid.fit(x_train, y_train)
		acc = grid.score(x_test, y_test)

		print '[INFO] randomized search best parameters: {}'.format(grid.best_params_)
		print 'Accuracy: %f \n' % acc
		print classification_report(y_test, grid.predict(x_test))

	else:
		model = KNeighborsClassifier(n_neighbors=29, metric='euclidean')
		model.fit(x_train, y_train)

		print '[INFO] n_neighbors: 29, metric: euclidean'
		acc = metrics.accuracy_score(y_test, model.predict(x_test))
		print 'Accuracy: %f \n' % acc
		print classification_report(y_test, model.predict(x_test))

def rf(x_train, y_train, x_test, y_test):
	print '\n\n-------------------Random Forest Classifier-------------------'

	model = RandomForestClassifier(bootstrap=False, min_samples_leaf=5, n_estimators=800, min_samples_split=5, max_features='auto', max_depth=16)
	model.fit(x_train, y_train)

	print '[INFO] bootstrap: False, min_samples_leaf: 5, n_estimators: 800, min_samples_split: 5, max_features: auto, max_depth: 16'
	acc = metrics.accuracy_score(y_test, model.predict(x_test))
	print 'Accuracy %f \n' % acc
	print classification_report(y_test, model.predict(x_test))

def lr(shouldTune, x_train, y_train, x_test, y_test):
	print '\n\n-------------------Logistic Regression-------------------'

	if shouldTune:
		logistic = LogisticRegression()
		hyperparameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2'] }

		clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
		best_model = clf.fit(x_train, y_train)

		print("[INFO] grid search best parameters: {}".format(clf.best_params_))
		acc = best_model.score(x_test, y_test)
		print 'Accuracy: %f \n' % acc
		print(classification_report(y_test, best_model.predict(x_test)))

	else:
		model = LogisticRegression(C=0.1, penalty='l1')
		model.fit(x_train, y_train)

		print '[INFO] C: 0.1, penalty: l1'
		acc = metrics.accuracy_score(y_test, model.predict(x_test))
		print 'Accuracy: %f \n' % acc
		print classification_report(y_test, model.predict(x_test))

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='eval.py runs the three classification models used in this \
												  assignment. It takes two optional flags.')
	parser.add_argument('-t', '--tune', action='store_true', help='Use -t flag to tune hyperparameters when \
												running the script. Otherwise, evaluates models using default \
												hyperparameters found during previous tuning.')
	parser.add_argument('-c', '--con', action='store_true', help='Use -c flag to use dataset_con.csv when \
																  running the script. Otherwise, evaluates models using \
																  dataset_bin.csv')

	try:
		args = parser.parse_args()
	except IOError as msg:
		parser.error(str(msg))

	if args.con:
		df = pd.read_csv('data/dataset_con.csv')
	else:
		df = pd.read_csv('data/dataset_bin.csv')

	(X_train, y_train, X_test, y_test,
	train_data_knn, train_labels_knn, test_data_knn, test_labels_knn, 
	train_data_std, train_labels_std, test_data_std, test_labels_std) = prep_data(df)

	baseline(X_train, y_train, X_test, y_test)

	knn(args.tune, train_data_knn, train_labels_knn, test_data_knn, test_labels_knn)

	rf(X_train, y_train, X_test, y_test)

	lr(args.tune, train_data_std, train_labels_std, test_data_std, test_labels_std)
