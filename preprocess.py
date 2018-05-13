import argparse
import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from util import encode_categorical_data
from util import native, primary

# CITE: https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
def CheckExt(choices):
    class Act(argparse.Action):
        def __call__(self,parser,namespace,fname,option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("file doesn't end with one of {} {}".format(choices,option_string))
            else:
                setattr(namespace,self.dest,fname)

    return Act

def create_datasets(filename):
	# Read in raw data set
	df = pd.read_csv(filename)

	# Replace missing fields with NaN
	df.replace(' ?', np.nan, inplace=True)
	
	# drop observations with missing values
	df = df.dropna()

	# Trim whitespace
	for c in df.select_dtypes(include=[object]):
		df[c] = df[c].str.strip()

	# Target variable Income = 1 if observation has Income > 50k, 0 if Income <= 50k
	df['Income'] = df['Income'].apply(lambda x: 1 if x =='>50K' else 0)

	# Shuffle examples
	df = df.reindex(np.random.permutation(df.index))

	dataset_bin = pd.DataFrame() 
	dataset_con = pd.DataFrame()

	dataset_bin['Income'] = df['Income']
	dataset_con['Income'] = df['Income']

	dataset_bin['age'] = pd.cut(df['age'], 10)
	dataset_con['age'] = df['age']

	dataset_bin['fnlwgt'] = pd.cut(df['fnlwgt'], 10)
	dataset_con['fnlwgt'] = df['fnlwgt']

	dataset_bin['education-num'] = pd.cut(df['education-num'], 10)
	dataset_con['education-num'] = df['education-num']

	dataset_bin['capital-gain'] = pd.cut(df['capital-gain'], 5)
	dataset_con['capital-gain'] = df['capital-gain']

	dataset_bin['capital-loss'] = pd.cut(df['capital-loss'], 5)
	dataset_con['capital-loss'] = df['capital-loss']	

	dataset_bin['hours-per-week'] = pd.cut(df['hours-per-week'], 10)
	dataset_con['hours-per-week'] = df['hours-per-week']	

	df['age-hours'] = df['age'] * df['hours-per-week']
	dataset_bin['age-hours'] = pd.cut(df['age-hours'], 10)
	dataset_con['age-hours'] = df['age-hours']

	# Convert 'sex' to a binary feature
	df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
	dataset_bin['sex'] = df['sex']
	dataset_con['sex'] = df['sex']

	df['native-country'] = df['native-country'].apply(native)
	dataset_bin['native-country'] = df['native-country']
	dataset_con['native-country'] = df['native-country']

	df['education'] = df['education'].apply(primary)
	dataset_bin['education'] = df['education']
	dataset_con['education'] = df['education']

	df['marital-status'].replace('Married-AF-spouse', 'Married-civ-spouse', inplace=True)
	dataset_bin['marital-status'] = df['marital-status']
	dataset_con['marital-status'] = df['marital-status']

	df['occupation'].replace('Armed-Forces', 'Machine-op-inspct', inplace=True)
	dataset_bin['occupation'] = df['occupation']
	dataset_con['occupation'] = df['occupation']

	df['workclass'].replace('Without-pay', 'Self-emp-inc', inplace=True)
	dataset_bin['workclass'] = df['workclass']
	dataset_con['workclass'] = df['workclass']

	dataset_bin['relationship'] = df['relationship']
	dataset_con['relationship'] = df['relationship']

	dataset_bin['race'] = df['race']
	dataset_con['race'] = df['race']

	for column in dataset_bin:
		if dataset_bin[column].dtype.name == 'category':
			dataset_bin[column] = LabelEncoder().fit_transform(dataset_bin[column])

	dataset_bin = encode_categorical_data(dataset_bin)
	dataset_con = encode_categorical_data(dataset_con)

	dataset_con.to_csv('data/dataset_con.csv', index=False)
	dataset_bin.to_csv('data/dataset_bin.csv', index=False)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='preprocess.py takes a data set as input, cleans the data set \
												 and writes two different versions of the cleaned data to disk \
												 The raw data set, as well as the two resulting data sets, \
												 have .csv extensions')

	parser.add_argument('-i', '--input', type=str, default='data/Demog_DS_TRAIN_15K.csv', 
						action=CheckExt('csv'), help='INPUT should be the relative path to the CSV file with raw data set')

	try:
		args = parser.parse_args()
	except IOError as msg:
		parser.error(str(msg))

	create_datasets(args.input)

	print '-----------------DONE-----------------'