# Documentation
Should the directory structure be in alphabetical order?

```bash
.
├── README.md
├── preprocess.py
├── eval.py
├── util.py
├── data.ipynb
├── main.ipynb
├── data
│   ├── Demog_DS_TRAIN_15K.csv
│   ├── dataset_con.csv
│   └── dataset_bin.csv
├── DOCUMENTATION.md
├── REPORT.md
├── LICENSE.md
├── environment.yml
└── .gitignore
```

## [`README.md`](README.md)

Overview of repository contents as well as instructions for setting up and executing included Python scripts and Jupyter notebooks. Also contains links to the project's documentation and report, consulted references, and license.

## [`preprocess.py`](preprocess.py)
Python script that performs all preprocessing of `Demog_DS_TRAIN_15K.csv` and writes two different data sets to disk that can be used during model evaluation.

Options \
-h --help \
-i --input

Usage
- To run this script using the default csv file `Demog_DS_TRAIN_15K.csv`, (located in the data directory), simply run
	```python
	python preprocess.py
	```
- To run this script using a different csv file, run
	```python
	python preprocess.py -i path_to_file/dataset.csv
	```

## [`eval.py`](eval.py)
Python script that trains and evaluates a ML model. It takes a training and test file as input.

Options  
-h --help   
-t --tune  
-c --con  

Usage
- To run this script using `dataset_bin.csv` and the hyperparameter values I chose, simply run
	```
	python eval.py
	```
- To run this script using `dataset_bin.csv` and choose hyperparameters from tuning during the script's execution, run
	```
	python eval.py -t
	```
- To run this script using `dataset_con.csv` and the hyperparameter values I chose, simply run
	```
	python eval.py -c
	```
- To run this script using `dataset_con.csv` and choose hyperparameters from tuning during the script's execution, run
	```
	python eval.py -t -c
	```

Additional Notes
- The default hyperparameter values I chose were from tuning using `dataset_bin.csv` so they won't necessarily be good hyperparameter values for `dataset_con.csv`.
- Because tuning the hyperparameters in the random forest classifier is more involved and compute intensive, using the `-t` flag will not tune these hyperparameter's during the script's execution. If you wish to tinker with these hyperparameters, do so in the Jupyter notebook.

## [`util.py`](util.py)
Utility functions for preprocessing and data visualization.

## [`data.ipynb`](data.ipynb)
Visualization and basic analysis of initial data set.

## [`main.ipynb`](main.ipynb)
Analysis of cleaned data set, feature selection, and model evaluation.

## [`data`](#data)
### [`data/Demog_DS_TRAIN_15K.csv`](data/Demog_DS_TRAIN_15K.csv)
Given data set.
### [`data/dataset_con.csv`](data/dataset_con.csv)
Data set written to disk after preprocessing which preserves continuous values.
### [`data/dataset_bin.csv`](data/dataset_bin.csv)
Data set written to disk after preprocessing which uses binning to group continuous values.

## [`DOCUMENTATION.md`](DOCUMENTATION.md)
Brief description of each file contained in repository as well as description of directory structure

## [`REPORT.md`](REPORT.md)
Analysis of problem to be solved and data set, as well as approach, data preprocessing, model selection, evaluation metrics, and tradeoffs/comparisons between models.

## [`LICENSE.md`](LICENSE.md)
Standard MIT License for source code in this repository.  

## [`environment.yml`](environment.yml)
YAML file with full description of the conda env needed to run Python scripts and Jupyter notebooks contained in this repository. Disclaimer: I created this .yml file from my base conda environment, so it has dependencies not used in the project.

## [`.gitignore`](.gitignore)
Basic .gitignore that ignores .dSYM and .pyc files.
