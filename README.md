# Income-Prediction
Take home assignment for Compellon Data Science Internship Position

## Table of Contents
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Documentation](#documentation)
- [Report](#report)
- [Authors](#authors)
- [Credits](#credits)
- [License](#license)

## Getting Started
### Requirements
Python 2.7, Anaconda, and dependencies in environment.yml.
### Installation
- Clone this repository
	```
	git clone https://github.com/marshalljacobs12/Income-Prediction.git
	```
- Create Anaconda environment from .yml file
	```
	conda env create -f environment.yml
	```
- Activate new environment
	* On Windows
		```
		activate compellon
		```
	* On Mac
		```
		source activate compellon
		```
- You should now be able to execute the scripts.
- To run the jupyter notebooks, run
	```
	jupyter notebook
	``` 

## Usage
- The two scripts are fairly simple to use.
	* To execute preprocess.py with default options, run
		```
		python preprocess.py
		```
	* To execute eval.py with default options, run
		```
		python eval.py
		```
	* More detailed instructions on how to run these scripts can be found in [`DOCUMENTATION.md`](DOCUMENTATION.md) 

- The Jupyter notebooks are also fairly simple. It is important to execute the cells in order or a cell may reference an undefined value.

## Documentation
Detailed explanation of the contents of each file.

## Report
Analysis of preprocessing choices, models, and results.

## Authors
- [Marshall Jacobs](https://www.linkedin.com/in/marshalljacobs/)

## Credits
CITATIONS MISSING
Link to all consulted materials  
[missing data](https://machinelearningmastery.com/handle-missing-data-python/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details
