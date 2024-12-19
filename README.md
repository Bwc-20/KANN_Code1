# About The Project
KANN: Knowledge-embedded asynchronous deep p-neural network for prostate cancer metastasis prediction

Our KANN is validated on two datasets, NPCP and cBioPortal, respectively. The details of these datasets as well as the codes and validation results for KANN are stored in the above folders NPCP and cBioPortal, respectively. The code files for the implementation of KANN in the above folders are the same, but the data files in these folders are different.

# Getting Started
To get a local copy up and running, follow these simple steps

### Prerequisites
* Python 3.6.0, check environments.yml for list of needed packages

### Installation
* Clone the repo

   `git clone https://github.com/Bwc-20/KANN_Code1.git`
* Create conda environment

   `conda env create --name KANN_env --file=environment.yml`
 
# Usage
We perform the same operations for the implementation and validation of KANN on the NPCP and cBioPortal datasets, respectively. We take the NPCP dataset as an example to introduce the running of KANN.

* Activate the created conda environment (Please create your project environment based on environment.yml)

   `source activate KANN_env`
* To re-train a model from scratch run

   `cd ./NPCP/train`
 
    `python run_me.py`
  
  This will run a five-fold cross-validation experiment for KANN and baseline methods.  The experiment results of KANN will be stored in './NPCP/_logs/p1000/pnet
/crossvalidation_average_reg_10_tanh/AllActModelResult.csv'.  
In AllActModelResult.csv, the row with the eleNum as 'Avg25' is the average of the five-fold cross-validation results for KANN.
The experiment results of baseline methods will be stored in './NPCP/_logs/p1000/pnet
/crossvalidation_average_reg_10_tanh/fold.csv'

# Contact

If you have any problems, please feel free to give me feedback. Please communicate with me at 22B903007@stu.hit.edu.cn
