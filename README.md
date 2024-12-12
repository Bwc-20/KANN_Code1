# About The Project
KANN: Knowledge-embedded asynchronous deep p-neural network for prostate cancer metastasis prediction

There are two implementations of our KANN, including KANN (NPCP) and KANN (cBioPortal), corresponding to the two datasets used in this study, NPCP and cBioPortal, respectively.  Both implementations differ only in the gene layer, others are the same. The codes for both implementations and their used datasets are stored in the above folders NPCP and cBioPortal, respectively.

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
Both implementations of KANN, KANN (NPCP) and KANN (cBioPortal), perform the same operations at runtime. We take the KANN (NPCP) as an example to introduce the running of KANN.

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
