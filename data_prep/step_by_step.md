# Step by step download guide to download, pre-processing and combining data for JUMP-CP and ChEMBL data

Make sure to have Python 3.7 or above available


# Image Download Instructions

## Step 1:
### Define a base folder:

Base_Folder = ```/path/to/folder/```

## Step 2:
### Download metadata

Add a a metadata folder to the based folder.

Metadata_Folder = ```/path/to/folder/metadata/```

Download image metadata from:

https://github.com/jump-cellpainting/datasets/tree/main/metadata

Download the well.csv.gz , plate.csv.gz and compound.csv.gz files. Store these in the metadata folder.


## Step 3:
### Install DeepProfiler 

Install DeepProfiler using the instructions available at:

https://cytomining.github.io/DeepProfiler-handbook/docs/01-install.html

Clone the repository in the Base_Folder defined above, install the venv and activate the environment following the instructions in the DeepProfiler handbook.
```
virtualenv -p python3 deepprofenv

source ./deepprofenv/bin/activate

git clone https://github.com/broadinstitute/DeepProfiler.git

cd DeepProfiler/

pip install -e .

pip install awscli
pip install rdkit-pypi
```



## Step 4:
### Download and process image data

Edit the base path in the download_and_process.py file and set it to the Base_Folder defined above.

Then run ```python download_and_process.py```

This requires several TB of storage space and can take several days.




# Activity data

The activity data can be downloaded from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/


## Step 1: 
Download the chembl_33_sqlite.tar.gz from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ and add it to the Base_Folder.

Untar it using: ```tar -xvzf chembl_33_sqlite.tar.gz```

## Step 2: 

Edit the base path in prep_activity_data.py and set it to the Base_Path defined above.

Then run it using ```python prep_activity_data.py```

## Step 3:

Split the acitivity data using ```python data_splitting.py```

# Combine image and activity data

### Merge image data with metadata and activity data

Run python remove_missing_files.py to remove any image files that were not downloaded correctly and create the final dataset file.


