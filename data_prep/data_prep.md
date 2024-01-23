# Information regarding download of Hofmarcher data


### To download the Hofmarcher data
Download the individual zip files from: https://ml.jku.at/software/cellpainting/dataset/
Label data etc. can be found at: https://github.com/ml-jku/hti-cnn/tree/master/labels


# Information regarding download, pre-processing and combining data for JUMP-CP and ChEMBL data

## Downloading and pre-paring all data

See step_by_step.md for detailed download and pre-processing steps.

## Image data
The Image data used from source 11 of the JUMP-CP consortium.
Downloadable from https://registry.opendata.aws/cellpainting-gallery/ .
The metadata regarding compounds can be found at: https://github.com/jump-cellpainting/datasets .

Once downloaded, image normalization was carried out to reduce illumination effects and intensity outliers.
See [^1] for easy to use open source pre-processing functionality.

Following normalization, images coming from the same site were also combined into one png file to speedup data loading.

## Activity data

The activity data can be downloaded from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

## Combinding the image and activity data

The overlap between the JUMP-CP and ChEMBL data and the activity data was done in: "prep_activity_data.py"

Data splitting done in "data_splitting.py"

Finally, overlap between each well and the downloaded wells was checked and any missing images were removed from the data files, done in "remove_missing_files.py"




[^1]: https://cytomining.github.io/DeepProfiler-handbook/docs/00-welcome.html


