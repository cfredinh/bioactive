# Information regarding pre-processing and combining the data

## Image data
The Image data used from source 11 of the JUMP-CP consortsium
Downloadable from https://registry.opendata.aws/cellpainting-gallery/ .
The metadata regarding compounds can be found at: https://github.com/jump-cellpainting/datasets .

Once downloaded image, image normalization was carried out to reduce illumination effects and intensity outliers.
See [^1] for easy to use open source pre-processing functionality.

Following normalization, images coming from the same site were also combined into one png file to speedup data loading, see combining_images.py.

## Activity data

The activity data can be downloaded from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

## Combinding the image and activity data

The overlap between the JUMP-CP and ChEMBL data and the activity data was done in: "Fetch Activity data from ChEMBL.ipynb"

Data splitting done in "Data_splitting.ipynb"

Finally, overlap between each well and the downloaded wells was checked and any missing images were removed from the data files, done in "remove_missing_files.py"








[^1]: https://cytomining.github.io/DeepProfiler-handbook/docs/00-welcome.html


