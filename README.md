# Bioactivity Prediction

Code for training the ResNet50 and MLP models used in the paper: **Cell Painting-based bioactivity prediction boosts high-throughput screening hit-rates and compound diversity**



Code for ResNet50 and MLP model training
    
    Training and testing with ResNet50 model can be done "python classification.py --params_path params/params.json"  
    Specify, parameters in params/params.json for data location and training settings
    
    Training and testing with MLP models can be done using 
    "python MLP_predictor.py" 
    Specifying parameters for data location and training setings can be done in the MLP_predictor.py file.
    
    The training environment can be setup using the provided dockerfile in the docker_image folder.

JUMP-CP & ChEMBL dataset preperation
    
    The Image data can be downloaded from the JUMP-CP data repository, under dataset name CPG0016. Available from the Cell Painting Gallery on the Registry of Open Data on AWS (https://registry.opendata.aws/cellpainting-gallery/). Metadata can be downloaded at: https://github.com/jump-cellpainting/datasets

    Similarly the activity downloaded from ChEMBL at https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    See data_prep folder for information regarding pre-processing and combining the data

