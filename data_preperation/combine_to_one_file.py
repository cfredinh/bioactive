import pandas as pd
import numpy as np
import os

from PIL import Image

import multiprocessing
from joblib import Parallel, delayed
import subprocess

def combined_images_in_plate(plate_):
    
    base_path          = "/path/to/imagefolder/"
    base_path_combined = "/path/to/new/imagefolder/"
    base_path_meta     = "/path/to/image_metadata/"
    METADATA_PATH      = "/path/to/metadata"

    plate  = pd.read_csv(METADATA_PATH+"metadata/plate.csv.gz")
    well   = pd.read_csv(METADATA_PATH+"metadata/well.csv.gz")

    
    path  = base_path
    path += plate_ + "/"
    
    plate_specific = well[well.Metadata_Plate == plate_]
    
    df = pd.read_csv(base_path_meta + str(plate_) + "/load_data.csv") # Get metadata for all images of plate

    df = df.astype(str)

    print("Working on plate: ", plate_)
    comb = df.merge(plate_specific, how="left", on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
    
    for i in range(comb.shape[0]): # Loop over all the images and combined each site into a single image and store in the new image folder
        row =  comb.iloc[i]
        image_ = []
        for col in ['FileName_OrigDNA', 'FileName_OrigAGP', 'FileName_OrigER', 'FileName_OrigMito', 'FileName_OrigRNA']:
            path  = base_path
            path += row["Metadata_Plate"] + "/"
            path += row[col][:-4] if row[col][-4:] == ".tif" else row[col][:-5]
            path += ".png"

            im_frame = Image.open(path)
            image_.append(np.array(im_frame))
            
        img_ = np.hstack(image_)
        img = Image.fromarray(img_)
        
        path  = base_path_combined
        path += row["Metadata_Plate"] + "/"
        
        os.makedirs(path, exist_ok=True)
        
        path += row["Metadata_Well"]  + "_"
        path += row["Metadata_Site"]  + ".png"

        img.save(path)

        if i % 100 == 0:
            print(plate_, i/comb.shape[0])
        
       



dirs = ['EC000053']


print('Combined seperate images . . .')
Parallel(n_jobs=1, verbose=1)(
    delayed(combined_images_in_plate)(
        dirs[i]) for i in range(len(dirs)) )
    



