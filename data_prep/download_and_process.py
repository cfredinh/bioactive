import subprocess
import pandas as pd
from PIL import Image
import numpy as np
import json
from pathlib import Path
import os
import tifffile

import multiprocessing
from joblib import Parallel, delayed


# Basepath

BASE_PATH     = "/path/to/folder/"


METADATA_PATH     = BASE_PATH + "metadata/"
DEEPPROFILER_PATH = BASE_PATH + "DeepProfiler/"

# Define how many plates to handle concurently
n_jobs = 1 


######################################################
######################################################
######################################################



def get_plate_metadata(s_, b_, p_, folder, folder_ld):
    '''    Download and unzip the metadata for plate p_    '''
    
    load_metadata_ = (
    "s3://cellpainting-gallery/cpg0016-jump/"
    "{Metadata_Source}/workspace/load_data_csv/"
    "{Metadata_Batch}/{Metadata_Plate}/load_data.csv.gz")
    
    aws_path_load_data = load_metadata_.format(Metadata_Source = s_, Metadata_Batch = b_, Metadata_Plate = p_)
    
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", aws_path_load_data, folder])

    subprocess.run(["gzip", "-d", folder_ld])

######################################################

def get_image_paths(s_, b_, p_):
    '''    Check for the correct path for images to download    '''
        
    load_image_folders = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/images/"
        "{Metadata_Batch}/images/"
    )
    
    image_data_path = (load_image_folders.format(Metadata_Source = s_, Metadata_Batch = b_, Metadata_Plate = p_))

    result = subprocess.run(['aws', 's3', 'ls', '--no-sign-request', image_data_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
    
    plate_path = None
    for row in result.split("\n"):
        print(row)
        
        if p_ in row.split(" ")[-1]:
            plate_path = row.split(" ")[-1]
            break
        
    if plate_path is None:
        return "ERROR"
    

    if " " in plate_path:
        plate_path = plate_path.replace(" ", "\ ")

    
    load_image_path = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/images/"
        "{Metadata_Batch}/images/{Metadata_Plate_path}Images/"
    )
    aws_path_image_data = load_image_path.format(Metadata_Source = s_, Metadata_Batch = b_, Metadata_Plate_path = plate_path)

    return aws_path_image_data


######################################################

def get_images(s_, b_, p_, folder_imgs, get_subset=None):
    '''    Get the paths to the plate p_ images, check the plates path are correct and then download the images'''
    
    aws_path_image_data = get_image_paths(s_, b_, p_)
    
    
    if aws_path_image_data == "ERROR":
        return "ERROR"
    
    # Check if this is the correct path 
    cmd = "aws s3 ls --no-sign-request {aws_path_image_data} --recursive | wc -l".format(aws_path_image_data = aws_path_image_data)

    count = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


    aws_path_image_data = aws_path_image_data.replace("ment\\","ment")
    
    if get_subset is None:
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--recursive", aws_path_image_data, folder_imgs])
    else:
        
        for missin_p in get_subset:
            print(" Getting missing image: ", missin_p)
            subprocess.run(["aws", "s3", "cp", "--no-sign-request", aws_path_image_data+missin_p, folder_imgs])

######################################################

def get_image_info(path_ld, folder_imgs, folder_proj_im, path_index, path_config):
    df = pd.read_csv(path_ld)
    example_image = df.FileName_OrigAGP[0]

    if os.path.exists(folder_imgs + example_image):
        im = Image.open(folder_imgs + example_image)
    elif os.path.exists(folder_proj_im + "/images/" + example_image):
        im = Image.open(folder_proj_im + "/images/" + example_image)
    else:
        for i_i in range(df.shape[0]):
            example_image = df.FileName_OrigAGP[i_i]
            if os.path.exists(folder_imgs + example_image):
                im = Image.open(folder_imgs + example_image)
                continue
            
    
    im_array = np.array(im)

    # Open config file and edit it
    config_path = BASE_PATH + "config.json"
   
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    data["dataset"]["images"]["width"]  = im_array.shape[0]
    data["dataset"]["images"]["height"] = im_array.shape[1]
    data["dataset"]["images"]["file_format"] = example_image.split(".")[-1]

    # Write config to file
    with open(path_config, 'w') as f:
        json.dump(data, f, indent=4)
    print(data)
    
    # Prep metadata file for downscaling

    for p in ["FileName_OrigAGP","FileName_OrigDNA","FileName_OrigER","FileName_OrigMito","FileName_OrigRNA"]:
        df["path_"+p] = df[p].astype(str)

    df.to_csv(path_index)

    return df

def check_missing_images(df, folder_proj_im, folder_imgs, ):
    
    
    # Check for missing data
    
    missing_images = []
    for col in ["path_FileName_OrigAGP","path_FileName_OrigDNA","path_FileName_OrigER","path_FileName_OrigMito","path_FileName_OrigRNA"]:
        for p_i in df[col].values:
            
            my_file     = Path(folder_proj_im + "/images/" +p_i)
            my_file_org = Path(folder_imgs + p_i)
            
            
            if my_file.is_file() and (os.path.getsize(my_file) > 10**6):
                
                try:
                    with tifffile.TiffFile(my_file) as tif:
                        
                        continue
                except:
                    missing_images.append(p_i)
                    continue
                    
            elif my_file_org.is_file() and (os.path.getsize(my_file_org) > 10**6):
                try:
                    with tifffile.TiffFile(my_file_org) as tif:
                        continue
                except:
                    missing_images.append(p_i)
                    continue
            else:
                missing_images.append(p_i)
        print("check")

    return missing_images

######################################################

def combined_images_to_one_file(plate_):
    
    plate  = pd.read_csv(METADATA_PATH+"plate.csv.gz")
    well   = pd.read_csv(METADATA_PATH+"well.csv.gz")

    
    path  = BASE_PATH + plate_ + "/" + "proj/outputs/compressed/images/combined/" + plate_ + "/"
    
    plate_specific = well[well.Metadata_Plate == plate_]
    
    df = pd.read_csv(BASE_PATH + str(plate_) + "/load_data.csv") # Get metadata for all images of plate

    df = df.astype(str)

    print("Working on plate: ", plate_)
    comb = df.merge(plate_specific, how="left", on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
    

    path  = BASE_PATH + plate_ + "/proj/outputs/compressed/images/combined/" + plate_ + "/"

    os.makedirs(path, exist_ok=True)

    for i in range(comb.shape[0]): # Loop over all the images and combined each site into a single image and store in the new image folder
        row =  comb.iloc[i]
        image_ = []
        for col in ['FileName_OrigDNA', 'FileName_OrigAGP', 'FileName_OrigER', 'FileName_OrigMito', 'FileName_OrigRNA']:
            path  = BASE_PATH + row["Metadata_Plate"] + "/proj/outputs/compressed/images/" + row["Metadata_Plate"] + "/"
            path += row[col][:-4] if row[col][-4:] == ".tif" else row[col][:-5]
            path += ".png"

            im_frame = Image.open(path)
            image_.append(np.array(im_frame))
            
        img_ = np.hstack(image_)
        img = Image.fromarray(img_)
        


        path  = BASE_PATH + plate_ + "/proj/outputs/compressed/images/combined/" + plate_ + "/"
        path += row["Metadata_Well"]  + "_"
        path += row["Metadata_Site"]  + ".png"

        img.save(path)

        if i % 100 == 0:
            print(plate_, i/comb.shape[0])
        
       






############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################



def get_and_prep_plate(plate_):
    
    folder         = BASE_PATH+plate_
    path_ld_gz     = BASE_PATH+plate_+"/load_data.csv.gz"
    path_ld        = BASE_PATH+plate_+"/load_data.csv"
    folder_imgs    = BASE_PATH+plate_+"/images/"
    folder_proj    = BASE_PATH+plate_+"/proj"
    folder_proj_im = BASE_PATH+plate_+"/proj/inputs/"
    path_config    = BASE_PATH+plate_+"/proj/inputs/config/config.json"
    path_index     = BASE_PATH+plate_+"/proj/inputs/metadata/index.csv"
    print(folder)


    # Check if folder has been downloaded
    if os.path.isdir(folder):
        print("PLATE: ",folder ," Allready done") 
        return

    # Create necessary folders
    if not os.path.isdir(folder):
        subprocess.run(["mkdir", folder])
        subprocess.run(["mkdir", folder_imgs])
        subprocess.run(["mkdir", folder_proj])

    plates    = pd.read_csv(METADATA_PATH+"plate.csv.gz")

    # Get plate info
    s_, b_, p_ , _ = plates[plates.Metadata_Plate == plate_].values[0]

    # Get metadata for plate, including image_paths
    get_plate_metadata(s_, b_, p_, folder, path_ld_gz)
        

    # Download the tif/tiff images
    get_images(s_, b_, p_, folder_imgs)



    # Setup folder for image processing
    cm_ = "python3 deepprofiler --root {folder_proj} setup".format(folder_proj=folder_proj)
    subprocess.run(cm_, shell=True, cwd=DEEPPROFILER_PATH)


    
    # Load example image to get file type and image size
    df = get_image_info(path_ld, folder_imgs, folder_proj_im, path_index, path_config)

    # Check the missing images and get list of all to re-download
    missing_images = check_missing_images(df, folder_proj_im, folder_imgs)
    
    print(missing_images)

    # If missing images, try downloading them again

    if len(missing_images) > 0:
        print("Getting Missing Images")
        get_images(s_, b_, p_, folder_imgs, missing_images)

    # Move downloaded data to pre-processing folder
    subprocess.run(["cp", "-r", folder_imgs, folder_proj_im])
    
    # Remove the original images
    subprocess.run(["rm", "-r", folder_imgs])

    
    
    # Process images
    cm_ = "python3 deepprofiler --root={folder_proj} --config=config.json prepare".format(folder_proj=folder_proj)
    
    try:
        subprocess.run(cm_, shell=True, cwd=DEEPPROFILER_PATH)
    except:
        print("Something wrong with the processing step")
        return 0
    
    print("Done processing the images")
    
    combined_images_to_one_file(plate_)

    # Delete the input folder
    subprocess.run(["rm", "-r", folder_proj_im])


############################################################
############################################################
############################################################
############################################################
############################################################

# Run data download and processing plate wise below:

# Read plate metadata
df  = pd.read_csv(METADATA_PATH+"plate.csv.gz")

# Only download data from source 11 and Compound plates
df  = df[(df.Metadata_Source.isin(["source_11"]) & (df.Metadata_PlateType == "COMPOUND"))]
df  = df.sort_values(by=['Metadata_Plate'])

print("Number of Plates: ", df.shape)

plates = list(df.Metadata_Plate.values)


print('Downloading starts . . .')
Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(get_and_prep_plate)(
        plates[i]) for i in range(len(plates)) )