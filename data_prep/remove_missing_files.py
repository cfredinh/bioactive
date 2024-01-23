import pandas as pd
import os 

df_path = "/path/to/split_data.csv"

df = pd.read_csv(df_path, index_col=0)

base_path = "/path/to/images/"

missing_indicies = []
for i, row in df.iterrows():
    
    specific_path = row.Metadata_Path
    
    if os.path.isfile(base_path + str(specific_path)):
        continue
    else:
        print("Missing", specific_path)
        missing_indicies.append(i)

df = df.drop(missing_indicies)

path_new_csv = "/path/to/training.csv" # the new csv used for training only containing data points where the images are known to have been downloaded and pre_processed

df.to_csv(path_new_csv)