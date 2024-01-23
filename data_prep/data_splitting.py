import pandas as pd
import numpy as np
import random as rd
import torch
from collections import defaultdict

import rdkit

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import Chem
from rdkit.Chem import AllChem

def ClusterFps(fps,cutoff=0.2):
 
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
 
    
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs

def split_data_cross_validation_structure_clustering(clusters, important_data, splits = [0,0.1667*1,0.1667*2,0.1667*3,0.1667*4,0.1667*5], split_on='compound', split_number=1):
    data = pd.read_csv("data.csv",index_col=0)
    unique = data.drop_duplicates(subset="Metadata_InChI")
    
    unique_cmp = list(unique.Metadata_InChI.unique())
    rd.shuffle(clusters)
 
    data_splits = []
 
    cluster_index = 0
    for i in range(len(splits)):
        split_data = []

        # While the current split doesn't contain more compounds than allocated and there are still clusters left do:
        while len(split_data) <= int(splits[1] * len(unique_cmp)) and cluster_index < len(clusters):
            
            split_data.extend(important_data.iloc[list(clusters[cluster_index])]['Metadata_JCP2022'].values)
            cluster_index += 1

        data_splits.append(split_data)
        print(cluster_index)

    return data_splits
    

data = pd.read_csv("data.csv",index_col=0)

scaffolds      = defaultdict(list)
scaffolds_mols = defaultdict(list)
unique = data.drop_duplicates(subset="Metadata_InChI")

for i, row in enumerate(range(len(unique))):
    val   = unique.iloc[row]
    m     = Chem.MolFromInchi(val.Metadata_InChI)
    scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    scaffolds[scaff].append(val.Metadata_JCP2022)
    scaffolds_mols[scaff].append(m)


ms  = [Chem.MolFromInchi(unique.iloc[i].Metadata_InChI) for i in range(len(unique))]

fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024,) for x in ms]

clusters = ClusterFps(fps,cutoff=0.7)

clu_list = list(clusters)

splits = split_data_cross_validation_structure_clustering(clu_list, unique)

data["split_number"] = -1
print(len(splits))
for i in range(len(splits)):
    data.loc[data.Metadata_JCP2022.isin(splits[i]),"split_number"] = i

data.to_csv("split_data.csv")