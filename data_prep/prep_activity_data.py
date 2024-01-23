
import sqlite3
import pandas as pd
import os
import numpy as np

METADATA_PATH = "/path/to/metadata/"
PATH_ChEMBL   = "/path/to/chembl_33/chembl_33.db"


wells   = pd.read_csv(METADATA_PATH+"metadata/well.csv.gz")

con = sqlite3.connect(PATH_ChEMBL)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
command = """select * from assays"""

assay_df = pd.read_sql_query(command, con)

command = """select * from compound_structures"""

compound_df = pd.read_sql_query(command, con)


chembl_compounds = set(compound_df.standard_inchi_key.unique())
chembl_compounds_InChI = set(compound_df.standard_inchi_key.unique())


compond = pd.read_csv(METADATA_PATH+"metadata/compound.csv.gz")


jump_compounds       = set(compond.Metadata_InChIKey.unique())
jump_compounds_InChI = set(compond.Metadata_InChI.unique())


overlapping_compounds = chembl_compounds.intersection(jump_compounds)

df_overlap_compounds = compound_df[compound_df.standard_inchi_key.isin(overlapping_compounds)]

molregno_overlapping = df_overlap_compounds.molregno.values


# # Check activity data

command = """select * from activities"""

activity_df = pd.read_sql_query(command, con)


activity_df.standard_type.value_counts().head(50)


activity_of_overlapping_compounds = activity_df[activity_df.molregno.isin(molregno_overlapping)]

activity_of_overlapping_compounds.head()

setting = {"standard_type" : "Potency"}


with_known_relation = True # if only using those with defined relation
potency_subset = activity_of_overlapping_compounds[activity_of_overlapping_compounds.standard_type == "Potency"]
potency_subset = potency_subset[potency_subset.activity_comment.isin(['inactive', 'active', 'Active', 'Not Active'])]
compound_counts = potency_subset.assay_id.value_counts()

selected_subset = potency_subset[potency_subset.assay_id.isin(compound_counts[(compound_counts > 100)].index)]

selected_subset.loc[:,"activity_based_on_comment"] = 0

selected_subset.loc[selected_subset.activity_comment.isin(["Active", "active"]), "activity_based_on_comment"] = 1

selected_subset.loc[selected_subset.activity_comment.isin(["Not Active", "inactive"]), "activity_based_on_comment"] = -1

selected_subset.activity_based_on_comment.value_counts()

label_matrix = selected_subset.pivot_table(values="activity_based_on_comment", index="molregno",columns="assay_id", aggfunc=np.median)

known = label_matrix.isna() == 0
label_matrix = label_matrix.fillna(0)

assay_ids = ((label_matrix == 1).sum() > 50)
subset_of_assays = assay_ids[assay_ids].index
subset_with_sufficent_negatives = ((label_matrix[subset_of_assays] == -1.0).sum() > 50)
subset_of_assays_to_keep = subset_with_sufficent_negatives[subset_with_sufficent_negatives].index


label_matrix_subset = label_matrix[subset_of_assays_to_keep]
known = (label_matrix_subset != 0.0).sum(axis=1)


selected_subset = compound_df[compound_df.molregno.isin(label_matrix_subset.index)]

subset_of_compounds_JUMP_id = compond[compond.Metadata_InChIKey.isin(selected_subset.standard_inchi_key)].Metadata_JCP2022.values


well_subset_ = wells[wells.Metadata_JCP2022.isin(subset_of_compounds_JUMP_id)]
subset_of_compouns_in_source_JCP = well_subset_[well_subset_.Metadata_Source == "source_11"].Metadata_JCP2022.values

source_11_molregno = compound_df[compound_df.standard_inchi_key.isin(compond[compond.Metadata_JCP2022.isin(subset_of_compouns_in_source_JCP)].Metadata_InChIKey.values)].molregno.values

positive_ids = ((label_matrix_subset.loc[source_11_molregno] == 1.0).sum() > 50)
negative_ids = ((label_matrix_subset[positive_ids[positive_ids].index].loc[source_11_molregno] == -1.0).sum() > 50)

label_matrix_filtered = label_matrix_subset[negative_ids[negative_ids].index].loc[source_11_molregno]


label_matrix_filtered.reset_index()
label_matrix_renamed = label_matrix_filtered.merge(df_overlap_compounds[["molregno", "standard_inchi_key"]],
                            how="left", left_on="molregno", right_on="molregno",)



wells_11      = wells[wells.Metadata_Source == "source_11"]
wells_11_meta = wells_11.merge(compond, how="left", on="Metadata_JCP2022")
wells_11_meta_overlapping = wells_11_meta[wells_11_meta.Metadata_InChIKey.isin(label_matrix_renamed.standard_inchi_key.unique())]

combined = wells_11_meta_overlapping.merge(label_matrix_renamed, how = "left", right_on = "standard_inchi_key", left_on = "Metadata_InChIKey")

wells_11.loc[:,"sample"] = 0
df_sites = pd.DataFrame([1,2,3,4,5,6,7,8,9], columns=["Metadata_Site"])
df_sites.loc[:,"sample"] = 0
well_11_with_sites = wells_11[["Metadata_Plate", "Metadata_Well", "sample"]].merge(df_sites)

well_11_with_sites["Metadata_Path"] = well_11_with_sites["Metadata_Plate"] + "/" + well_11_with_sites["Metadata_Well"] + "_" + well_11_with_sites["Metadata_Site"].astype(str)  + ".png"

files_df = well_11_with_sites

merged_with_wells = combined.merge(files_df, how="left", on = ["Metadata_Plate", "Metadata_Well"])



merged_with_wells.to_csv("data.csv")

