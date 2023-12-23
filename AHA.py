import os
import pydicom
import numpy as np
import glob
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import multipreprocess
import torch

print("Last Executed: " + str(datetime.now()))
DIMS = 256
WORKDIR = os.getcwd()
DATADIR = os.path.join("/home", "ddavilag", "cardio", "data")
IMGSAVEDIR = os.path.join(DATADIR, "pkl_files", "256_final")
CSVSAVEDIR = os.path.join(DATADIR, "csv_files", "preprocessed")

dicom_files = glob.glob(
    os.path.join(
        "/home/ddavilag/cardio/data/dicom_files/CXRdl_2-28-23_unzip", "**", "*.dcm"
    ),
    recursive=True,
)
print("Example DICOM file paths:", dicom_files[:5])
print("# of DICOM files:", len(dicom_files))

pid = 7
pids = [dicom_file.split("/")[pid] for dicom_file in dicom_files]
phonetic_ids = {dicom_file.split("/")[pid] for dicom_file in dicom_files}
print(len(phonetic_ids))
dicom_files1 =(
    glob.glob(
        os.path.join(
            "/home/ddavilag/cardio/data/dicom_files/CXR_10-10-23_unzip", "**", "*.dcm"
        ),
        recursive=True,
    )
)
print("Example DICOM file paths:", dicom_files1[:5])
print("# of DICOM files:", len(dicom_files1))
pid = 8
pids2 = [dicom_file.split("/")[pid] for dicom_file in dicom_files1]
phonetic_ids1 = {dicom_file.split("/")[pid] for dicom_file in dicom_files1}
print(len(phonetic_ids1))
phonetic_ids = phonetic_ids.union(phonetic_ids1)
dicom_files.extend(dicom_files1)
print(len(dicom_files))
pids = pids + pids2

# get a column indicating which dataset the image is from
dataset = []
for dicom_file in dicom_files:
    if "CXR_10-10" in dicom_file:
        dataset.append(2)
    else:
        dataset.append(1)
print(len(dataset), len(dicom_files), len(pids))
dicom_df = pd.DataFrame({'Phonetic ID':pids, "Path": dicom_files, "Dataset": dataset})


print("# of IDs:", len(phonetic_ids))
for phonetic_id in phonetic_ids:
    if os.path.exists(os.path.join(IMGSAVEDIR, phonetic_id)):
        print("Folders already exist")
        break
    else:
        os.mkdir(os.path.join(IMGSAVEDIR, phonetic_id))

# Get Samira's Frontal/Lateral Output
frontLat1 = pd.read_csv(
    os.path.join(DATADIR, "csv_files", "samira_front_lat_output", "FrontalLateral1.csv")
)
frontLat2 = pd.read_csv(
    os.path.join(DATADIR, "csv_files", "samira_front_lat_output", "FrontalLateral2.csv")
)
frontLat = pd.concat([frontLat1, frontLat2])
frontLat.columns = [
    "Phonetic ID",
    "Image ID",
    "Image View",
    "Coverage",
    "Process",
    "AP/PA",
]

front_chest = frontLat[
    (frontLat["Image View"] == "Frontal") & (frontLat["Coverage"] == "Chest")
]
front_chest.sort_values(by=["Phonetic ID", "Image ID"], inplace=True)
front_chest.reset_index(drop=True, inplace=True)
front_chest["Image ID"] = front_chest["Image ID"].str.replace(".dcm", "")
display(front_chest)
front_chest.to_csv(
    os.path.join(DATADIR, "csv_files", "samira_front_lat_output", "front_chest.csv"),
    index=False,
)

image_dict = defaultdict(list)
for _, row in tqdm(front_chest.iterrows(), total=front_chest.shape[0]):
    phonetic_id = row["Phonetic ID"]
    image_id = row["Image ID"]
    if len(image_dict[phonetic_id]) == 0:
        image_dict[phonetic_id] = [image_id]
    else:
        image_dict[phonetic_id].append(image_id)


# Run multipreprocess.py
dicom_df.to_csv(os.path.join(CSVSAVEDIR, "dicom_df.csv"), index=False)
multipreprocess.main()

# Rash's cleaned IDs
cleaned_ids = []
with open(os.path.join(DATADIR, "txt_files", "cleaned_IDs.txt"), "r") as f:
    cleaned_ids.extend(f.read().splitlines())
print(len(cleaned_ids))

# Finding paths of frontal chest numpy images
pickled_imgs = glob.glob(os.path.join(IMGSAVEDIR, "**", "*.pt"), recursive=True)
print(len(pickled_imgs))
#visualize 10 random images
for i in range(10):
    img = torch.load(pickled_imgs[i])
    plt.imshow(img[0], cmap="gray")
    plt.show()

valid_dict = {
    "Phonetic ID": [],
    "Path": [],
    "Image ID": [],
    "Image View": [],
    "Coverage": [],
    "Process": [],
    "AP/PA": [],
}

# Creating a list of valid frontal chest image paths (torch arrays)
for img in tqdm(pickled_imgs):
    split = img.split("/")
    phonetic_id = split[7]

    # Checking if the image is in the cleaned IDs from Rash
    if phonetic_id not in image_dict.keys():
        continue
    if phonetic_id not in cleaned_ids:
        continue
    # Checking if the image is in the frontal chest csv file
    if row_exists := (
        front_chest.loc[
            (front_chest["Phonetic ID"] == phonetic_id)
            & (front_chest["Image ID"] == split[-1].replace(".pt", ""))
        ]
        .any()
        .any()
    ):
        row = front_chest.loc[
            (front_chest["Phonetic ID"] == phonetic_id)
            & (front_chest["Image ID"] == split[-1].replace(".pt", ""))
        ].values[0]
        valid_dict["Phonetic ID"].append(phonetic_id)
        valid_dict["Path"].append(img)
        valid_dict["Image ID"].append(row[1])
        valid_dict["Image View"].append(row[2])
        valid_dict["Coverage"].append(row[3])
        valid_dict["Process"].append(row[4])
        valid_dict["AP/PA"].append(row[5])
valid_front_chest = pd.DataFrame(valid_dict)
display(valid_front_chest)


gb_first = valid_front_chest.groupby("Phonetic ID").first()
gb_first.reset_index(inplace=True)
display(gb_first)



# getting the echo reports
chamber_df = pd.read_csv(os.path.join(DATADIR, "csv_files",'preprocessed', "AllLabels12-2-23.csv"))
#chamber_df = chamber_df.dropna()
chamber_df = chamber_df.groupby("Phonetic ID").first().reset_index(drop=True)
chamber_df.drop(columns=["Unnamed: 0"], inplace=True)
display(chamber_df)

chamber_df = pd.merge(chamber_df, gb_first, on="Phonetic ID", how="right")
display(chamber_df['RV'].value_counts(dropna=False))

def replaceLabels(col):
    if col == "LV":
        replace_dict = {
            "normal": "normal",
            "mildly increased": "mildly enlarged",
            "moderately increased": "moderately enlarged",
            "severely increased": "severely enlarged",
            "small": "normal",
            "moderately increased by linear measurement but normal by volumes": "moderately enlarged",
            "dilated": "severely enlarged",
        }
    elif col == "RV":
        replace_dict = {
            "normal": "normal",
            "mildly increased": "mildly enlarged",
            "moderately increased": "moderately enlarged",
            "severely increased": "severely enlarged",
            "small": "normal",
            "borderline enlarged": "mildly enlarged",
            "mild-moderately enlarged": "mildly enlarged",
            "borderline enlareged": "mildly enlarged",
            "normal to mildly enlared": "normal",
            "mild-mmoderately enlarged": "mildly enlarged",
            "borderline enlarged and the global right ventricular function is borderline reduced": "mildly enlarged",
            "moderalely enlarged": "moderately enlarged",
            "mildly to moderately enlarged": "mildly enlarged",
            "mildly enlarged and systolic function is mildly depressed": "mildly enlarged",
            "mildly enlarged and systolic function is reduced": "mildly enlarged",
            "normal and systolic function appears mildly reduced": "normal",
            "somewhat enlarged": "mildly enlarged",
        }
    elif col == "LA":
        replace_dict = {
            "normal sized": "normal",
            "mildly dilated": "mildly enlarged",
            "moderately dilated": "moderately enlarged",
            "severely dilated": "severely enlarged",
            "borderline dilated": "mildly enlarged",
            "normal sized by measurement but visually appears enlarged compared to the small LV size": "normal",
            "mildly dilated (s/p transplant)": "mildly enlarged",
            "severely dilated (s/p OHT)": "severely enlarged",
        }
    elif col == "RA":
        replace_dict = {
            "normal in size": "normal",
            "mildly increased": "mildly enlarged",
            "moderately increased": "moderately enlarged",
            "severely increased": "severely enlarged",
            "mildly dilated": "mildly enlarged",
            "moderately dilated": "moderately enlarged",
            "severely dilated": "severely enlarged",
            "enlarged": "moderately enlarged",
            "not well seen": "normal",
            "right atrium not well visualized": "normal",
            "not visualized": "normal",
            "small in size and appears compressed by extracardiac structure / fluid": "normal",
            "small": "normal",
            "not well visualized": "normal",
        }
    chamber_df[col] = chamber_df[col].replace(replace_dict)
    return chamber_df


print(replaceLabels("LV").LV.value_counts(dropna=False))
print(replaceLabels("RV").RV.value_counts(dropna=False))
print(replaceLabels("LA").LA.value_counts(dropna=False))
print(replaceLabels("RA").RA.value_counts(dropna=False))


def replaceOrdinal(df):
    replace_dict = {
        "normal": 0,
        "mildly enlarged": 1,
        "moderately enlarged": 2,
        "severely enlarged": 3,
        # "moderately enlarged": 2,
        # "severely enlarged": 3,
    }
    df.RA = df.RA.replace(replace_dict)
    df.LV = df.LV.replace(replace_dict)
    df.RV = df.RV.replace(replace_dict)
    df.LA = df.LA.replace(replace_dict)


def create_binary_chamber_enlargement(row):
    if row["RV"] > 0 or row["LV"] > 0 or row["LA"] > 0 or row["RA"] > 0:
        return 1
    else:
        return 0


replaceOrdinal(chamber_df)
chamber_df["enl"] = chamber_df.apply(create_binary_chamber_enlargement, axis=1)
demo = pd.read_excel(DATADIR + "/csv_files/preprocessed/Demographic_data.xlsx")
demo
chamber_df = chamber_df[[
    "Phonetic ID",
    "Path",
    "AP/PA",
    "LV",
    "RV",
    "LA",
    "RA",
    "enl",
    'Height (in)',
    'Weight (lb)',
    'BSA (m²)',
    'Patient Gender',
    'LVIDd (cm)',
    'Left Atrium (cm)'
]]
chamber_df = pd.merge(chamber_df, demo, on="Phonetic ID", how="left")

display(chamber_df)

echo_quant = pd.read_csv(
    os.path.join(DATADIR, "csv_files", "echo_quant", "echo_quant.csv"), index_col=0
)
display(echo_quant)

demographics_df = pd.read_excel(
    os.path.join(DATADIR, "csv_files", "Demographic_data_by_pid.xlsx")
)
display(demographics_df)

drop_nans_in_cols = [
    'LA',
    'RA',
    'LV',
    'RV',
    'enl',
    'LVIDd (cm)',
    'Left Atrium (cm)',
]
    
chamber_df = chamber_df.dropna(subset=drop_nans_in_cols)
chamber_df.groupby("Phonetic ID").first().reset_index(drop=True, inplace=True)
chamber_df['AP/PA'].value_counts(dropna=False)

    

# separating into train/val/test split
train_df, test_df = train_test_split(chamber_df, test_size=0.2, random_state=33, stratify=chamber_df["enl"])
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=33, stratify=test_df["enl"])

display(train_df['LA'].value_counts(dropna=False, normalize=True))
display(val_df['LA'].value_counts(dropna=False, normalize=True))
display(test_df['LA'].value_counts(dropna=False, normalize=True))
display(train_df['RA'].value_counts(dropna=False, normalize=True))
display(val_df['RA'].value_counts(dropna=False, normalize=True))
display(test_df['RA'].value_counts(dropna=False, normalize=True))
display(train_df['LV'].value_counts(dropna=False, normalize=True))
display(val_df['LV'].value_counts(dropna=False, normalize=True))
display(test_df['LV'].value_counts(dropna=False, normalize=True))
display(train_df['RV'].value_counts(dropna=False, normalize=True))
display(val_df['RV'].value_counts(dropna=False, normalize=True))
display(test_df['RV'].value_counts(dropna=False, normalize=True))
display(train_df['enl'].value_counts(dropna=False, normalize=True))
display(val_df['enl'].value_counts(dropna=False, normalize=True))
display(test_df['enl'].value_counts(dropna=False, normalize=True))

# train_merged = pd.merge(train_df, chamber_df, on="Phonetic ID", how="inner")
# train_merged = pd.merge(train_merged, echo_quant, on="Phonetic ID", how="inner")
# train_merged = pd.merge(train_merged, demographics_df, on="Phonetic ID", how="inner")
# display(train_merged)

# test_merged = pd.merge(test_df, chamber_df, on="Phonetic ID", how="inner")
# test_merged = pd.merge(test_merged, echo_quant, on="Phonetic ID", how="left")
# test_merged = pd.merge(test_merged, demographics_df, on="Phonetic ID", how="left")
# display(test_merged)

# val_merged = pd.merge(val_df, chamber_df, on="Phonetic ID", how="inner")
# val_merged = pd.merge(val_merged, echo_quant, on="Phonetic ID", how="left")
# val_merged = pd.merge(val_merged, demographics_df, on="Phonetic ID", how="left")
# display(val_merged)

train_df.groupby("Phonetic ID").first().reset_index(drop=True, inplace=True)
test_df.groupby("Phonetic ID").first().reset_index(drop=True, inplace=True)
val_df.groupby("Phonetic ID").first().reset_index(drop=True, inplace=True)


train_df["split"] = "train"
test_df["split"] = "test"
val_df["split"] = "val"
train_df.iloc[:, 10:]

train_df = train_df[
    [
        "Phonetic ID",
        "Path",
        "split",
        "AP/PA",
        "LV",
        "RV",
        "LA",
        "RA",
        "enl",
        "Height (in)",
        "Weight (lb)",
        "BSA (m²)",
        "Sex",
        "Age",
        "LVIDd (cm)",
        "Left Atrium (cm)",
    ]
]
test_df = test_df[
    [
        "Phonetic ID",
        "Path",
        "split",
        "AP/PA",
        "LV",
        "RV",
        "LA",
        "RA",
        "enl",
        "Height (in)",
        "Weight (lb)",
        "BSA (m²)",
        "Sex",
        "Age",
        "LVIDd (cm)",
        "Left Atrium (cm)",
    ]
]
val_df = val_df[
    [
        "Phonetic ID",
        "Path",
        "split",
        "AP/PA",
        "LV",
        "RV",
        "LA",
        "RA",
        "enl",
        "Height (in)",
        "Weight (lb)",
        "BSA (m²)",
        "Sex",
        "Age",
        "LVIDd (cm)",
        "Left Atrium (cm)",
    ]
]
display(train_df)
display(test_df)
display(val_df)
concatenated = pd.concat([train_df, test_df, val_df])
concatenated.split.value_counts(dropna=False)
concatenated.drop_duplicates(subset=["Phonetic ID"], keep="last", inplace=True)
concatenated.isnull().sum()
concatenated.groupby("split").count()
concatenated.to_csv(os.path.join(CSVSAVEDIR, "AHA_data_final.csv"), index=False)

#show images of each chamber from train
train_df = pd.read_csv(os.path.join(CSVSAVEDIR, "AHA_data_final.csv"))
train_df = train_df[train_df['split'] == 'train']
train_df.reset_index(drop=True, inplace=True)
# plt.imshow one image for each ordinal category for LV, RV, LA, RA

#generate Table 1 for all data
aha_df = pd.read_csv(os.path.join(CSVSAVEDIR, "AHA_data_final.csv"))
from tableone import TableOne, load_dataset
aha_df.drop(columns = ['Phonetic ID', 'Path'], inplace=True)
#rename columns
replace_dict = {
    0: "normal",
    1: "mildly enlarged",
    2: "moderately enlarged",
    3: "severely enlarged",
}
aha_df.LV = aha_df.LV.replace(replace_dict)
aha_df.RV = aha_df.RV.replace(replace_dict)
aha_df.LA = aha_df.LA.replace(replace_dict)
aha_df.RA = aha_df.RA.replace(replace_dict)

aha_df.split = aha_df.split.replace(replace_dict)
aha_df.rename(columns={'AP/PA':'AP_PA', 'BSA (m²)':'BSA (m²)', 'LVIDd (cm)':'LVIDd (cm)', 'Left Atrium (cm)':'Left_Atrium (cm)', 'LA': 'Left Atrium', 'RA': 'Right Atrium', 'LV': 'Left Ventricle', 'RV': 'Right Ventricle', 'enl':'Cardiomegaly', 'AP/PA': 'Projection'}, inplace=True)
table_one = TableOne(aha_df,groupby='split',pval=True)
print(table_one.tabulate(tablefmt = "fancy_grid"))
table_one.to_latex('table1.tex')