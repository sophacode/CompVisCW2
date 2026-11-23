from pathlib import Path
import shutil
import pandas as pd

ZIP_NAME = "archive.zip"
EXTRACTED_FOLDER_NAME = "raw_dataset"
OUTPUT_FOLDER_NAME = "clean_dataset"

ROI_X1 = "Roi.X1"
ROI_Y1 = "Roi.Y1"
ROI_X2 = "Roi.X2"
ROI_Y2 = "Roi.Y2"

print("Finished loading libraries")

# Zip extraction function
print("Starting zip extraction")
zip_path = Path(ZIP_NAME)
if not zip_path.exists():
    raise FileNotFoundError(f"Zip file '{ZIP_NAME}' not found.")
if Path(EXTRACTED_FOLDER_NAME).exists():
    print(f"Folder '{EXTRACTED_FOLDER_NAME}' already exists. Skipping extraction.")
else:
    shutil.unpack_archive(ZIP_NAME, EXTRACTED_FOLDER_NAME)
    print("Finished zip extraction")

# Create the output folder if it doesn't exist
print("Creating output folder if it doesn't exist")
output_path = Path(OUTPUT_FOLDER_NAME)
if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)

# Extract the CSV files and load them and find the elligible images
print("Starting CSV extraction")
train_data = pd.read_csv("raw_dataset/train.csv")
test_data = pd.read_csv("raw_dataset/test.csv")
filtered_data = []
for df in [train_data, test_data]:
    widths = df[ROI_X2] - df[ROI_X1]
    heights = df[ROI_Y2] - df[ROI_Y1]
    mask = (widths >= 32) & (heights >= 32)
    filtered = df.loc[mask].copy()
    filtered_data.append(filtered)
clean_data = pd.concat(filtered_data, ignore_index=True)
clean_data.to_csv(f"{OUTPUT_FOLDER_NAME}/data.csv", index=False)
print("Finished CSV extraction")