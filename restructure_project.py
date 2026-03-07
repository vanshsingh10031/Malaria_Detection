import os
import shutil

# Create required folders
folders = [
    "data",
    "models",
    "src",
    "sample_images"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Folders created.")

# Move dataset
if os.path.exists("malaria_dataset"):
    shutil.move("malaria_dataset", "data/malaria_dataset")
    print("Dataset moved.")

# Move model
if os.path.exists("malaria_model.h5"):
    shutil.move("malaria_model.h5", "models/malaria_model.h5")
    print("Model moved.")

# Move scripts
scripts = ["train_model.py", "predict.py", "DatasetSplitting.py"]

for script in scripts:
    if os.path.exists(script):
        shutil.move(script, f"src/{script}")
        print(f"{script} moved.")

# Move sample image
if os.path.exists("sample_cell.webp"):
    shutil.move("sample_cell.webp", "sample_images/sample_cell.webp")
    print("Sample image moved.")

print("Project restructuring complete.")