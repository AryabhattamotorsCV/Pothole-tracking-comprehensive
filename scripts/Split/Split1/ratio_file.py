import os
import shutil
from pathlib import Path

# Change to the scripts directory
os.chdir(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts")

# Create directories if they don't exist
os.makedirs("Split/Split1", exist_ok=True)
os.makedirs("Split/Split2", exist_ok=True)

# Get all Python files
files = list(Path(".").glob("*.py"))
total_files = len(files)

# Calculate split sizes
count1 = int(0.8 * total_files)  # 80% of files to Split1
count2 = total_files - count1     # Remaining 20% to Split2

# Get a list of filenames already copied to Split1
split1_files = set()

# Copy first 80% of files to Split1
for file in files[:count1]:
    shutil.copy(str(file), "Split/Split1")
    split1_files.add(file.name)  # Store copied file names
    print(f"Copying {file} to Split1")

# Copy remaining files to Split2
for file in files[count1:]:
    if file.name not in split1_files:  # Ensure it's not in Split1
        shutil.copy(str(file), "Split/Split2")
        print(f"Copying {file} to Split2")
