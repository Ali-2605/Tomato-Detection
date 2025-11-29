# Tomato Detection Project

A machine learning project for classifying tomatoes as fresh or rotten using computer vision and logistic regression.

## Project Structure

```
MlProject/
├── README.md
├── requirements.txt
├── .gitignore
├── dataSet/                    # Create this folder and put your dataset here
│   ├── train/
│   │   ├── images/            # Training images (.jpg files)
│   │   └── labels/            # Training labels (.txt files)
│   ├── val/
│   │   ├── images/            # Validation images (.jpg files)
│   │   └── labels/            # Validation labels (.txt files)
│   └── test/
│       ├── images/            # Test images (.jpg files)
│       └── labels/            # Test labels (.txt files)
└── src/
    ├── preprocessing/
    │   ├── preprocessing.py    # Main preprocessing script
    │   └── visualize_tomatoes.py  # Dataset visualization (optional)
    └── logisticRegression/
        └── LogisticRegression.py  # Logistic regression implementation
```

## Setup Instructions

### 1. Dataset Setup
- Create a folder called `dataSet` in the project root
- Place your YOLO format dataset inside this folder following the structure above
- The dataset should contain tomato images with corresponding label files
- Labels should use YOLO format with class IDs: 2 (Fresh) and 3 (Rotten)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing
```bash
cd src/preprocessing
python preprocessing.py
```

After running this script, you should see two files created:
- `preprocessed_data_train.csv` - CSV representation of the data (for viewing only)
- `preprocessed_data_train.pkl` - Pickle file containing the actual preprocessed data

**Note**: The CSV file is not important for the machine learning pipeline - it's just there for data representation and inspection. The pickle file contains the actual preprocessed features used by the model.

### 4. Optional: Visualize Dataset
```bash
python visualize_tomatoes.py
```

**Note**: The `visualize_tomatoes.py` script is optional and only for visualization purposes to help you understand the preprocessed dataset. It's not required for the machine learning workflow.

## Features

- **Color Histogram Extraction**: Extracts 96-dimensional color histogram features (32 bins each for R, G, B channels)
- **YOLO Label Processing**: Converts YOLO format labels to binary classification (Fresh=0, Rotten=1)
- **Image Preprocessing**: Crops tomato regions from images and resizes to 64x64 pixels
- **Data Persistence**: Saves preprocessed data in both CSV and pickle formats

## Usage

1. Follow the setup instructions above
2. Run preprocessing to extract features from your dataset
3. Use the generated pickle file for training machine learning models
4. Optionally use visualization scripts to inspect your data