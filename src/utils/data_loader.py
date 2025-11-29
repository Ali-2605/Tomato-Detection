import pickle 
import numpy as np
import os

def load_data(split="train", bins=32):
    """
    Load preprocessed data from pickle file
    
    Args:
        path (str): Path to preprocessed pickle file
        
    Returns:
        tuple: (X, y) where X is feature matrix and y is labels
    """
    if split.lower() not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', or 'test'")
    
    path = os.path.join(os.path.dirname(__file__), f"../preprocessing/preprocessed_data_{split.lower()}{bins}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed data file not found: {path}")
        
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
        
    # Extract features and labels
    X = []
    y = []
    
    for item in data:
        X.append(item['feature_vector'])
        y.append(item['class_id'])

    X = np.array(X)
    y = np.array(y)

    return X, y
