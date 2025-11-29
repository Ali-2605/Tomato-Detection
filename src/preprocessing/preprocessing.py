import cv2 as cv 
import numpy as np
import os
from glob import glob
import pickle
import pandas as pd

class TomatoDataPreprocessor:
    def __init__(self, dataset_path, bins=32, target_size=(64, 64)):
        """
        Initialize the preprocessor for tomato classification
        
        Args:
            dataset_path: Path to the dataset folder containing train/val/test
            bins: Number of bins for color histogram (default: 32)
            target_size: Target size for resizing images (default: (64, 64))
        """
        self.dataset_path = dataset_path
        self.bins = bins
        self.target_size = target_size
        
    def parse_yolo_label(self, label_path, img_width, img_height):
        """
        Args:
            label_path: Path to the label .txt file
            img_width: Width of the corresponding image
            img_height: Height of the corresponding image
            
        Returns:
            List of tuples (class_id, x1, y1, x2, y2) for tomato classes only
        """
        tomato_boxes = []
        
        if not os.path.exists(label_path):
            return tomato_boxes
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                
                # Only process tomato classes (2 = Fresh Tomato, 3 = Rotten Tomato)
                if class_id in [2, 3]:
                    # Convert normalized YOLO coordinates to pixel coordinates
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Calculate bounding box corners
                    x1 = int(center_x - width/2)
                    y1 = int(center_y - height/2)
                    x2 = int(center_x + width/2)
                    y2 = int(center_y + height/2)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    tomato_boxes.append((class_id, x1, y1, x2, y2))
                    
        return tomato_boxes
    
    def extract_tomato_regions(self, image_path, label_path):
        """
        Args:
            image_path: Path to the image file
            label_path: Path to the corresponding label file
            
        Returns:
            List of tuples (cropped_image, class_id)
        """
        # Read the image
        image = cv.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return []
            
        # Convert BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Parse YOLO labels
        tomato_boxes = self.parse_yolo_label(label_path, img_width, img_height)
        
        cropped_tomatoes = []
        for class_id, x1, y1, x2, y2 in tomato_boxes:
            # Extract the tomato region
            cropped = image[y1:y2, x1:x2]
            
            # Skip if cropped region is too small
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue
                
            cropped_tomatoes.append((cropped, class_id))
            
        return cropped_tomatoes
    
    def extract_color_histogram(self, image):
        """
        Args:
            image: Input RGB image
            
        Returns:
            Flattened color histogram feature vector
        """
        # Calculate histograms for each RGB channel
        hist_r = cv.calcHist([image], [0], None, [self.bins], [0, 256])
        hist_g = cv.calcHist([image], [1], None, [self.bins], [0, 256])
        hist_b = cv.calcHist([image], [2], None, [self.bins], [0, 256])
        
        # Normalize histograms
        # it makes the sum of histogram values equal to 1
        hist_r = hist_r.flatten() / np.sum(hist_r)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_b = hist_b.flatten() / np.sum(hist_b)
        
        # Concatenate all histograms into single feature vector
        color_histogram = np.concatenate([hist_r, hist_g, hist_b])
        
        return color_histogram
    
    def process_dataset(self, split='train'):
        """
        Process a specific dataset split (train/val/test) and return structured results
        
        Args:
            split: Dataset split to process ('train', 'val', or 'test')
        
        Returns:
            List of dictionaries with img_name, feature_vector, class_id, and tomato_index
        """
        
        # Paths for images and labels
        images_path = os.path.join(self.dataset_path, split, 'images')
        labels_path = os.path.join(self.dataset_path, split, 'labels')
        
        # Check if split folder exists
        if not os.path.exists(images_path):
            print(f"Error: {split} images folder not found: {images_path}")
            return []
        
        if not os.path.exists(labels_path):
            print(f"Error: {split} labels folder not found: {labels_path}")
            return []
        
        # Get image files from specified split
        image_files = sorted(glob(os.path.join(images_path, '*.jpg')))
        
        if len(image_files) == 0:
            print(f"No JPG images found in {split} folder!")
            return []
        
        results = []
        
        print(f"Processing {len(image_files)} images from {split} folder...")
        for i, img_file in enumerate(image_files):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(image_files)}")
            
            # Get corresponding label file
            img_name = os.path.basename(img_file)
            label_file = os.path.join(labels_path, os.path.splitext(img_name)[0] + '.txt')
            
            # Extract tomato regions
            cropped_tomatoes = self.extract_tomato_regions(img_file, label_file)
            
            for j, (cropped_img, class_id) in enumerate(cropped_tomatoes):
                # Resize image 
                resized_img = cv.resize(cropped_img, self.target_size)
                
                # Extract color histogram features (96 features: 32 R + 32 G + 32 B)
                feature_vector = self.extract_color_histogram(resized_img)
                
                # Convert YOLO class ID to binary label: 2 (Fresh) -> 0, 3 (Rotten) -> 1
                binary_label = 0 if class_id == 2 else 1
                
                # Create result entry
                result = {
                    'img_name': img_name,
                    'feature_vector': feature_vector.tolist(),  
                    'class_id': binary_label,  # Now stores 0 or 1 instead of 2 or 3
                    'tomato_index': j  
                }
                
                results.append(result)
        
        return results
    
    def save_data(self, data, base_filename="preprocessed_data"):
        """
        Save data to both CSV and pickle files
        
        Args:
            data: List of dictionaries with processed data
            base_filename: Base name for output files
        """
        # Create DataFrame
        df_data = []
        for item in data:
            row = {
                'img_name': item['img_name'],
                'class_id': item['class_id'],
                'tomato_index': item['tomato_index']
            }
            # Add feature vector components (96 features: R0-R31, G0-G31, B0-B31)
            for i, value in enumerate(item['feature_vector']):
                if i < 32:
                    row[f'R{i}'] = value
                elif i < 64:
                    row[f'G{i-32}'] = value
                else:
                    row[f'B{i-64}'] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        csv_file = f"{base_filename}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Data saved to CSV: {csv_file}")
        
        # Save to pickle (preserving original structure)
        pkl_file = f"{base_filename}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to pickle: {pkl_file}")
        
        return csv_file, pkl_file

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    dataset_path = "../../dataSet"      

    split_choice = 'train'  # Change this to 'val' or 'test' as needed
    
    preprocessor = TomatoDataPreprocessor(dataset_path, bins=32, target_size=(64, 64))

    # Process specified split 
    processed_data = preprocessor.process_dataset(split_choice)
    
    # Save to CSV and pickle files
    if processed_data:
        csv_file, pkl_file = preprocessor.save_data(processed_data, f"preprocessed_data_{split_choice}")
        
        print(f"\nProcessing completed!")
        print(f"Total tomato samples: {len(processed_data)}")
        print(f"Feature vector length: 96 (32 R + 32 G + 32 B)")
        print(f"Files created:")
        print(f"  - {csv_file}")
        print(f"  - {pkl_file}")
        
    else:
        print("No data processed!")