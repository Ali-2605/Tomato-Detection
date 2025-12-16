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
            Flattened color histogram feature vector with RGB and HSV features
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
        
        # Convert to HSV for additional color features
        hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        hist_h = cv.calcHist([hsv_image], [0], None, [self.bins], [0, 180])
        hist_s = cv.calcHist([hsv_image], [1], None, [self.bins], [0, 256])
        hist_v = cv.calcHist([hsv_image], [2], None, [self.bins], [0, 256])
        
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)
        
        # Concatenate all histograms into single feature vector
        # RGB (3 * bins) + HSV (3 * bins) = 6 * bins features
        color_histogram = np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v])
        
        return color_histogram
    
    def augment_image(self, image, augmentation_type='brightness'):
        """
        Apply augmentation to an image
        
        Args:
            image: Input RGB image
            augmentation_type: Type of augmentation ('brightness', 'contrast', 'flip', 'rotate')
            
        Returns:
            Augmented image
        """
        if augmentation_type == 'brightness':
            # Increase brightness
            hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] = hsv[:, :, 2] * 1.2  # Increase V channel by 20%
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        
        elif augmentation_type == 'contrast':
            # Increase contrast
            alpha = 1.3  # Contrast control
            beta = 0     # Brightness control
            return cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        elif augmentation_type == 'flip':
            # Horizontal flip
            return cv.flip(image, 1)
        
        elif augmentation_type == 'rotate':
            # Rotate by 10 degrees
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv.getRotationMatrix2D(center, 10, 1.0)
            return cv.warpAffine(image, rotation_matrix, (w, h))
        
        elif augmentation_type == 'noise':
            # Add small gaussian noise
            noise = np.random.normal(0, 10, image.shape).astype(np.float32)
            noisy_image = image.astype(np.float32) + noise
            return np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return image
    
    def balance_dataset(self, data, images_path=None, labels_path=None):
        """
        Balance the dataset by augmenting the minority class with real image augmentations
        
        Args:
            data: List of dictionaries with processed data
            images_path: Path to images folder (needed for real augmentation)
            labels_path: Path to labels folder (needed for real augmentation)
            
        Returns:
            Balanced dataset
        """
        # Count samples per class
        class_counts = {}
        class_samples = {}
        
        for item in data:
            class_id = item['class_id']
            if class_id not in class_counts:
                class_counts[class_id] = 0
                class_samples[class_id] = []
            class_counts[class_id] += 1
            class_samples[class_id].append(item)
        
        print(f"\nOriginal class distribution:")
        for class_id, count in class_counts.items():
            class_name = "Fresh" if class_id == 0 else "Rotten"
            print(f"  Class {class_id} ({class_name}): {count} samples")
        
        # Find majority class count
        max_count = max(class_counts.values())
        
        # Augment minority classes
        balanced_data = data.copy()
        augmentation_types = ['brightness', 'contrast', 'flip', 'rotate', 'noise']
        
        for class_id, count in class_counts.items():
            if count < max_count:
                samples_to_add = max_count - count
                class_name = "Fresh" if class_id == 0 else "Rotten"
                print(f"\nAugmenting class {class_id} ({class_name}): adding {samples_to_add} samples")
                
                # Randomly select samples from this class and augment them
                for i in range(samples_to_add):
                    # Select a random sample from this class
                    original_sample = class_samples[class_id][i % len(class_samples[class_id])]
                    
                    # Select augmentation type
                    aug_type = augmentation_types[i % len(augmentation_types)]
                    
                    # Try real image augmentation if paths are provided
                    if images_path and labels_path:
                        img_name = original_sample['img_name']
                        tomato_idx = original_sample['tomato_index']
                        
                        # Progress indicator
                        images_remaining = samples_to_add - (i + 1)
                        print(f"  [{i+1}/{samples_to_add}] Duplicating '{img_name}' (tomato #{tomato_idx}) with '{aug_type}' augmentation - {images_remaining} images remaining")
                        
                        img_file = os.path.join(images_path, img_name)
                        label_file = os.path.join(labels_path, os.path.splitext(img_name)[0] + '.txt')
                        
                        # Extract tomato regions from original image
                        cropped_tomatoes = self.extract_tomato_regions(img_file, label_file)
                        
                        # Find the specific tomato by index
                        if tomato_idx < len(cropped_tomatoes):
                            cropped_img, _ = cropped_tomatoes[tomato_idx]
                            
                            # Apply augmentation to the actual image
                            augmented_img = self.augment_image(cropped_img, aug_type)
                            
                            # Resize and extract features from augmented image
                            resized_img = cv.resize(augmented_img, self.target_size)
                            augmented_features = self.extract_color_histogram(resized_img)
                            
                            # Create new sample with real augmented features
                            augmented_sample = {
                                'img_name': f"{img_name}_aug{aug_type}{i}",
                                'feature_vector': augmented_features.tolist(),
                                'class_id': class_id,
                                'tomato_index': tomato_idx
                            }
                            
                            balanced_data.append(augmented_sample)
                            continue
                    
        
        # Count samples after balancing
        balanced_counts = {}
        for item in balanced_data:
            class_id = item['class_id']
            balanced_counts[class_id] = balanced_counts.get(class_id, 0) + 1
        
        print(f"\nBalanced class distribution:")
        for class_id, count in balanced_counts.items():
            class_name = "Fresh" if class_id == 0 else "Rotten"
            print(f"  Class {class_id} ({class_name}): {count} samples")
        
        return balanced_data
    
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
                
                # Extract color histogram features (192 features: 32 R + 32 G + 32 B + 32 H + 32 S + 32 V)
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
    
    def compute_normalization_stats(self, data):
        """
        Compute mean and std for Z-score normalization from data
        
        Args:
            data: List of dictionaries with processed data
            
        Returns:
            Tuple of (mean, std) arrays
        """
        # Extract all feature vectors
        feature_vectors = np.array([item['feature_vector'] for item in data])
        
        # Compute mean and std for each feature
        mean = np.mean(feature_vectors, axis=0)
        std = np.std(feature_vectors, axis=0) + 1e-8  # Add small constant to avoid division by zero
        
        return mean, std
    
    def normalize_data(self, data, mean, std):
        """
        Apply Z-score normalization to data using provided statistics
        
        Args:
            data: List of dictionaries with processed data
            mean: Mean array for normalization
            std: Std array for normalization
            
        Returns:
            Normalized data
        """
        normalized_data = []
        for item in data:
            feature_vector = np.array(item['feature_vector'])
            normalized_features = (feature_vector - mean) / std
            
            normalized_item = item.copy()
            normalized_item['feature_vector'] = normalized_features.tolist()
            normalized_data.append(normalized_item)
        
        return normalized_data
    
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
            # Add feature vector components (192 features: R0-R31, G0-G31, B0-B31, H0-H31, S0-S31, V0-V31)
            for i, value in enumerate(item['feature_vector']):
                if i < self.bins:
                    row[f'R{i}'] = value
                elif i < 2 * self.bins:
                    row[f'G{i - self.bins}'] = value
                elif i < 3 * self.bins:
                    row[f'B{i - 2 * self.bins}'] = value
                elif i < 4 * self.bins:
                    row[f'H{i - 3 * self.bins}'] = value
                elif i < 5 * self.bins:
                    row[f'S{i - 4 * self.bins}'] = value
                else:
                    row[f'V{i - 5 * self.bins}'] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        csv_file = f"{base_filename}{self.bins}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Data saved to CSV: {csv_file}")
        
        # Save to pickle (preserving original structure)
        pkl_file = f"{base_filename}{self.bins}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to pickle: {pkl_file}")
        
        return csv_file, pkl_file

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    dataset_path = "../../dataSet"      

    split_choice = 'train'  # Change this to 'val' or 'test' or 'train' as needed
    
    preprocessor = TomatoDataPreprocessor(dataset_path, bins=32, target_size=(64, 64))

    # Process specified split 
    processed_data = preprocessor.process_dataset(split_choice)
    
    # Balance dataset if it's training data
    if processed_data and split_choice == 'train':
        # Pass image paths for real augmentation
        images_path = os.path.join(dataset_path, split_choice, 'images')
        labels_path = os.path.join(dataset_path, split_choice, 'labels')
        processed_data = preprocessor.balance_dataset(processed_data, split_choice, images_path, labels_path)
    
    # Normalize data
    if processed_data:
        if split_choice == 'train':
            # Compute and save normalization statistics from training data
            mean, std = preprocessor.compute_normalization_stats(processed_data)
            
            # Save normalization stats
            norm_stats = {'mean': mean, 'std': std}
            norm_stats_file = f"norm_stats{preprocessor.bins}.pkl"
            with open(norm_stats_file, 'wb') as f:
                pickle.dump(norm_stats, f)
            print(f"\nNormalization statistics saved to {norm_stats_file}")
            
            # Apply normalization
            processed_data = preprocessor.normalize_data(processed_data, mean, std)
        else:
            # Load normalization stats from training data
            norm_stats_file = f"norm_stats{preprocessor.bins}.pkl"
            if os.path.exists(norm_stats_file):
                with open(norm_stats_file, 'rb') as f:
                    norm_stats = pickle.load(f)
                mean = norm_stats['mean']
                std = norm_stats['std']
                print(f"\nLoaded normalization statistics from {norm_stats_file}")
                
                # Apply normalization
                processed_data = preprocessor.normalize_data(processed_data, mean, std)
            else:
                print(f"\nWarning: {norm_stats_file} not found. Data will not be normalized.")
                print("Please process training data first to generate normalization statistics.")
    
    # Save to CSV and pickle files
    if processed_data:
        csv_file, pkl_file = preprocessor.save_data(processed_data, f"preprocessed_data_{split_choice}")
        
        print(f"\nProcessing completed!")
        print(f"Total tomato samples: {len(processed_data)}")
        print(f"Feature vector length: {preprocessor.bins * 6} ({preprocessor.bins} R + {preprocessor.bins} G + {preprocessor.bins} B + {preprocessor.bins} H + {preprocessor.bins} S + {preprocessor.bins} V)")
        print(f"Files created:")
        print(f"  - {csv_file}")
        print(f"  - {pkl_file}")
        
    else:
        print("No data processed!")