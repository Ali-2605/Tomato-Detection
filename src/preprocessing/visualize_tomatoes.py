import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

# Configuration - Change this to switch between datasets
DATASET_SPLIT = 'train'  # Change to 'val' or 'test' as needed

def load_data(filename=None):
    """Load preprocessed data from pickle file"""
    if filename is None:
        filename = f"preprocessed_data_{DATASET_SPLIT}.pkl"
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Please run preprocessing.py first.")
        return None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} tomato samples from {filename}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def group_tomatoes_by_image(data):
    """Group tomatoes by their source image"""
    grouped = defaultdict(list)
    for item in data:
        grouped[item['img_name']].append(item)
    return dict(grouped)

def reconstruct_original_image(img_name, dataset_path="../../dataSet"):
    """Load the original image from dataset"""
    img_path = os.path.join(dataset_path, DATASET_SPLIT, 'images', img_name)
    
    if os.path.exists(img_path):
        img = cv.imread(img_path)
        if img is not None:
            return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return None

def get_tomato_bounding_boxes(img_name, dataset_path="../../dataSet"):
    """Get bounding boxes for tomatoes in the image"""
    label_path = os.path.join(dataset_path, DATASET_SPLIT, 'labels', os.path.splitext(img_name)[0] + '.txt')
    
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                if class_id in [2, 3]:  # Tomato classes
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append((class_id, center_x, center_y, width, height))
    
    return boxes

def plot_original_image_with_boxes(ax, original_img, boxes, tomato_idx=None, title="Original Image"):
    """Plot original image with bounding boxes, optionally highlighting one tomato"""
    ax.imshow(original_img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    if not boxes:
        return
    
    img_h, img_w = original_img.shape[:2]
    
    if tomato_idx is None:
        # Single tomato case - just draw the first box
        if len(boxes) > 0:
            class_id, cx, cy, w, h = boxes[0]
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                               edgecolor='red' if class_id == 3 else 'green', 
                               facecolor='none')
            ax.add_patch(rect)
    else:
        # Multiple tomatoes case - highlight the specified one
        for i, (class_id, cx, cy, w, h) in enumerate(boxes):
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            
            if i == tomato_idx:
                # Highlight the current tomato with thick green/red border
                color = 'red' if class_id == 3 else 'green'
                linewidth = 4
                alpha = 1.0
            else:
                # Other tomatoes with thin gray border
                color = 'gray'
                linewidth = 1
                alpha = 0.5
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=linewidth, 
                               edgecolor=color, facecolor='none', alpha=alpha)
            ax.add_patch(rect)
            
            # Add tomato number
            text_color = color if i == tomato_idx else 'gray'
            ax.text(x1, y1-5, f'T{i+1}', fontsize=10 if i == tomato_idx else 8, 
                          color=text_color, weight='bold' if i == tomato_idx else 'normal',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

def plot_cropped_tomato(ax, original_img, boxes, tomato_idx, class_text):
    """Plot cropped and resized tomato"""
    ax.set_title(f'Resized Tomato{" " + str(tomato_idx + 1) if tomato_idx is not None else ""} (64x64) - {class_text}', fontsize=10)
    ax.axis('off')
    
    if not boxes or tomato_idx >= len(boxes):
        return
    
    img_h, img_w = original_img.shape[:2]
    class_id, cx, cy, w, h = boxes[tomato_idx if tomato_idx is not None else 0]
    x1 = max(0, int((cx - w/2) * img_w))
    y1 = max(0, int((cy - h/2) * img_h))
    x2 = min(img_w, int((cx + w/2) * img_w))
    y2 = min(img_h, int((cy + h/2) * img_h))
    
    cropped = original_img[y1:y2, x1:x2]
    if cropped.size > 0:
        resized = cv.resize(cropped, (64, 64))
        ax.imshow(resized)

def plot_color_histogram(ax, feature_vector, tomato_idx=None):
    """Plot RGB color histogram"""
    bins = len(feature_vector) // 3
    hist_r = feature_vector[:bins]
    hist_g = feature_vector[bins:2*bins]
    hist_b = feature_vector[2*bins:3*bins]
    
    x = np.arange(bins)
    ax.bar(x - 0.3, hist_r, 0.3, alpha=0.7, color='red', label='Red')
    ax.bar(x, hist_g, 0.3, alpha=0.7, color='green', label='Green')
    ax.bar(x + 0.3, hist_b, 0.3, alpha=0.7, color='blue', label='Blue')
    
    title = 'RGB Color Histogram'
    if tomato_idx is not None:
        title += f' - Tomato {tomato_idx + 1}'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Bin', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8)

def plot_feature_vector(ax, feature_vector, tomato_idx=None):
    """Plot feature vector"""
    bins = len(feature_vector) // 3
    
    ax.plot(feature_vector, 'o-', markersize=2)
    
    title = 'Feature Vector (96 features)'
    if tomato_idx is not None:
        title += f' - Tomato {tomato_idx + 1}'
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Feature Index', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.axvline(x=bins, color='red', linestyle='--', alpha=0.5, label='R|G')
    ax.axvline(x=2*bins, color='green', linestyle='--', alpha=0.5, label='G|B')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8)

def visualize_image_tomatoes(img_name, tomato_data, dataset_path="../../dataSet"):
    """Visualize all tomatoes from a specific image"""
    
    # Load original image
    original_img = reconstruct_original_image(img_name, dataset_path)
    if original_img is None:
        print(f"Could not load original image: {img_name}")
        return
    
    # Get bounding boxes
    boxes = get_tomato_bounding_boxes(img_name, dataset_path)
    
    num_tomatoes = len(tomato_data)
    print(f"Visualizing {num_tomatoes} tomatoes from image: {img_name}")
    
    if num_tomatoes == 1:
        # Single tomato: 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Image: {img_name} - Single Tomato', fontsize=14)
        
        tomato = tomato_data[0]
        class_text = "FRESH" if tomato['class_id'] == 2 else "ROTTEN"
        feature_vector = np.array(tomato['feature_vector'])
        
        # Use common plotting functions
        plot_original_image_with_boxes(axes[0, 0], original_img, boxes, title='Original Image')
        plot_cropped_tomato(axes[0, 1], original_img, boxes, 0, class_text)
        plot_color_histogram(axes[1, 0], feature_vector)
        plot_feature_vector(axes[1, 1], feature_vector)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Multiple tomatoes: create separate figure for each tomato
        for tomato_idx, tomato in enumerate(tomato_data):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            class_text = "FRESH" if tomato['class_id'] == 2 else "ROTTEN"
            fig.suptitle(f'Image: {img_name} - Tomato {tomato_idx + 1} of {num_tomatoes} ({class_text})', fontsize=14)
            
            feature_vector = np.array(tomato['feature_vector'])
            
            # Use common plotting functions
            plot_original_image_with_boxes(axes[0, 0], original_img, boxes, tomato_idx, 
                                         f'Original Image - Tomato {tomato_idx + 1} Highlighted')
            plot_cropped_tomato(axes[0, 1], original_img, boxes, tomato_idx, class_text)
            plot_color_histogram(axes[1, 0], feature_vector, tomato_idx)
            plot_feature_vector(axes[1, 1], feature_vector, tomato_idx)
            
            plt.tight_layout()
            plt.show()
            
            # Small pause between figures so they don't overlap
            plt.pause(0.1)

def visualize_by_index(index, data=None):
    """Visualize tomatoes by image index"""
    if data is None:
        data = load_data()
    
    if data is None:
        return
    
    # Group by image
    grouped = group_tomatoes_by_image(data)
    image_names = list(grouped.keys())
    
    if index >= len(image_names):
        print(f"Index {index} out of range. Available indices: 0 to {len(image_names)-1}")
        print("Available images:")
        for i, img_name in enumerate(image_names[:10]): 
            count = len(grouped[img_name])
            print(f"  {i}: {img_name} ({count} tomato{'s' if count > 1 else ''})")
        if len(image_names) > 10:
            print(f"  ... and {len(image_names)-10} more")
        return
    
    img_name = image_names[index]
    tomato_data = grouped[img_name]
    
    visualize_image_tomatoes(img_name, tomato_data)

if __name__ == "__main__":
    import sys
    
    # Load data
    data = load_data()
    if data is None:
        exit(1)
    
    if len(sys.argv) > 1:
        # Command line usage: python visualize_tomatoes.py 5
        try:
            index = int(sys.argv[1])
            visualize_by_index(index, data)
        except ValueError:
            print("Usage: python visualize_tomatoes.py <image_index>")
            print("Example: python visualize_tomatoes.py 5")
    else:
        # Interactive mode
        grouped = group_tomatoes_by_image(data)
        print(f"Found {len(grouped)} unique images with tomatoes:")
        
        for i, (img_name, tomatoes) in enumerate(list(grouped.items())[:10]):
            print(f"  {i}: {img_name} ({len(tomatoes)} tomato{'s' if len(tomatoes) > 1 else ''})")
        
        if len(grouped) > 10:
            print(f"  ... and {len(grouped)-10} more")
        
        try:
            index = int(input(f"\nEnter image index to visualize (0-{len(grouped)-1}): "))
            visualize_by_index(index, data)
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")