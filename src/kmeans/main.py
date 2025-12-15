import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_data
from KMeansCluster import KMeansCluster

class ModelVisualization:
    def __init__(self, bins=32, data_split='test', k=2, use_normalization=True):
        """
        Initialize visualization with data
        
        Args:
            bins: Number of histogram bins
            data_split: Which split to evaluate on ('train', 'test')
            k: Number of clusters
            use_normalization: Whether to normalize features
        """
        self.bins = bins
        self.data_split = data_split
        
        # Load dataset
        self.X_train, self.y_train = load_data("train", bins=self.bins)
        self.X_eval, self.y_eval = load_data(data_split, bins=self.bins)
        
        print(f"Training set size: {len(self.y_train)}")
        print(f"Evaluation set ({data_split}): {len(self.y_eval)}")
        print(f"Training distribution - Fresh: {np.sum(self.y_train == 0)}, Rotten: {np.sum(self.y_train == 1)}")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
        
        self.model = KMeansCluster(k=k, bins=self.bins, use_normalization=use_normalization)
        self.metrics = None
        
    def train_model(self, save_path):
        """
        Train model and save to specified path
        
        Args:
            save_path: Path to save trained model
        """
        print("\n" + "="*50)
        print("Training model...")
        print("="*50)
        
        self.model.train("train")
        self.model.save_model(save_path)
        print(f"\nModel saved to: {save_path}")
        
        # Print cluster composition
        composition = self.model.get_cluster_composition(self.X_train, self.y_train)
        print("\nCluster Composition:")
        for c, data in composition.items():
            print(f"  Cluster {c}: Fresh={data['fresh']}, Rotten={data['rotten']}, Total={data['total']}")
            print(f"    -> Mapped to label: {self.model.cluster_to_label[c]}")
        
    def load_model(self, model_path):
        """
        Load model from specified path
        
        Args:
            model_path: Path to load model from
        """
        print(f"\nLoading model from: {model_path}")
        self.model.load_model(model_path)
        print("Model loaded successfully!")
        
    def evaluate(self):
        """
        Compute metrics for evaluation set
        """
        print(f"\nEvaluating on {self.data_split} set...")
        self.metrics = self.model.compute_metrics(self.X_eval, self.y_eval)

    def plot_inertia_curve(self):
        """Plot inertia (loss) curve over iterations"""
        inertia_history = self.model.get_inertia_history()
        
        if len(inertia_history) == 0:
            print("No inertia history available. Train the model first.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(inertia_history) + 1), inertia_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        plt.title(f'KMeans Convergence Curve (K={self.model.k})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add annotation for final inertia
        final_inertia = inertia_history[-1]
        plt.annotate(f'Final: {final_inertia:.2f}', 
                     xy=(len(inertia_history), final_inertia),
                     xytext=(len(inertia_history) - 10, final_inertia + (inertia_history[0] - final_inertia) * 0.1),
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='red'),
                     color='red')
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        if self.metrics is None:
            print("No metrics available. Run evaluate() first.")
            return
        
        cm = self.metrics['confusion_matrix']
        confusion_matrix = np.array([
            [cm['true_fresh'], cm['false_rotten']],
            [cm['false_fresh'], cm['true_rotten']]
        ])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap='Reds')
        plt.title('Confusion Matrix: Fresh vs Rotten Tomatoes', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        classes = ['Fresh (0)', 'Rotten (1)']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=11)
        plt.yticks(tick_marks, classes, fontsize=11)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black",
                        fontsize=20, fontweight='bold')
        
        # Add accuracy info
        accuracy = self.metrics['accuracy'] * 100
        total = len(self.y_eval)
        plt.text(0.02, 0.98, f'Total: {total}\nAccuracy: {accuracy:.1f}%', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat'), 
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def print_metrics(self):
        """Print comprehensive model metrics"""
        if self.metrics is None:
            print("No metrics available. Run evaluate() first.")
            return
        
        m = self.metrics
        
        print("\n" + "="*60)
        print(" "*15 + "MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Dataset: {self.data_split.upper()}")
        print(f"Accuracy:             {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        print(f"Macro F1-Score:       {m['macro_f1']:.4f}")
        print()
        print("Class-wise Metrics:")
        print("-" * 60)
        print(f"Fresh Tomatoes (Class 0):")
        print(f"  Precision: {m['fresh']['precision']:.4f}  |  Recall: {m['fresh']['recall']:.4f}  |  F1: {m['fresh']['f1']:.4f}")
        print()
        print(f"Rotten Tomatoes (Class 1):")
        print(f"  Precision: {m['rotten']['precision']:.4f}  |  Recall: {m['rotten']['recall']:.4f}  |  F1: {m['rotten']['f1']:.4f}")
        print("="*60)


def main():
    # Configuration
    TRAIN_NEW_MODEL = True  # Set to False to load existing model
    MODEL_PATH = "model_v6.pkl"  # Path for saving/loading model
    DATA_SPLIT = "test"  # Which split to evaluate on: 'train' or 'test'
    BINS = 32  # Number of histogram bins
    K = 17  # Number of clusters (try 2, 3, 4, 5, etc.)
    USE_NORMALIZATION = True  # Normalize features before clustering
    
    print("="*60)
    print(" "*15 + "TOMATO CLASSIFIER (KMeans)")
    print("="*60)
    
    # Initialize visualization
    viz = ModelVisualization(bins=BINS, data_split=DATA_SPLIT, k=K, use_normalization=USE_NORMALIZATION)
    
    # Train or load model
    if TRAIN_NEW_MODEL:
        viz.train_model(MODEL_PATH)
    else:
        viz.load_model(MODEL_PATH)
    
    # Evaluate model
    viz.evaluate()
    
    # Display results
    viz.print_metrics()
    viz.plot_inertia_curve()
    viz.plot_confusion_matrix()


if __name__ == "__main__":
    main()
