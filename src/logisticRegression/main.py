import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_data
from LogisticRegression import LogisticRegression

class ModelVisualization:
    def __init__(self, bins=32, data_split='test', learning_rate=0.1, max_iterations=1000, l2_lambda=0.01):
        """
        Initialize visualization with data
        
        Args:
            bins: Number of histogram bins
            data_split: Which split to evaluate on ('train', 'val', 'test')
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of training iterations
            l2_lambda: L2 regularization parameter
        """
        self.bins = bins
        self.data_split = data_split
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l2_lambda = l2_lambda
        
        # Load dataset
        self.X_train, self.y_train = load_data("train", bins=self.bins)
        self.X_eval, self.y_eval = load_data(data_split, bins=self.bins)
        
        print(f"Training set size: {len(self.y_train)}")
        print(f"Evaluation set ({data_split}): {len(self.y_eval)}")
        print(f"Training distribution - Fresh: {np.sum(self.y_train == 0)}, Rotten: {np.sum(self.y_train == 1)}")
        
        self.model = LogisticRegression(learning_rate=self.learning_rate, max_iterations=self.max_iterations, 
                                       bins=self.bins, l2_lambda=self.l2_lambda)
        self.metrics = None
        self.roc_data = None
        self.loss_history = []
        
    def train_model(self, save_path):
        """
        Train model and save to specified path
        
        Args:
            save_path: Path to save trained model
        """
        print("\n" + "="*50)
        print("Training model...")
        print("="*50)
        
        self.loss_history = self.model.train("train")
        self.model.save_model(save_path)
        print(f"\nModel saved to: {save_path}")
        
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
        Compute metrics and ROC data for evaluation set
        """
        print(f"\nEvaluating on {self.data_split} set...")
        self.metrics = self.model.compute_metrics(self.X_eval, self.y_eval)
        self.roc_data = self.model.compute_roc_curve(self.X_eval, self.y_eval)
        
    def plot_loss(self):
        """Plot training loss curve"""
        if len(self.loss_history) == 0:
            print("No loss history available. Model was loaded, not trained.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Iterations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
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
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        if self.roc_data is None:
            print("No ROC data available. Run evaluate() first.")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.roc_data['fpr'], self.roc_data['tpr'], 'b-', linewidth=2, 
                label=f"ROC Curve (AUC = {self.roc_data['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curve: Fresh vs Rotten Tomatoes', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
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
        print(f"Validation Loss:      {m['loss']:.4f}")
        print(f"Accuracy:             {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        print(f"Macro F1-Score:       {m['macro_f1']:.4f}")
        if self.roc_data:
            print(f"AUC-ROC:              {self.roc_data['auc']:.4f}")
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
    MODEL_PATH = "model_v5.pkl"  # Path for saving/loading model
    EVALUATE_SPLIT = "test"  # Which split to evaluate on: 'val', or 'test'
    BINS = 32  # Number of histogram bins
    LEARNING_RATE = 0.1  # Learning rate for gradient descent
    MAX_ITERATIONS = 10000  # Maximum training iterations
    L2_LAMBDA = 0.01  # L2 regularization parameter
    
    print("="*60)
    print(" "*15 + "TOMATO CLASSIFIER")
    print("="*60)
    
    # Initialize visualization
    viz = ModelVisualization(bins=BINS, data_split=EVALUATE_SPLIT, 
                            learning_rate=LEARNING_RATE, max_iterations=MAX_ITERATIONS,
                            l2_lambda=L2_LAMBDA)
    
    # Train or load model
    if TRAIN_NEW_MODEL:
        viz.train_model(MODEL_PATH)
        viz.plot_loss()
    else:
        viz.load_model(MODEL_PATH)
    
    # Evaluate model
    viz.evaluate()
    
    # Display results
    viz.print_metrics()
    viz.plot_confusion_matrix()
    viz.plot_roc_curve()


if __name__ == "__main__":
    main()
