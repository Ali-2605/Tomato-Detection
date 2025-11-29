import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_data
from LogisticRegression import LogisticRegression

class ModelVisualization:
    def __init__(self):
        # Load dataset splits
        self.X_train, self.y_train = load_data("train")
        self.X_val, self.y_val = load_data("val")
        
        self.model = LogisticRegression(learning_rate=0.05, max_iterations=10000)
        self.predictions = None
        self.version_counter = 3
        
    def train_model(self):
        """Train logistic regression model on training data"""
        print("Training model...")
        
        # Check training data distribution
        print(f"Training set distribution - Fresh (0): {np.sum(self.y_train == 0)}, Rotten (1): {np.sum(self.y_train == 1)}")
        
        self.model.train("train")
        
        # Save model with version counter
        model_path = f"model_v{self.version_counter}.pkl"
        self.model.save_model(model_path)
        print(f"Model saved as {model_path}")
        self.version_counter += 1
        
    def predict(self):
        """Make predictions on validation data"""
        print("Making predictions on validation set...")
        self.predictions = self.model.predict(self.X_val)
        
        
    def plot_loss(self):
        """Plot training loss over epochs (simulated)"""
        # Simulate fake loss values for demonstration
        epochs = np.arange(1, 101)
        fake_loss = 0.7 * np.exp(-epochs/30) + 0.1 + np.random.normal(0, 0.02, len(epochs))
        
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, fake_loss, 'b-', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        if self.predictions is None:
            print("No predictions available. Run predict() first.")
            return
            
        # Calculate confusion matrix components
        # Fresh = 0, Rotten = 1
        true_fresh = np.sum((self.y_val == 0) & (self.predictions == 0))  # True Negatives
        true_rotten = np.sum((self.y_val == 1) & (self.predictions == 1))  # True Positives
        false_fresh = np.sum((self.y_val == 1) & (self.predictions == 0))  # False Negatives
        false_rotten = np.sum((self.y_val == 0) & (self.predictions == 1))  # False Positives
        
        # Create confusion matrix
        confusion_matrix = np.array([[true_fresh, false_rotten],
                                   [false_fresh, true_rotten]])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix: Fresh vs Rotten Tomatoes')
        plt.colorbar()
        
        # Add labels
        classes = ['Fresh (0)', 'Rotten (1)']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black",
                        fontsize=20)
        
        # Add accuracy info
        total = len(self.y_val)
        accuracy = (true_fresh + true_rotten) / total * 100
        plt.text(0.02, 0.98, f'Total: {total}\nAccuracy: {accuracy:.1f}%', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat'), 
                verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def print_model_metrics(self):
        """Print model performance metrics: loss, F1-score, and accuracy"""
        if self.predictions is None:
            print("No predictions available. Run predict() first.")
            return
            
        # Calculate final loss on validation set
        final_loss = self.model._compute_loss(self.X_val, self.y_val, self.model.weights, self.model.bias)
        
        # Calculate confusion matrix components
        true_fresh = np.sum((self.y_val == 0) & (self.predictions == 0))  # True Negatives
        true_rotten = np.sum((self.y_val == 1) & (self.predictions == 1))  # True Positives
        false_fresh = np.sum((self.y_val == 1) & (self.predictions == 0))  # False Negatives
        false_rotten = np.sum((self.y_val == 0) & (self.predictions == 1))  # False Positives
        
        # Calculate metrics for each class
        # For Fresh (class 0)
        precision_fresh = true_fresh / (true_fresh + false_rotten) if (true_fresh + false_rotten) > 0 else 0
        recall_fresh = true_fresh / (true_fresh + false_fresh) if (true_fresh + false_fresh) > 0 else 0
        f1_fresh = 2 * (precision_fresh * recall_fresh) / (precision_fresh + recall_fresh) if (precision_fresh + recall_fresh) > 0 else 0
        
        # For Rotten (class 1)
        precision_rotten = true_rotten / (true_rotten + false_fresh) if (true_rotten + false_fresh) > 0 else 0
        recall_rotten = true_rotten / (true_rotten + false_rotten) if (true_rotten + false_rotten) > 0 else 0
        f1_rotten = 2 * (precision_rotten * recall_rotten) / (precision_rotten + recall_rotten) if (precision_rotten + recall_rotten) > 0 else 0
        
        # Overall metrics
        accuracy = (true_fresh + true_rotten) / len(self.y_val)
        macro_f1 = (f1_fresh + f1_rotten) / 2
        
        # Print results
        print("\n" + "="*50)
        print("         MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Final Validation Loss: {final_loss:.4f}")
        print(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1-Score:       {macro_f1:.4f}")
        print()
        print("Class-wise Metrics:")
        print("-" * 30)
        print(f"Fresh Tomatoes (Class 0):")
        print(f"  Precision: {precision_fresh:.4f}")
        print(f"  Recall:    {recall_fresh:.4f}")
        print(f"  F1-Score:  {f1_fresh:.4f}")
        print()
        print(f"Rotten Tomatoes (Class 1):")
        print(f"  Precision: {precision_rotten:.4f}")
        print(f"  Recall:    {recall_rotten:.4f}")
        print(f"  F1-Score:  {f1_rotten:.4f}")
        print("="*50)

if __name__ == "__main__":
    # Create visualization instance
    viz = ModelVisualization()
    
    # Train model
    viz.train_model()
    
    # Make predictions
    viz.predict()
    
    # Show plots
    viz.plot_loss()
    viz.plot_confusion_matrix()
    
    # Print final metrics
    viz.print_model_metrics()
