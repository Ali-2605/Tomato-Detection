import numpy as np
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_data

class LogisticRegression:
    """
    Logistic Regression classifier for tomato quality classification (fresh vs rotten)
    Uses color histogram features extracted from YOLO dataset preprocessing
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, bins = 32, l2_lambda=0.01):
        """
        Initialize logistic regression model
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            max_iterations (int): Maximum number of iterations
            l2_lambda (float): L2 regularization parameter
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.bins = bins
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Sigmoid activation function with numerical stability
        
        Args:
            z (numpy.ndarray): Input values
            
        Returns:
            numpy.ndarray: Sigmoid output
        """
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _compute_gradient(self, X, y, w, b):
        # number of training examples
        m = X.shape[0]

        z = np.dot(X, w) + b
        a = self._sigmoid(z)

        error = a - y

        # Add L2 regularization to gradient
        dw = (1 / m) * np.dot(X.T, error) + (self.l2_lambda / m) * w
        db = (1 / m) * np.sum(error)

        return dw, db

    def _compute_loss(self, X, y, w, b):
        """
        Compute logistic loss with L2 regularization
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Labels
            w (numpy.ndarray): Weights
            b (float): Bias
            
        Returns:
            float: Logistic loss with L2 regularization term
        """
        m = X.shape[0]
        z = np.dot(X, w) + b
        a = self._sigmoid(z)

        # Clip a to prevent log(0)
        a = np.clip(a, 1e-15, 1 - 1e-15)

        # Compute cross-entropy loss
        cross_entropy_loss = - (1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        
        # Add L2 regularization term
        l2_term = (self.l2_lambda / (2 * m)) * np.sum(w ** 2)
        
        return cross_entropy_loss + l2_term
    
    def train(self, split="train"):
        """
        Train logistic regression model and return loss history
        
        Returns:
            List of loss values at each iteration
        """
        # Load preprocessed data
        X, y = load_data(split, bins=self.bins)
        
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        loss_history = []
        
        for iteration in range(self.max_iterations):
            dw, db = self._compute_gradient(X, y, self.weights, self.bias)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store loss
            loss = self._compute_loss(X, y, self.weights, self.bias)
            loss_history.append(loss)
            
            # Print loss every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")
        
        return loss_history
    def _predict_proba(self, X):
        """
        Predict probabilities for input features
        Args:
            X (numpy.ndarray): Feature matrix
        Returns:
            numpy.ndarray: Predicted probabilities for class 1 (rotten)
        """
        z = np.dot(X, self.weights) + self.bias
        a = self._sigmoid(z)
        return a
    
    def predict(self, X):
        """
        Predict class labels for input features
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Predicted class labels (0 or 1)
        """

        a = self._predict_proba(X)
        predictions = (a >= 0.5).astype(int)
        return predictions
    
    def evaluate(self, split):
        """
        Evaluate model accuracy on a specific dataset split
        
        Args:
            split (str): Dataset split to evaluate ('train', 'val', or 'test')
            
        Returns:
            float: Accuracy score
        """
        X, y = load_data(split, bins=self.bins)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy on {split} set: {accuracy * 100:.2f}%")
        return accuracy
    
    def compute_metrics(self, X, y):
        """
        Compute comprehensive metrics for predictions
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary containing all metrics
        """
        predictions = self.predict(X)
        
        # Calculate confusion matrix components
        true_fresh = np.sum((y == 0) & (predictions == 0))  # True Negatives
        true_rotten = np.sum((y == 1) & (predictions == 1))  # True Positives
        false_fresh = np.sum((y == 1) & (predictions == 0))  # False Negatives
        false_rotten = np.sum((y == 0) & (predictions == 1))  # False Positives
        
        # Calculate loss
        loss = self._compute_loss(X, y, self.weights, self.bias)
        
        # Calculate metrics for Fresh (class 0)
        precision_fresh = true_fresh / (true_fresh + false_rotten) if (true_fresh + false_rotten) > 0 else 0
        recall_fresh = true_fresh / (true_fresh + false_fresh) if (true_fresh + false_fresh) > 0 else 0
        f1_fresh = 2 * (precision_fresh * recall_fresh) / (precision_fresh + recall_fresh) if (precision_fresh + recall_fresh) > 0 else 0
        
        # Calculate metrics for Rotten (class 1)
        precision_rotten = true_rotten / (true_rotten + false_fresh) if (true_rotten + false_fresh) > 0 else 0
        recall_rotten = true_rotten / (true_rotten + false_rotten) if (true_rotten + false_rotten) > 0 else 0
        f1_rotten = 2 * (precision_rotten * recall_rotten) / (precision_rotten + recall_rotten) if (precision_rotten + recall_rotten) > 0 else 0
        
        # Overall metrics
        accuracy = (true_fresh + true_rotten) / len(y)
        macro_f1 = (f1_fresh + f1_rotten) / 2
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': {
                'true_fresh': true_fresh,
                'true_rotten': true_rotten,
                'false_fresh': false_fresh,
                'false_rotten': false_rotten
            },
            'fresh': {
                'precision': precision_fresh,
                'recall': recall_fresh,
                'f1': f1_fresh
            },
            'rotten': {
                'precision': precision_rotten,
                'recall': recall_rotten,
                'f1': f1_rotten
            },
            'predictions': predictions
        }
    
    def compute_roc_curve(self, X, y, num_thresholds=100):
        """
        Compute ROC curve data points
        
        Args:
            X: Feature matrix
            y: True labels
            num_thresholds: Number of threshold points to compute
            
        Returns:
            Dictionary with fpr_list, tpr_list, and auc
        """
        y_proba = self._predict_proba(X)
        
        thresholds = np.linspace(0, 1, num_thresholds)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            predictions = (y_proba >= threshold).astype(int)
            
            tp = np.sum((y == 1) & (predictions == 1))
            tn = np.sum((y == 0) & (predictions == 0))
            fp = np.sum((y == 0) & (predictions == 1))
            fn = np.sum((y == 1) & (predictions == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Calculate AUC
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sorted_indices = np.argsort(fpr_array)
        fpr_sorted = fpr_array[sorted_indices]
        tpr_sorted = tpr_array[sorted_indices]
        auc = np.trapz(tpr_sorted, fpr_sorted)
        
        return {
            'fpr': fpr_list,
            'tpr': tpr_list,
            'auc': auc
        }

    def save_model(self, path):
        """
        Save model parameters to a file
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'weights': self.weights,
            'bias': self.bias
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path):
        """
        Load model parameters from a file
        
        Args:
            path (str): Path to load the model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
