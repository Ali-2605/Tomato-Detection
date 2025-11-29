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
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, dataset_path="../preprocessing/preprocessed_data_train.pkl", bins = 32):
        """
        Initialize logistic regression model
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance for early stopping
        """
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.bins = bins
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
        
        # weighted error to handle class imbalance
        class_weights = np.where(y == 1, 1.8, 1.0)
        weighted_error = error * class_weights

        dw = (1 / m) * np.dot(X.T, weighted_error)
        db = (1 / m) * np.sum(weighted_error)

        return dw, db

    def _compute_loss(self, X, y, w, b):
        """
        Compute logistic loss
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Labels
            w (numpy.ndarray): Weights
            b (float): Bias
            
        Returns:
            float: Logistic loss
        """
        m = X.shape[0]
        z = np.dot(X, w) + b
        a = self._sigmoid(z)

        # Clip a to prevent log(0)
        a = np.clip(a, 1e-15, 1 - 1e-15)

        return - (1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    
    def train(self, split="train"):
        """
        Train logistic regression model using gradient descent
        """
        # Load preprocessed data
        X, y = load_data(split, bins=self.bins)
        
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for iteration in range(self.max_iterations):
            dw, db = self._compute_gradient(X, y, self.weights, self.bias)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Optional: Print loss every 100 iterations
            if iteration % 100 == 0:
                loss = self._compute_loss(X, y, self.weights, self.bias)
                print(f"Iteration {iteration}, Loss: {loss:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probabilities for input features
        
        Args:
            X (numpy.ndarray): Feature matrix
        Returns:
            numpy.ndarray: Predicted probabilities for class 1
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

        a = self.predict_proba(X)
        predictions = (a >= 0.5).astype(int)
        return predictions
    
    def evaluate(self, split="val"):
        """
        Evaluate model accuracy on a specific dataset split
        
        Args:
            split (str): Dataset split to evaluate ('train', 'val', or 'test')
            
        Returns:
            float: Accuracy score
        """
        X, y = load_data(split)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy on {split} set: {accuracy * 100:.2f}%")
        return accuracy

    
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
