import numpy as np
import pickle
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_data

class KMeansCluster:
    """
    KMeans clustering for tomato quality classification (fresh vs rotten)
    Uses color histogram features extracted from YOLO dataset preprocessing
    """
    
    def __init__(self, k=2, bins=32, use_normalization=True, normalization_type='standard'):
        """
        Initialize KMeans model
        
        Args:
            k (int): Number of clusters
            bins (int): Number of histogram bins for feature extraction
            use_normalization (bool): Whether to normalize features
            normalization_type (str): 'standard' (zero mean, unit variance) or 'minmax' (0-1 range)
        """
        self.bins = bins
        self.k = k
        self.kmeans = KMeans(
            n_clusters=k, 
            n_init=50,           # More initializations for better results
            max_iter=300,        # More iterations
            random_state=42,
            algorithm='elkan'    # Use full EM-style algorithm
        )
        self.cluster_to_label = {}
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type
        self.scaler = None
        self.inertia_history = []  # Track inertia across training

    def _normalize_features(self, X, fit=False):
        """
        Normalize features for better clustering
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Normalized feature matrix
        """
        if not self.use_normalization:
            return X
            
        if fit:
            if self.normalization_type == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Train the model first.")
            return self.scaler.transform(X)

    def train(self, split="train"):
        """
        Train KMeans model
        
        Args:
            split (str): Dataset split to train on
        """
        # Load preprocessed data
        X, y = load_data(split, bins=self.bins)
        
        # Normalize features
        X_normalized = self._normalize_features(X, fit=True)
        
        # Track inertia by running KMeans incrementally
        self.inertia_history = []
        for n_iter in range(1, 51):  # Track up to 50 iterations
            temp_kmeans = KMeans(
                n_clusters=self.k,
                n_init=1,
                max_iter=n_iter,
                random_state=42,
                algorithm='elkan'
            )
            temp_kmeans.fit(X_normalized)
            self.inertia_history.append(temp_kmeans.inertia_)
        
        # Fit final KMeans with full iterations
        self.kmeans.fit(X_normalized)
        clusters = self.kmeans.labels_

        # Map each cluster to the most common label in that cluster
        for cluster_id in range(self.k):
            labels_in_cluster = y[clusters == cluster_id]
            if len(labels_in_cluster) == 0:
                self.cluster_to_label[cluster_id] = 0
            else:
                counts = np.bincount(labels_in_cluster, minlength=2)
                self.cluster_to_label[cluster_id] = int(np.argmax(counts))
        
        # Print training info
        print(f"\nKMeans Training Complete:")
        print(f"  - Number of clusters: {self.k}")
        print(f"  - Inertia (within-cluster sum of squares): {self.kmeans.inertia_:.2f}")
        print(f"  - Number of iterations: {self.kmeans.n_iter_}")

    def predict(self, X):
        """
        Predict class labels for input features
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Predicted class labels (0 or 1)
        """
        X_normalized = self._normalize_features(X, fit=False)
        clusters = self.kmeans.predict(X_normalized)
        return np.array([self.cluster_to_label[c] for c in clusters])
    
    def get_cluster_composition(self, X, y):
        """
        Count fresh/rotten samples inside each cluster
        """
        X_normalized = self._normalize_features(X, fit=False)
        clusters = self.kmeans.predict(X_normalized)
        composition = {}

        for c in range(self.k):
            mask = clusters == c
            fresh = np.sum((mask) & (y == 0))
            rotten = np.sum((mask) & (y == 1))
            total = fresh + rotten
            
            # Calculate purity
            purity = max(fresh, rotten) / total if total > 0 else 0

            composition[c] = {
                "fresh": int(fresh),
                "rotten": int(rotten),
                "total": int(total),
                "purity": purity,
                "assigned_label": self.cluster_to_label.get(c, -1)
            }

        return composition

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
        Compute classification metrics
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True labels
            
        Returns:
            dict: Dictionary containing accuracy, precision, recall, f1, and confusion matrix
        """
        y_pred = self.predict(X)
        y_true = y
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true)

        # Rotten (class 1) metrics
        precision_rotten = tp / (tp + fp + 1e-9)
        recall_rotten = tp / (tp + fn + 1e-9)
        f1_rotten = 2 * precision_rotten * recall_rotten / (precision_rotten + recall_rotten + 1e-9)

        # Fresh (class 0) metrics
        precision_fresh = tn / (tn + fn + 1e-9)
        recall_fresh = tn / (tn + fp + 1e-9)
        f1_fresh = 2 * precision_fresh * recall_fresh / (precision_fresh + recall_fresh + 1e-9)

        macro_f1 = (f1_fresh + f1_rotten) / 2

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "loss": 0.0,  # KMeans doesn't have a loss in the same sense
            "fresh": {
                "precision": precision_fresh,
                "recall": recall_fresh,
                "f1": f1_fresh
            },
            "rotten": {
                "precision": precision_rotten,
                "recall": recall_rotten,
                "f1": f1_rotten
            },
            "confusion_matrix": {
                "true_fresh": int(tn),
                "true_rotten": int(tp),
                "false_fresh": int(fn),
                "false_rotten": int(fp)
            }
        }

    def get_inertia_history(self):
        """
        Get the inertia history from training
        
        Returns:
            list: Inertia values at each iteration
        """
        return self.inertia_history

    def save_model(self, path):
        """
        Save model parameters to a file
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'kmeans': self.kmeans,
            'cluster_to_label': self.cluster_to_label,
            'k': self.k,
            'bins': self.bins,
            'scaler': self.scaler,
            'use_normalization': self.use_normalization,
            'normalization_type': self.normalization_type,
            'inertia_history': self.inertia_history
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
        self.kmeans = model_data['kmeans']
        self.cluster_to_label = model_data['cluster_to_label']
        self.k = model_data['k']
        self.bins = model_data['bins']
        self.scaler = model_data.get('scaler', None)
        self.use_normalization = model_data.get('use_normalization', False)
        self.normalization_type = model_data.get('normalization_type', 'standard')
        self.inertia_history = model_data.get('inertia_history', [])
