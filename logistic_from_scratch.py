import numpy as np


def sigmoid(z):
    """Applies the sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def standardize_features(X):
    """
    Standardize features to have mean 0 and standard deviation 1.
    
    X: input features
    Returns standardized X, mean, and std for later use
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def accuracy_score(y_true, y_pred):
    """Calculates the accuracy of predictions."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='binary'):
    """Calculates precision score."""
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        classes = np.unique(y_true)
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)


def recall_score(y_true, y_pred, average='binary'):
    """Calculates recall score."""
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='binary'):
    """Calculates F1 score."""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """Calculates confusion matrix."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


class LogisticRegression:
    """
    Base class for logistic regression models.
    Provides shared prediction and scoring functionality.
    """
    
    def __init__(self, fit_intercept=True):
        """
        fit_intercept: whether to include an intercept term in the model
        """
        self.fit_intercept = fit_intercept
        self.coefficients_ = None
        self.intercept_ = None
    
    def predict_proba(self, X):
        """
        Generate probability predictions for new data.
        
        X: input features of shape (n_samples, n_features)
        Returns predicted probabilities
        """
        X = np.array(X)
        z = X @ self.coefficients_ + self.intercept_
        return sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Generate class predictions for new data.
        
        X: input features of shape (n_samples, n_features)
        threshold: probability threshold for classification
        Returns predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score for the model predictions.
        
        X: input features of shape (n_samples, n_features)
        y: true target values of shape (n_samples,)
        Returns the accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class LogisticRegressionFromScratch(LogisticRegression):
    """
    Logistic regression using gradient descent optimization.
    Supports binary and multiclass classification.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 tolerance=1e-6, fit_intercept=True, batch_size=None):
        """
        learning_rate: size of each step during optimization
        max_iterations: maximum number of gradient descent iterations
        tolerance: convergence threshold to stop training early
        fit_intercept: whether to include an intercept term
        batch_size: number of samples per batch (None = use all samples)
        """
        super().__init__(fit_intercept)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.cost_history_ = []
    
    def fit(self, X, y, verbose=False):
        """
        Train the model using iterative gradient descent.
        
        X: training features of shape (n_samples, n_features)
        y: target values of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        self.coefficients_ = np.zeros(X.shape[1])
        n_samples = X.shape[0]
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        
        indices = np.arange(n_samples)
        cost = 0
        
        for iteration in range(self.max_iterations):
            np.random.shuffle(indices)
            prev_coef = self.coefficients_.copy()
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                z = X_batch @ self.coefficients_
                predictions = sigmoid(z)
                gradient = (1 / len(batch_indices)) * X_batch.T @ (predictions - y_batch)
                
                self.coefficients_ = self.coefficients_ - self.learning_rate * gradient
            
            z_full = X @ self.coefficients_
            predictions_full = sigmoid(z_full)
            cost = -np.mean(y * np.log(predictions_full + 1e-15) + 
                           (1 - y) * np.log(1 - predictions_full + 1e-15))
            self.cost_history_.append(cost)
            
            if np.linalg.norm(self.coefficients_ - prev_coef) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        if self.fit_intercept:
            self.intercept_ = self.coefficients_[0]
            self.coefficients_ = self.coefficients_[1:]
        else:
            self.intercept_ = 0
    
    def get_batch_type(self):
        """Get type of gradient descent based on batch size."""
        if self.batch_size is None:
            return "Batch (Full Dataset)"
        elif self.batch_size == 1:
            return "Stochastic (SGD)"
        else:
            return f"Mini-Batch ({self.batch_size} samples)"

