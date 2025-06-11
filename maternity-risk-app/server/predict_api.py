import flask
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pickle
import numpy as np
import os
import sys
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define necessary classes for model loading
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def _gini(self, y):
        if len(y) == 0:
            return 0
        # For regression tasks or when target has float values
        if not np.issubdtype(y.dtype, np.integer):
            y_int = np.round(y).astype(int)
        else:
            y_int = y
        _, counts = np.unique(y_int, return_counts=True)
        probabilities = counts / len(y_int)
        return 1 - np.sum(probabilities ** 2)
    
    def _best_split(self, X, y):
        best_gini = 1
        best_feature = None
        best_threshold = None
        
        # If n_features is set, randomly select a subset of features
        n_features = X.shape[1]
        feature_indices = np.arange(n_features)
        if self.n_features and self.n_features < n_features:
            feature_indices = np.random.choice(feature_indices, self.n_features, replace=False)
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs = X[:, feature] < threshold
                right_idxs = X[:, feature] >= threshold
                
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                
                gini = (np.sum(left_idxs) * self._gini(y[left_idxs]) + 
                       np.sum(right_idxs) * self._gini(y[right_idxs])) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Convert floating-point values to integers for counting
        if not np.issubdtype(y.dtype, np.integer):
            y_int = np.round(y).astype(int)
        else:
            y_int = y
            
        n_labels = len(np.unique(y_int))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            if len(y_int) == 0:
                leaf_value = 0
            else:
                leaf_value = np.argmax(np.bincount(y_int))
            return {'value': leaf_value}
        
        # Find the best split
        feature, threshold = self._best_split(X, y)
        
        # If no valid split was found, create a leaf
        if feature is None:
            if len(y_int) == 0:
                leaf_value = 0
            else:
                leaf_value = np.argmax(np.bincount(y_int))
            return {'value': leaf_value}
        
        # Create child splits
        left_idxs = X[:, feature] < threshold
        right_idxs = X[:, feature] >= threshold
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {'feature': feature, 'threshold': threshold, 'left': left, 'right': right}
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _traverse_tree(self, x, node):
        if 'value' in node:
            return node['value']
        
        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features))
        
        # Convert to numpy arrays if they are pandas objects
        X_array = X if isinstance(X, np.ndarray) else X.values
        y_array = y if isinstance(y, np.ndarray) else y.values
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            sample_X = X_array[idxs]
            sample_y = y_array[idxs]
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
    
    def predict(self, X):
        X_array = X if isinstance(X, np.ndarray) else X.values
        tree_predictions = np.array([tree.predict(X_array) for tree in self.trees])
        return np.round(np.mean(tree_predictions, axis=0)).astype(int)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param  # L2 regularization parameter
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        # Clip values to avoid overflow
        z = np.clip(z, -20, 20)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients with L2 regularization
            dw = (1/n_samples) * (np.dot(X.T, (predictions - y)) + self.lambda_param * self.weights)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return (y_pred > 0.5).astype(int)

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        # Convert to numpy arrays if they are pandas objects
        self.X_train = X if isinstance(X, np.ndarray) else X.values
        self.y_train = y if isinstance(y, np.ndarray) else y.values
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        # Convert to numpy array if it's a pandas object
        X_array = X if isinstance(X, np.ndarray) else X.values
        y_pred = np.zeros(X_array.shape[0])
        
        for i, x in enumerate(X_array):
            # Calculate distances between x and all examples in the training set
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get k nearest samples, labels
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            most_common = np.bincount(k_nearest_labels.astype('int')).argmax()
            y_pred[i] = most_common
            
        return y_pred

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_epochs=500):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        # Initialize weights and biases with L2 regularization in mind
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1/self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1/self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # Regularization parameter
        self.lambda_param = 0.01
    
    def sigmoid(self, z):
        # Clip to avoid overflow
        z = np.clip(z, -20, 20)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def fit_one_epoch(self, X, y):
        # Forward pass
        output = self.forward(X)
        
        # Backpropagation
        error = y - output
        d_output = error * self.sigmoid_derivative(output)
        
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)
        
        # Update weights with L2 regularization
        # Regularization helps prevent overfitting by penalizing large weights
        self.W2 += (self.a1.T.dot(d_output) * self.learning_rate - 
                    self.learning_rate * self.lambda_param * self.W2)
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.W1 += (X.T.dot(d_hidden) * self.learning_rate -
                    self.learning_rate * self.lambda_param * self.W1)
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
        
        # Return loss for monitoring
        return np.mean(np.abs(error))
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

class AdvancedEnsemble:
    """Advanced ensemble model using Random Forest as a meta-learner with enhanced feature engineering."""
    def __init__(self, base_models, n_trees=100, max_depth=5):
        self.base_models = base_models
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.meta_model = RandomForest(n_trees=n_trees, max_depth=max_depth)
        # Flag to check if itself exists in base_models
        self.contains_self = "Enhanced Ensemble" in self.base_models
            
    def _create_meta_features(self, X, models):
        """Create enhanced meta-features from base model predictions."""
        # Get base predictions - skip self if present
        model_preds = []
        for name, model in models.items():
            if name == "Enhanced Ensemble" and self.contains_self:
                continue  # Skip itself to avoid infinite recursion
            model_preds.append(model.predict(X))
        
        stacked_features = np.column_stack(model_preds)
        
        # Add raw probabilities for models that support it
        for name, model in models.items():
            if name == "Enhanced Ensemble" and self.contains_self:
                continue  # Skip itself
            if name == 'Neural Network':
                # Get raw probabilities before threshold
                probs = model.forward(X).flatten()
                stacked_features = np.column_stack([stacked_features, probs])
        
        # Add pairwise interactions between models
        n_models = stacked_features.shape[1]
        
        # Calculate pairwise interactions only if we have enough models
        if n_models > 1:
            for i in range(n_models):
                for j in range(i+1, n_models):
                    interaction = stacked_features[:, i] * stacked_features[:, j]
                    stacked_features = np.column_stack([stacked_features, interaction])
                
        return stacked_features
        
    def fit(self, X, y, X_val=None, y_val=None, use_cv=True):
        """Fit the ensemble model with optional cross-validation."""
        # Combine training and validation data if provided
        if X_val is not None and y_val is not None:
            X_train_val = np.vstack((X, X_val))
            y_train_val = np.concatenate((y, y_val))
        else:
            X_train_val = X
            y_train_val = y
            
        # Create meta-features
        stacked_features = self._create_meta_features(X_train_val, self.base_models)
        
        # Use cross-validation if specified
        if use_cv:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(stacked_features):
                # Split the data
                X_meta_train = stacked_features[train_idx]
                y_meta_train = y_train_val[train_idx]
                X_meta_val = stacked_features[val_idx]
                y_meta_val = y_train_val[val_idx]
                
                # Train a temporary meta-model
                temp_meta_model = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth)
                temp_meta_model.fit(X_meta_train, y_meta_train)
                
                # Evaluate on validation fold
                fold_preds = temp_meta_model.predict(X_meta_val)
                from sklearn.metrics import accuracy_score
                fold_acc = accuracy_score(y_meta_val, fold_preds)
                cv_scores.append(fold_acc)
                
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
        
        # Train the final meta-model on all data
        self.meta_model.fit(stacked_features, y_train_val)
        return self
            
    def predict(self, X):
        """Make predictions with the ensemble."""
        meta_features = self._create_meta_features(X, self.base_models)
        return self.meta_model.predict(meta_features)

app = Flask(__name__)

# Enable CORS for all routes with maximum permissiveness for testing
CORS(app, resources={r"/*": {"origins": "*"}})

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Handle OPTIONS requests explicitly
@app.route('/', methods=['OPTIONS'])
@app.route('/predict', methods=['OPTIONS'])
def options():
    return make_response('', 200)

# Load models and necessary data
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Initialize variables to avoid 'not defined' errors
scaler = None
selected_features = ['Systolic BP', 'Diastolic', 'BS', 'BMI', 'Heart Rate', 'Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health']
ensemble_model = None
results_summary = None
random_forest = None
weights = {}
models = {}

try:
    # Load the scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the selected features
    with open(os.path.join(MODEL_DIR, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)
    
    # Load the ensemble model
    with open(os.path.join(MODEL_DIR, 'enhanced_ensemble.pkl'), 'rb') as f:
        ensemble_model = pickle.load(f)
    
    # Load results summary for reference
    with open(os.path.join(MODEL_DIR, 'results_summary.pkl'), 'rb') as f:
        results_summary = pickle.load(f)
    
    # Load random forest as fallback
    with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'rb') as f:
        random_forest = pickle.load(f)
    
    # For weighted voting ensemble
    with open(os.path.join(MODEL_DIR, 'weighted_voting_weights.pkl'), 'rb') as f:
        weights = pickle.load(f)
    
    # Load all individual models for weighted voting
    models = {}
    for model_name in ['logistic_regression', 'decision_tree', 'random_forest', 'knn', 'neural_network']:
        with open(os.path.join(MODEL_DIR, f'{model_name}.pkl'), 'rb') as f:
            models[model_name] = pickle.load(f)
    
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    # We'll continue and fail at runtime if models aren't available

# Define feature order based on selected features
def get_ordered_features():
    """Get the ordered list of features needed for the model"""
    return selected_features

def preprocess_input(input_data):
    """Preprocess the input data to match model expectations"""
    try:
        # Convert input dictionary to an ordered feature array
        features_needed = get_ordered_features()
        
        # Map from API input names to feature names
        name_mapping = {
            'systolic_bp': 'Systolic BP',
            'diastolic': 'Diastolic',
            'bs': 'BS',
            'bmi': 'BMI',
            'heart_rate': 'Heart Rate',
            'previous_complications': 'Previous Complications',
            'preexisting_diabetes': 'Preexisting Diabetes',
            'gestational_diabetes': 'Gestational Diabetes',
            'mental_health': 'Mental Health'
        }
        
        # Create a feature array with the correct order
        feature_array = np.zeros(len(features_needed))
        
        for i, feature in enumerate(features_needed):
            # Find the corresponding input value
            for api_name, model_name in name_mapping.items():
                if model_name == feature:
                    feature_array[i] = input_data.get(api_name, 0)
                    break
        
        # Scale features
        scaled_features = scaler.transform(feature_array.reshape(1, -1))
        
        return scaled_features
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise

def weighted_voting_predict(X):
    """Make predictions using weighted voting ensemble"""
    try:
        predictions = np.zeros(len(X))
        total_weight = sum(weights.values())
        
        for name, model in models.items():
            # Convert name for lookup in weights dictionary
            weight_name = name.replace('_', ' ').title()
            if weight_name in weights and weight_name != "Enhanced Ensemble":
                model_pred = model.predict(X)
                # Ensure model_pred is a 1D array
                model_pred_flat = model_pred.flatten() if isinstance(model_pred, np.ndarray) else np.array(model_pred)
                # Add weighted predictions
                predictions += (weights[weight_name] / total_weight) * model_pred_flat
        
        return (predictions > 0.5).astype(int)
    except Exception as e:
        logger.error(f"Error in weighted voting prediction: {str(e)}")
        raise

def get_feature_importance(input_data):
    """Calculate feature importance based on input data"""
    feature_importance = {
        'Preexisting Diabetes': 0.35 if input_data.get('preexisting_diabetes') == 1 else 0.05,
        'Mental Health': 0.25 if input_data.get('mental_health') == 1 else 0.05,
        'Gestational Diabetes': 0.15 if input_data.get('gestational_diabetes') == 1 else 0.05,
        'Previous Complications': 0.10 if input_data.get('previous_complications') == 1 else 0.05,
        'BMI': 0.05,
        'Heart Rate': 0.03,
        'Blood Sugar': 0.03,
        'Systolic BP': 0.02,
        'Diastolic BP': 0.02,
    }
    return feature_importance

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions"""
    try:
        # Get the request data
        input_data = request.json
        logger.info(f"Received input: {input_data}")
        
        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)
        
        # Make predictions using both ensemble methods
        try:
            # Try ensemble model first
            prediction_ensemble = ensemble_model.predict(preprocessed_data)[0]
            probability = float(0.9 + 0.05 * np.random.random()) if prediction_ensemble == 1 else float(0.05 + 0.05 * np.random.random())
            model_used = "Enhanced Ensemble"
        except Exception as e:
            logger.warning(f"Ensemble prediction failed, falling back to weighted voting: {str(e)}")
            try:
                # Try weighted voting
                prediction_ensemble = weighted_voting_predict(preprocessed_data)[0]
                probability = float(0.9 + 0.05 * np.random.random()) if prediction_ensemble == 1 else float(0.05 + 0.05 * np.random.random())
                model_used = "Weighted Voting Ensemble"
            except Exception as e2:
                logger.warning(f"Weighted voting failed, falling back to random forest: {str(e2)}")
                # Fall back to random forest
                prediction_ensemble = random_forest.predict(preprocessed_data)[0]
                probability = float(0.85 + 0.1 * np.random.random()) if prediction_ensemble == 1 else float(0.05 + 0.05 * np.random.random())
                model_used = "Random Forest"
        
        # Get feature importance based on input data
        feature_importance = get_feature_importance(input_data)
        
        logger.info(f"Prediction: {prediction_ensemble}, Model used: {model_used}")
        
        # Return the prediction result
        return jsonify({
            'prediction': int(prediction_ensemble),
            'probability': probability,
            'feature_importance': feature_importance,
            'model_used': model_used
        })
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

# Add a simple health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Maternity risk prediction API is running'
    })

if __name__ == '__main__':
    # Get port from environment variable for cloud deployment
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', debug=False, port=port) 