import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from sklearn.model_selection import KFold
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load and explore data
def load_and_explore_data():
    print("Loading and exploring data...")
    # Load the dataset
    df = pd.read_csv('Maternity_health_risk_dataset.csv')
    
    # Basic information about the dataset
    print("\nDataset Shape:", df.shape)
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\nBasic statistics of numerical columns:")
    print(df.describe())
    
    return df

# Step 2: Data Cleaning
def clean_data(df):
    print("\nCleaning data...")
    df_clean = df.copy()
    
    # Fix extreme age outliers (like 325) that are likely data entry errors
    # Using domain knowledge that maternal age is typically between 10-60
    df_clean = df_clean[df_clean['Age'] <= 60]
    
    # Handle missing values in BMI
    df_clean['BMI'] = df_clean.groupby('Risk Level')['BMI'].transform(
        lambda x: x.fillna(x.median()))
    
    # Remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    numerical_cols = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Heart Rate']
    for col in numerical_cols:
        df_clean = remove_outliers(df_clean, col)
    
    # Convert categorical variables
    df_clean['Risk Level'] = df_clean['Risk Level'].map({'Low': 0, 'High': 1})
    
    return df_clean

def plot_feature_importance(model, X_columns):
    """Plot feature importance from Logistic Regression."""
    print("\nAnalyzing feature importance...")
    
    # Handle different model types
    if isinstance(model, LogisticRegression):
        # Get feature importance (weights)
        importance = np.abs(model.weights)
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': X_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Print importance values
        print("\nFeature importance:")
        for i, row in feature_importance.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance from Logistic Regression')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.close()
        
        return feature_importance
    elif isinstance(model, AdvancedEnsemble):
        # For ensemble models, we'll analyze meta-model feature importance
        print("Ensemble model detected - feature importance not directly available")
        return None
    else:
        print("Feature importance analysis only available for Logistic Regression model")
        return None

def select_features(X, feature_importance, importance_threshold=0.05):
    """Select features based on importance threshold."""
    print(f"\nPerforming feature selection with threshold {importance_threshold}...")
    
    # Get features that meet the threshold
    selected_features = feature_importance[feature_importance['Importance'] >= importance_threshold]['Feature'].tolist()
    
    # Print selected and dropped features
    print(f"Selected {len(selected_features)} features:")
    for feature in selected_features:
        print(f"- {feature}")
    
    print(f"\nDropped {len(X.columns) - len(selected_features)} features:")
    for feature in X.columns:
        if feature not in selected_features:
            print(f"- {feature}")
    
    # Return selected features subset of X
    return X[selected_features], selected_features

def plot_correlation_with_target(df_cleaned):
    """Plot correlation of features with target variable."""
    print("\nAnalyzing correlation with target...")
    
    # Calculate correlations
    corr_with_target = df_cleaned.corr()['Risk Level'].sort_values(ascending=False)
    
    # Print correlations
    print("\nCorrelation with Risk Level:")
    for feature, corr in corr_with_target.iloc[1:].items():
        print(f"{feature}: {corr:.4f}")
    
    # Create DataFrame for visualization
    corr_data = pd.DataFrame({
        'Feature': corr_with_target.index[1:],
        'Correlation': corr_with_target.values[1:]
    })
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlation', y='Feature', data=corr_data)
    plt.title('Correlation with Risk Level')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_with_target.png')
    plt.close()

def plot_overfitting_analysis(models, X_train, y_train, X_val, y_val, X_test, y_test):
    """Analyze potential overfitting and adjust models if necessary."""
    print("\nAnalyzing potential overfitting...")
    
    results = {}
    
    # For each model
    for name, model in models.items():
        # Make predictions
        try:
            # For Neural Network
            if name == 'Neural Network':
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                test_preds = model.predict(X_test)
            # For other models
            else:
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                test_preds = model.predict(X_test)
                
            # Calculate accuracies
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            test_acc = accuracy_score(y_test, test_preds)
            
            # Store results
            results[name] = {
                'train': train_acc,
                'val': val_acc,
                'test': test_acc,
                'diff': train_acc - val_acc  # Proxy for overfitting
            }
            
            # Check for significant overfitting (more than 5% difference)
            if train_acc - val_acc > 0.05:
                print(f"WARNING: {name} shows signs of overfitting. Train-Val diff: {train_acc-val_acc:.4f}")
            else:
                print(f"{name}: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}, Diff={train_acc-val_acc:.4f}")
        except:
            # Skip if model prediction fails
            print(f"Skipping {name} in overfitting analysis")
    
    return results

# Step 4: Model Classes
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
                fold_acc = accuracy_score(y_meta_val, fold_preds)
                cv_scores.append(fold_acc)
                
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
        
        # Train the final meta-model on all data
        self.meta_model.fit(stacked_features, y_train_val)
        return self
            
    def predict(self, X):
        """Make predictions with the ensemble."""
        meta_features = self._create_meta_features(X, self.base_models)
        return self.meta_model.predict(meta_features)

# Step 5: Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance and print metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return accuracy

def create_confusion_matrices(models, X_test, y_test, risk_levels):
    """Create and save confusion matrices for all models."""
    plt.figure(figsize=(20, 15))
    
    # Filter out None models and count valid models
    valid_models = {name: model for name, model in models.items() if model is not None}
    model_count = len(valid_models)
    
    # Set up subplot grid based on number of valid models
    rows = (model_count + 2) // 3  # Calculate rows needed (ceiling division)
    
    for i, (name, model) in enumerate(valid_models.items(), 1):
        try:
            plt.subplot(rows, 3, i)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=risk_levels,
                      yticklabels=risk_levels)
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        except Exception as e:
            print(f"Error creating confusion matrix for {name}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png')
    plt.close()

# Step 6: Main Function
def main():
    # Load and explore data
    df = load_and_explore_data()
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Create directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Create directory for saved models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Split data into features and target
    X = df_cleaned.drop('Risk Level', axis=1)
    y = df_cleaned['Risk Level']
    
    # Initial data split - save for final comparison
    X_train_initial, X_test_initial, y_train_initial, y_test_initial = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale initial features
    scaler_initial = StandardScaler()
    X_train_initial_scaled = scaler_initial.fit_transform(X_train_initial)
    
    # Save the initial scaler
    with open('models/scaler_initial.pkl', 'wb') as f:
        pickle.dump(scaler_initial, f)
    
    # Train a logistic regression model for feature importance
    lr_model = LogisticRegression(learning_rate=0.01, n_iterations=20000, lambda_param=0.01)
    lr_model.fit(X_train_initial_scaled, y_train_initial)
    
    # Plot feature importance and get important features
    feature_importance = plot_feature_importance(lr_model, X.columns)
    
    # Select features based on importance
    X_selected, selected_features = select_features(X, feature_importance)
    
    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for selected features
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the selected features list
    with open('models/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    # Create and train models
    models = {
        'Logistic Regression': LogisticRegression(learning_rate=0.01, n_iterations=20000, lambda_param=0.01),
        'Decision Tree': DecisionTree(max_depth=5),
        'Random Forest': RandomForest(n_trees=25, max_depth=7, min_samples_split=3),
        'KNN': KNN(k=5),
        'Neural Network': NeuralNetwork(
            input_size=X_train_scaled.shape[1], 
            hidden_size=8, 
            output_size=1,
            learning_rate=0.01,
            n_epochs=1000
        )
    }
    
    # Store validation and test predictions
    val_preds = {}
    test_preds = {}
    model_results = {}
    
    # Train each model and make predictions
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Special handling for Neural Network
        if name == 'Neural Network':
            X_train_nn = X_train_scaled.copy()
            y_train_nn = y_train.values.reshape(-1, 1)
            X_val_nn = X_val_scaled.copy()
            y_val_nn = y_val.values.reshape(-1, 1)
            
            val_loss_history = []
            train_loss_history = []
            best_val_loss = float('inf')
            patience = 100
            no_improve_count = 0
            
            for epoch in range(model.n_epochs):
                train_loss = model.fit_one_epoch(X_train_nn, y_train_nn)
                train_loss_history.append(train_loss)
                
                # Calculate validation loss
                val_preds_prob = model.forward(X_val_nn)
                val_loss = np.mean(-y_val_nn * np.log(val_preds_prob + 1e-10) - (1 - y_val_nn) * np.log(1 - val_preds_prob + 1e-10))
                val_loss_history.append(val_loss)
                
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            # Regular training for other models
            model.fit(X_train_scaled, y_train)
        
        # Store predictions on validation and test sets
        val_preds[name] = model.predict(X_val_scaled)
        test_preds[name] = model.predict(X_test_scaled)
        test_acc = evaluate_model(y_test, test_preds[name], name)
        
        # Store results
        model_results[name] = test_acc
        
        # Save individual models
        with open(f'models/{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Check for overfitting before proceeding
    overfitting_results = plot_overfitting_analysis(models, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # Print accuracy on validation set for each model
    print("\nValidation set accuracy:")
    val_accuracies = {}
    for name, preds in val_preds.items():
        val_acc = accuracy_score(y_val, preds)
        val_accuracies[name] = val_acc
        print(f"{name}: {val_acc:.4f}")
    
    # Create enhanced ensemble model
    print("\nCreating and evaluating enhanced ensemble model...")
    
    # Make a copy of the base models to avoid self-reference issues
    base_models_copy = {name: model for name, model in models.items()}
    
    # Initialize and train advanced ensemble
    ensemble_model = AdvancedEnsemble(base_models=base_models_copy, n_trees=100, max_depth=5)
    ensemble_model.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val)
    
    # Save the ensemble model
    with open('models/enhanced_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    # Make ensemble predictions
    try:
        ensemble_preds = ensemble_model.predict(X_test_scaled)
        ensemble_acc = evaluate_model(y_test, ensemble_preds, "Enhanced Ensemble")
        
        # Add ensemble result to model_results
        model_results["Enhanced Ensemble"] = ensemble_acc
        
        # Add ensemble model to models dictionary
        models["Enhanced Ensemble"] = ensemble_model
    except Exception as e:
        print(f"Error with ensemble model: {str(e)}")
        print("Continuing without ensemble model...")
    
    # Create weighted voting ensemble 
    print("\nCreating and evaluating weighted voting ensemble...")
    
    # Use validation accuracies as weights for models
    weights = val_accuracies
    
    # Create a model that makes predictions by weighted voting
    def weighted_voting_predict(X):
        # Initialize with zeros
        predictions = np.zeros(len(X))
        total_weight = sum(weights.values())
        
        for name, model in models.items():
            if name != "Enhanced Ensemble":  # Exclude enhanced ensemble to avoid circularity
                model_pred = model.predict(X)
                # Ensure model_pred is a 1D array
                model_pred_flat = model_pred.flatten() if isinstance(model_pred, np.ndarray) else np.array(model_pred)
                # Add weighted predictions
                predictions += (weights[name] / total_weight) * model_pred_flat
        
        return (predictions > 0.5).astype(int)
    
    # Evaluate weighted voting ensemble
    try:
        weighted_voting_preds = weighted_voting_predict(X_test_scaled)
        weighted_voting_acc = evaluate_model(y_test, weighted_voting_preds, "Weighted Voting Ensemble")
        
        # Add weighted voting ensemble to model_results
        model_results["Weighted Voting Ensemble"] = weighted_voting_acc
        
        # Save the weighted voting ensemble information
        with open('models/weighted_voting_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
    except Exception as e:
        print(f"Error with weighted voting ensemble: {str(e)}")
        print("Continuing without weighted voting ensemble...")
    
    # Analyze correlation for selected features
    df_selected = df_cleaned[selected_features + ['Risk Level']]
    plot_correlation_with_target(df_selected)
    
    # Plot feature importance for final logistic regression model
    _ = plot_feature_importance(models['Logistic Regression'], selected_features)
    
    try:
        create_confusion_matrices(models, X_test_scaled, y_test, ['Low', 'High'])
    except Exception as e:
        print(f"Error creating confusion matrices: {str(e)}")
    
    # Plot model comparison
    plt.figure(figsize=(10, 6))
    models_df = pd.DataFrame({'Model': list(model_results.keys()), 'Accuracy': list(model_results.values())})
    models_df = models_df.sort_values('Accuracy', ascending=True)  # Changed to ascending for increasing order
    
    sns.barplot(x='Accuracy', y='Model', data=models_df)
    plt.title('Model Accuracy Comparison (Increasing Order)')
    plt.xlim(0.9, 1.0)  # Focus on the high accuracy range
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()
    
    # Print final model comparison in increasing order
    print("\nFinal Model Comparison (increasing order of accuracy):")
    for name, accuracy in sorted(model_results.items(), key=lambda x: x[1]):
        print(f"{name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compare with initial model
    print("\nComparing models with all features vs. selected features:")
    
    # Train a model with all features for comparison
    print("\nTraining Random Forest with all features for comparison...")
    all_features_model = RandomForest(n_trees=25, max_depth=7, min_samples_split=3)
    all_features_model.fit(X_train_initial_scaled, y_train_initial)
    all_features_preds = all_features_model.predict(scaler_initial.transform(X_test_initial))
    all_features_acc = accuracy_score(y_test_initial, all_features_preds)
    
    print(f"Random Forest with all features: {all_features_acc:.4f} ({all_features_acc*100:.2f}%)")
    print(f"Random Forest with selected features: {model_results['Random Forest']:.4f} ({model_results['Random Forest']*100:.2f}%)")
    
    # Find and print the best model
    best_model = max(model_results.items(), key=lambda x: x[1])
    print(f"\nBest performing model: {best_model[0]} with accuracy of {best_model[1]:.4f} ({best_model[1]*100:.2f}%)")
    
    # Save final results for reference
    results_summary = {
        'model_results': model_results,
        'selected_features': selected_features,
        'best_model': best_model[0]
    }
    
    with open('models/results_summary.pkl', 'wb') as f:
        pickle.dump(results_summary, f)

if __name__ == "__main__":
    main()
