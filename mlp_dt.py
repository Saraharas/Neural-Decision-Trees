import numpy as np
import random

from sklearn.neural_network import MLPClassifier, MLPRegressor

class DecisionTreeClassifier():
    def __init__(self, 
        X=None, 
        y=None, 
        depth=1, 
        min_leaf=10, 
        threshold=0.5, 
        mlp_features_ratio=1.):
        self.X = X
        self.y = y
        self.n = 0 if X is None else X.shape[0]
        self.depth = depth
        self.min_leaf = min_leaf
        self.inner_model = None
        self.lhs = None
        self.rhs = None
        self.threshold = threshold
        self.mlp_features_ratio = mlp_features_ratio
        
    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.n = X.shape[0]
    
    def get_tree_depth():
        return self.depth
    
    def get_tree_min_leaves():
        return self.min_leaf
    
    def fit(self, X, y, inner_model=MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,)), 
                        threshold=0.5, split_ratio=None):
        self.inner_model = inner_model
        if X is not None:
            self.set_data(X, y)
            self.threshold = threshold
            self.mlp_features_indices = np.sort(np.array(
                            random.sample(range(X.shape[1]), k=max(2, int(np.ceil(self.mlp_features_ratio * X.shape[1]))))))
        if X.shape[0] == 0:
            return
        
        if len(y) != self.n:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n))
        
        if self.depth < 0:
            raise ValueError("tree depth must be positive")
        
        inner_model.fit(X[:, self.mlp_features_indices], y)
        if self.depth == 0 or self.min_leaf >= self.n:
            return
        if split_ratio is None:
            routing_probs = (inner_model.predict_proba(X[:, self.mlp_features_indices])[:, 0] >= threshold).astype(int)
            X_left = X[np.where(routing_probs == 0)[0]]
            X_right = X[np.where(routing_probs == 1)[0]]
            y_left = y[np.where(routing_probs == 0)[0]]
            y_right = y[np.where(routing_probs == 1)[0]]
        else:
            routing_probs = inner_model.predict_proba(X[:, self.mlp_features_indices])[:, 0]
            ord_indices = np.argsort(routing_probs)
            left_indices = ord_indices[:-int(split_ratio * X.shape[0])]
            right_indices = ord_indices[-int(split_ratio * X.shape[0]):]
            X_left = X[left_indices]
            X_right = X[right_indices]
            y_left = y[left_indices]
            y_right = y[right_indices]
        
        self.lhs = DecisionTreeClassifier(X=X_left, y=y_left, depth=self.depth - 1, min_leaf=self.min_leaf, mlp_features_ratio=self.mlp_features_ratio)
        self.rhs = DecisionTreeClassifier(X=X_right, y=y_right, depth=self.depth - 1, min_leaf=self.min_leaf, mlp_features_ratio=self.mlp_features_ratio)
        self.lhs.fit(X_left, y_left, inner_model, threshold, split_ratio)
        self.rhs.fit(X_right, y_right, inner_model, threshold, split_ratio)
    
    def predict(self, X):
        return np.array([self.predict_instance(X[i:i + 1, self.mlp_features_indices]) for i in range(len(X))])
    
    def predict_instance(self, X):
        output = self.inner_model.predict(X)
        if self.depth == 1 or (self.lhs is None and self.rhs is None):
            return output
        if output <= self.threshold:
            return self.lhs.predict_instance(X)
        else:
            return self.rhs.predict_instance(X)
    
    def print_tree(self, depth = 0):
        if self is None:
            return
        print(self.X.shape[0], depth)
        if self.rhs is not None:
            self.lhs.print_tree(depth + 1)
        if self.rhs is not None:
            self.rhs.print_tree(depth + 1)