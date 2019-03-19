from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np

class DecisionTreeClassifier():
    def __init__(self, X=None, y=None, depth=1, min_leaf=10):
        self.X = X
        self.y = y
        self.n = 0 if X is None else X.shape[0]
        self.depth = depth
        self.min_leaf = min_leaf
        self.inner_model = None
        self.lhs = None
        self.rhs = None
        self.threshold = 0.5
        
    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.n = X.shape[0]
    def fit(self, X, y, inner_model=MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,)), 
                        threshold=0.5, split_ratio=None):
        self.inner_model = inner_model
        if X is not None:
            self.set_data(X, y)
            self.threshold = threshold
        if X.shape[0] == 0:
            return
        inner_model.fit(X, y)
        if self.depth == 0 or self.min_leaf >= self.n:
            return
        if split_ratio is None:
            routing_probs = (inner_model.predict_proba(X)[:, 0] >= threshold).astype(int)
            X_left = X[np.where(routing_probs == 0)[0]]
            X_right = X[np.where(routing_probs == 1)[0]]
            y_left = y[np.where(routing_probs == 0)[0]]
            y_right = y[np.where(routing_probs == 1)[0]]
        else:
            routing_probs = inner_model.predict_proba(X)[:, 0]
            ord_indices = np.argsort(routing_probs)
            left_indices = ord_indices[:-int(split_ratio * X.shape[0])]
            right_indices = ord_indices[-int(split_ratio * X.shape[0]):]
            X_left = X[left_indices]
            X_right = X[right_indices]
            y_left = y[left_indices]
            y_right = y[right_indices]
            
        self.lhs = DecisionTreeClassifier(X_left, y_left, self.depth - 1, self.min_leaf)
        self.rhs = DecisionTreeClassifier(X_right, y_right, self.depth - 1, self.min_leaf)
        self.lhs.fit(X_left, y_left, inner_model, threshold, split_ratio)
        self.rhs.fit(X_right, y_right, inner_model, threshold, split_ratio)
        return self.lhs, self.rhs
    
    def predict(self, X):
        return np.array([self.predict_instance(X[i:i + 1]) for i in range(len(X))])
    
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