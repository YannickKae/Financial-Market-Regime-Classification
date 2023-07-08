#The problem with using the BayesSearchCV function from the scikit-optimize library with the MLPClassifier function from scikit-learn is that the MLPClassifier takes the number of neurons and hidden layers as a tuple (for example, (50, 50) for two hidden layers with 50 neurons each). However, the BayesSearchCV function cannot optimize tuples as parameters. To get around this problem, we need to use a little trick.

We implemented a wrapper class for the MLPClassifier that takes the number of neurons and hidden layers as separate decoubled parameters. By soing so, we can use BayesSearchCV, which we definittely want.

class MLPWrapper(MLPClassifier):
    def __init__(self, layer_size=100, num_layers=1, alpha=0.0001, activation='relu'):
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.alpha = alpha
        self.activation = activation
        self.hidden_layer_sizes = tuple([self.layer_size]*self.num_layers)
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, 
                                   alpha=self.alpha, 
                                   activation=self.activation,
                                   solver='lbfgs', 
                                   learning_rate='adaptive', 
                                   max_iter=100000, 
                                   warm_start=True,
                                   early_stopping=True)
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == "layer_size":
                self.layer_size = value
            elif parameter == "num_layers":
                self.num_layers = value
            elif parameter == "alpha":
                self.alpha = value
            elif parameter == "activation":
                self.activation = value
            else:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (parameter, self))
        self.hidden_layer_sizes = tuple([self.layer_size]*self.num_layers)
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, 
                                   alpha=self.alpha, 
                                   activation=self.activation,
                                   solver='lbfgs', 
                                   learning_rate='adaptive', 
                                   max_iter=100000, 
                                   warm_start=True,
                                   early_stopping=True)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        y_pred_proba = self.model.predict_proba(X)
        return -log_loss(y, y_pred_proba, labels=[0.0, 1.0])
