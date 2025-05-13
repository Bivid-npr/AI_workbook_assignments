from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to  complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object
    
    ax: matplotlib.axes.Axes
        axis
    """
    
    # ====> insert your code below here

    # list of hidden layer widths 1 to 10 (inclusive)
    hidden_layer_width = list(range(1, 11))
    
    # array to hold number of successful runs
    successes = np.zeros(10)  
    
    # array to store epoch for each run
    epochs = np.zeros((10, 10))  
    
    # nest a loop for repetitions inside hidden layer width
    for h_nodes in hidden_layer_width:
        for repetition in range(10):
            # MLP with h_nodes hidden nodes
            xorMLP = MLPClassifier(
                hidden_layer_sizes=(h_nodes),
                max_iter=1000,
                alpha=1e-4,
                solver="sgd",
                #verbose=0,
                learning_rate_init=0.1,
                random_state=repetition
            )
            
            # fit the model to training data
            _ = xorMLP.fit(train_x, train_y)
            
            # measure accuracy
            training_accuracy = 100 * xorMLP.score(train_x, train_y)

            
            # check for 100% accuracy
            if training_accuracy == 100:
                successes[h_nodes-1] += 1
                epochs[h_nodes-1][repetition] = xorMLP.n_iter_
    
    # create efficiency array to either hold 1000 if no runs got 100% accuracy or mean epochs taken for successful runs
    efficiency = np.zeros(10) 
    for i in range(10):
        successful_epochs = epochs[i][epochs[i] > 0] 
        if len(successful_epochs) > 0:
            efficiency[i] = np.mean(successful_epochs)  
        else:
            efficiency[i] = 1000  
    
    # create side-by-side plots 
    fig, ax= plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    # plot Success Rate for Hidden Layer
    ax[0].plot(hidden_layer_width, successes/10, marker='o')
    ax[0].set_xlabel('Hidden Layer Size')
    ax[0].set_ylabel('Success Rate')
    ax[0].set_title('Success Rate for Hidden Layer', fontsize=16)
    ax[0].grid(True)
    
    # plot Training Efficiency for Hidden Layer
    ax[1].plot(hidden_layer_width, efficiency, marker='o')
    ax[1].set_xlabel('Hidden Layer Size')
    ax[1].set_ylabel('Mean Epochs (1000 for no success)')
    ax[1].set_title('Training Efficiency for Hidden Layer', fontsize=16)
    ax[1].grid(True)

    # Adjust layout to prevent overlap
    plt.show()
    # <==== insert your code above here

    return fig, ax

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """ 
    
    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.
        
        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        self.data_x = np.genfromtxt(datafilename, delimiter=',')  
        self.data_y = np.genfromtxt(labelfilename, delimiter=',', dtype=int)  
        # <==== insert your code above here

    def preprocess(self):
        """ Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if ther are more than 2 classes
 
           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, stratify=self.data_y, random_state=12345
        )
        
        scaler = MinMaxScaler()
        self.train_x = scaler.fit_transform(self.train_x)  
        self.test_x = scaler.transform(self.test_x) 
        
        if len(np.unique(self.data_y)) >= 3:
            binarizer = LabelBinarizer()
            self.train_y_onehot = binarizer.fit_transform(self.train_y)
            self.test_y_onehot = binarizer.transform(self.test_y)
        else:
            self.train_y_onehot = self.train_y  
            self.test_y_onehot = self.test_y        
        # <==== insert your code above here
    
    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.
        
        For each of the algorithms KNearest Neighbour, DecisionTreeClassifer and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination, 
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and  lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        
        """
        # ====> insert your code below here
        # using k values from set {1, 3, 5, 7, 9}
        for k in [1, 3, 5, 7, 9]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train_x, self.train_y)
            accuracy = model.score(self.test_x, self.test_y)
            self.stored_models["KNN"].append(model)
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = len(self.stored_models["KNN"]) - 1
        
        # try all combination of max_depth, min_split and min_leaf
        # using max_depth from the set {1,3,5}
        for max_depth in [1, 3, 5]:
            # using min_split from the set {2,5,10}
            for min_split in [2, 5, 10]:
                # using min_leaf from the set {1,5,10}
                for min_leaf in [1, 5, 10]:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y)
                    accuracy = model.score(self.test_x, self.test_y)
                    self.stored_models["DecisionTree"].append(model)
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = len(self.stored_models["DecisionTree"]) - 1
        
        # try all combinations of hidden_layer_sizes and activation
        for nodes1 in [2, 5, 10]:
            for nodes2 in [0, 2, 5]:
                hidden_layers = (nodes1,) if nodes2 == 0 else (nodes1, nodes2)
                for activation in ['logistic', 'relu']:
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        random_state=12345,
                        max_iter=1000,
                        solver='adam'
                    )
                    model.fit(self.train_x, self.train_y_onehot if len(np.unique(self.data_y)) >= 3 else self.train_y)
                    accuracy = model.score(self.test_x, self.test_y_onehot if len(np.unique(self.data_y)) >= 3 else self.test_y)
                    self.stored_models["MLP"].append(model)
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = len(self.stored_models["MLP"]) - 1       
        # <==== insert your code above here
    
    def report_best(self) :
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"
        
        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        # get algorithm with best accuracy
        best_algorithm = max(self.best_accuracy, key=self.best_accuracy.get)
        best_accuracy = self.best_accuracy[best_algorithm]
        best_index = self.best_model_index[best_algorithm]
        best_model = self.stored_models[best_algorithm][best_index]
        
        return best_accuracy, best_algorithm, best_model
        # <==== insert your code above here
