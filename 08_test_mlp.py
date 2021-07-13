import numpy as np
import pandas as pd
import sys
from time import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def evaluate(model, test_features, test_labels):
	predictions = model.predict(test_features)
	errors = abs(predictions - test_labels)
	mape = 100 * np.mean(errors / test_labels)
	accuracy = 100 - mape
	print('Model Performance')
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

def mlp_regressor(csvfilename): 

	features =['Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2'] 
	output = 'Vol0'
	target = [output]
	data = pd.read_csv(csvfilename, sep= ';')
	superfeatures = list(features)
	superfeatures.append(output)
	X = data[features]
	y = data[target]
	X_train, X_test, y_train, y_test = train_test_split( X, y,test_size = 0.25)
	X_train = np.array(X_train, dtype = 'float64')
	X_test = np.array(X_test, dtype = 'float64')
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	param_list = {"hidden_layer_sizes": [i for i in range(100,1000,300)], 
	"activation": ["relu", "logistic"],
	"solver": ["adam"],
	"max_iter" : [i for i in range(100,1000,200)] }
	mlp = MLPRegressor(random_state=1)
	grid_search = GridSearchCV(estimator = mlp, param_grid = param_list, cv = 5, n_jobs = -1, verbose = 2)
	grid_search.fit(X_train,y_train)
	print(grid_search.best_params_)
	best_grid = grid_search.best_estimator_
	grid_accuracy = evaluate(best_grid,  X_test, y_test)

if (len(sys.argv) == 2):
 	csvfilename = sys.argv[1]
 	mlp_regressor(csvfilename) 
else:
 	print ('Errore el numero di parametri')
