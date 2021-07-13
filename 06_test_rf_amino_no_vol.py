import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def evaluate(model, test_features, test_labels):
	predictions = model.predict(test_features)
	errors = abs(predictions - test_labels)
	mape = 100 * np.mean(errors / test_labels)
	accuracy = 100 - mape
	print('Model Performance')
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy

def random_forest(amino):
	# Feature settings
	features3 =['Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2',  'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2'] 
	features6 = ['Fun-6', 'Fun-5', 'Fun-4','Fun-3','Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Fun4', 'Fun5', 'Fun6','Chem-6', 'Chem-5', 'Chem-4','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Chem4', 'Chem5', 'Chem6','Hydro-6', 'Hydro-5','Hydro-4','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Hydro4','Hydro5','Hydro6','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2']
	features9= ['Fun-9','Fun-8','Fun-7','Fun-6', 'Fun-5', 'Fun-4','Fun-3','Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Fun4', 'Fun5', 'Fun6','Fun7','Fun8','Fun9','Chem-9','Chem-8','Chem-7','Chem-6', 'Chem-5', 'Chem-4','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Chem4', 'Chem5', 'Chem6','Chem7','Chem8','Chem9','Hydro-9','Hydro-8','Hydro-7','Hydro-6', 'Hydro-5','Hydro-4','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Hydro4','Hydro5','Hydro6','Hydro7','Hydro8','Hydro9', 'Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2']
	output = 'Vol0'
	target = [output]
	csvfilename = "random_"+str(sys.argv[1])+"_forest_"+ str(amino)+".csv"
	data = pd.read_csv(csvfilename, sep= ';')
	if sys.argv[1] == '3':
		features = features3
	elif sys.argv[1]== '6':
		features = features6
	elif sys.argv[1]== '9':
		features = features9
	else:
		print("Errore nel parametro inserito")
		sys.exit()

	superfeatures = list(features)
	superfeatures.append(output)
	X = data[features]
	y = data[target]
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 0)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 100)]
	# Maximum number of levels in tree
	max_depth = [int(x) for x in range(5, 12)]
	max_depth.append(None)
	# Create the param grid
	param_grid = {'n_estimators': n_estimators,
	               'max_depth': max_depth}
	# Create a based model
	rf = RandomForestRegressor()
	# Instantiate the grid search model
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
	grid_search.fit(X_train,y_train)
	print(grid_search.best_params_)
	best_grid = grid_search.best_estimator_
	grid_accuracy = evaluate(best_grid,  X_test, y_test)

if (len(sys.argv) == 3):
 	amino = sys.argv[2]
 	random_forest(amino)
else:
 	print ('Errore el numero di parametri')
