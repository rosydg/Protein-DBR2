import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

def evaluate(model, test_features, test_labels):
	predictions = model.predict(test_features)
	errors = abs(predictions - test_labels)
	mape = 100 * np.mean(errors / test_labels)
	accuracy = 100 - mape
	print('Model Performance')
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy

def random_forest(csvfilename):
	# Feature settings
	features =['Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2',  'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2'] 
	output = 'Vol0'
	target = [output]
	data = pd.read_csv(csvfilename, sep= ';')
	superfeatures = list(features)
	superfeatures.append(output)
	X = data[features]
	y = data[target]
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 0)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	'''model = RandomForestRegressor() 
	estimators = np.arange(50, 250, 50)
	scores = []
	rms = []
	for n in estimators:
		print('Processing...number of estimators = ',n)
		model.set_params(n_estimators=n)
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		scores.append(model.score(X_test, y_test))
		rms.append(np.sqrt(mean_squared_error(y_test,y_pred)))
	print('Plotting figures...')
	plt.figure()
	plt.title("Effect of n_estimators on score")
	plt.xlabel("n_estimator")
	plt.ylabel("score")
	plt.plot(estimators, scores)
	plt.savefig('test_score_rf.png')
	
	plt.figure()
	plt.title("Effect of n_estimators on root mean squared error")
	plt.xlabel("n_estimator")
	plt.ylabel("Root mean squared error")
	plt.plot(estimators, rms)
	plt.savefig('test_rmse_rf.png')
	best_estimators = estimators[scores.index(max(scores))]
	#best_estimators = 100

	depth = [5,6,7,8,9,10]
	scores_depth = []
	for m in depth:
		print('Processing...max_depth = ',m)
		model.set_params(max_depth=m, n_estimators = best_estimators)
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		scores_depth.append(model.score(X_test, y_test))
		
	print('Plotting figures...')
	plt.figure()
	plt.title("Effect of max_depth on score")
	plt.xlabel("max_depth")
	plt.ylabel("score")
	plt.plot(depth, scores_depth)
	plt.savefig('test_depth_rf.png')'''
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
	# Maximum number of levels in tree
	max_depth = [int(x) for x in range(5, 12)]
	max_depth.append(None)
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
	               'max_depth': max_depth}
	# Fit the random search model
	# Use the random grid to search for best hyperparameters
	# First create the base model to tune
	rf = RandomForestRegressor()
	# Random search of parameters, using 3 fold cross validation, 
	# search across 100 different combinations, and use all available cores
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
	# Fit the random search model
	rf_random.fit(X_train, y_train)
	print(rf_random.best_params_)
	best_random = rf_random.best_estimator_
	random_accuracy = evaluate(best_random, X_test, y_test)

if (len(sys.argv) == 2):
 	csvfilename = sys.argv[1]
 	random_forest(csvfilename)
else:
 	print ('Errore el numero di parametri')
