import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn import tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
from subprocess import call


def evaluate(model, y_pred, y_test):
	errors = abs(y_pred - y_test)
	mape = 100 * np.mean(errors / y_test)
	accuracy = 100 - mape
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy 


def random_forest(csvfilename):
	features =['Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2',  'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2'] 
	output = 'Vol0'
	target = [output]
	data = pd.read_csv(csvfilename, sep= ';')
	superfeatures = list(features)
	superfeatures.append(output)
	X = data[features]
	y = data[target]
	print("Start training RandomForestRegressor...")
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 0)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	regressor = RandomForestRegressor(n_estimators = 200) 
	regressor.fit(X_train, y_train)
	print("Prediction on test set...")
	y_pred = regressor.predict(X_test)
	#accuracy = evaluate(regressor, y_pred, y_test)
	# mean absolute error
	mae = metrics.mean_absolute_error(y_test,y_pred) 
	# mean squared error
	mse = metrics.mean_squared_error(y_test,y_pred)
	# root means square
	rms = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
	print("Means absolute error : ", mae)
	print("Means squared error : ", mse)
	print("Root means squared error : ", rms)
	'''feature_imp = pd.Series(regressor.feature_importances_, index=features).sort_values(ascending = False)
	sns.barplot(x = feature_imp, y = feature_imp.index)
	plt.xlabel('Punteggio di importanza delle caratteristiche')
	plt.ylabel('Caratteristiche')
	plt.title('Visualizzazione caratteristiche più importanti')
	plt.tight_layout()
	plt.savefig('Features_rf_no_pos.png')
	print("End creating Feauture importances plot")
	depth=[]
	for i in range(1,200):
		depth.append(regressor.estimators_[i].tree_.max_depth)

	print("Max_depth min = ", min(depth))
	ind = depth.index(min(depth))
	
	export_graphviz(regressor.estimators_[ind], out_file = "tree_tot_no_pos.dot",feature_names = features, rounded = True, proportion = 2, filled = True)
	call(["dot", "-Tsvg", "tree_tot_no_pos.dot", "-o", "tree_tot_no_pos.svg", "-v"])'''
if (len(sys.argv) == 2):
 	csvfilename = sys.argv[1]
 	random_forest(csvfilename)
 	
else:
 	print ('Errore')
 	

