#!/usr/bin/python3
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn import tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image
import pydot
import seaborn as sns
from subprocess import call

def evaluate(model, y_pred, y_test):
	errors = abs(y_pred - y_test)
	mape = 100 * np.mean(errors / y_test)
	accuracy = 100 - mape
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy 

def random_forest(csvfilename): 
	# Feature settings
	features3 =['Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2',  'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2'] 
	features6 = ['Vol-6', 'Vol-5', 'Vol-4','Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Vol4', 'Vol5', 'Vol6', 'Fun-6', 'Fun-5', 'Fun-4','Fun-3','Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Fun4', 'Fun5', 'Fun6','Chem-6', 'Chem-5', 'Chem-4','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Chem4', 'Chem5', 'Chem6','Hydro-6', 'Hydro-5','Hydro-4','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Hydro4','Hydro5','Hydro6','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2']
	features9= ['Vol-9', 'Vol-8', 'Vol-7','Vol-6', 'Vol-5', 'Vol-4','Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Vol4', 'Vol5', 'Vol6','Vol7', 'Vol8','Vol9', 'Fun-9','Fun-8','Fun-7','Fun-6', 'Fun-5', 'Fun-4','Fun-3','Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Fun4', 'Fun5', 'Fun6','Fun7','Fun8','Fun9','Chem-9','Chem-8','Chem-7','Chem-6', 'Chem-5', 'Chem-4','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Chem4', 'Chem5', 'Chem6','Chem7','Chem8','Chem9','Hydro-9','Hydro-8','Hydro-7','Hydro-6', 'Hydro-5','Hydro-4','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Hydro4','Hydro5','Hydro6','Hydro7','Hydro8','Hydro9', 'Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2']
	output = 'Vol0'
	target = [output]
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
	X_train = np.array(X_train, dtype='float64')
	X_test = np.array(X_test, dtype='float64')
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) 
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	accuracy = evaluate(regressor,y_pred, y_test)
	print ("Means absolute error ", metrics.mean_absolute_error(y_test,y_pred) )
	print ("Means squared error ", metrics.mean_squared_error(y_test,y_pred))
	print ("Root means squared error ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

	feature_imp = pd.Series(regressor.feature_importances_, index=features).sort_values(ascending = False)
	feature_imp = feature_imp.nlargest(31)
	sns.barplot(x = feature_imp, y = feature_imp.index)
	plt.xlabel('Punteggio di importanza delle caratteristiche')
	plt.ylabel('Caratteristiche')
	plt.title('Visualizzazione caratteristiche per importanza')
	plt.tight_layout()
	plt.savefig('Features_'+sys.argv[1]+"_"+sys.argv[2]+'.png')

	depth=[]
	for i in range(1,300):
		depth.append(regressor.estimators_[i].tree_.max_depth)

	print("Max_depth min = ", min(depth))
	ind = depth.index(min(depth))
	'''text_representation = tree.export_text(regressor.estimators_[ind])
	with open("decistion_tree.log", "w") as fout:
		fout.write(text_representation)'''
	
	export_graphviz(regressor.estimators_[ind], out_file = "tree_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".dot",feature_names = features, rounded = True, proportion = 2, filled = True)
	call(["dot", "-Tsvg", "tree_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".dot", "-o", "tree_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".svg", "-v"])

if (len(sys.argv) == 3):
	csvfilename = "random_"+sys.argv[1]+"_forest_"+sys.argv[2]+".csv"
	random_forest(csvfilename)
else:
	print ('Errore')


