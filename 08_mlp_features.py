import numpy as np
import pandas as pd
import sys
import pg
import time
import csv
from sklearn import metrics
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from eli5.sklearn import PermutationImportance

def evaluate(model, y_pred, y_test):
	errors = abs(y_pred - y_test)
	mape = 100 * np.mean(errors / y_test)
	accuracy = 100 - mape
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy 


def mlp_(csvfilename): 
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

	mlp = MLPRegressor(random_state=0,solver = 'adam',activation = 'logistic', hidden_layer_sizes= 700, max_iter= 900)
	mlp.fit(X_train, y_train)
	y_pred = mlp.predict(X_test)
	accuracy = evaluate(mlp, y_pred, y_test)
	# mean absolute error
	mae = str(metrics.mean_absolute_error(y_test,y_pred)).replace(".",",") 
	# mean squared error
	mse = str(metrics.mean_squared_error(y_test,y_pred)).replace(".",",") 
	# root means square
	rms = str(np.sqrt(metrics.mean_squared_error(y_test,y_pred))).replace(".",",") 

	perm = PermutationImportance(mlp).fit(X_test,y_test)
	l = list(zip(features, perm.feature_importances_))
	l.sort(key=lambda x: x[1], reverse=True)
	l = l[0:6]

	return l,mae,mse,rms

def create_CSV():

	aminos = ['GLY','ALA','VAL','LEU','ILE','MET','SER','PRO','THR','CYS','ASN','GLN','PHE','TYR','TRP','LYS','HIS','ARG','ASP','GLU']
	f= open("mlp_"+str(sys.argv[1])+".csv", 'wt')
	csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	csv_writer.writerow(['Amino','FREQ','F1', 'Val', 'F2', 'Val','F3', 'Val', 'F4', 'Val','F5', 'Val','F6', 'Val', 'MAE', 'MSE','RMS'])
	for i in aminos:
		print("Processing.. "+ str(i))
		s = " SELECT f.frequence FROM frequence_amino_myo as f WHERE f.name ='"+ str(i)+"';"
		freq = db.query(s).getresult()[0][0]
		csvfilename = "random_"+str(sys.argv[1])+"_forest_"+str(i)+".csv"
		(l,mae,mse,rms) = mlp_(csvfilename)
		csv_writer.writerow([i, freq, l[0][0], l[0][1], l[1][0],l[1][1],l[2][0],l[2][1],l[3][0],l[3][1],l[4][0],l[4][1],l[5][0],l[5][1],mae,mse,rms])
	f.close()

dbname = 'pdb165'
user = 'digiovannantonio165'
passwd = 'db_prto45'
hostname = 'localhost'

if (len(sys.argv) == 2):
	t0 = time.time()
	db = pg.DB(dbname=dbname,host = hostname,user=user,passwd=passwd)
	t1 = time.time()
	print( "DBsetting... " + str(t1 - t0) + " sec")
	create_CSV()
	t2 = time.time()
	print("Total time taken... " + str(t2 - t0) + " sec")
	db.close()
else:
	print ('Errore')


