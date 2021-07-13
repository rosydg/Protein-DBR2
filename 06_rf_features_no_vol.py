import numpy as np
import pandas as pd
import sys
import pg
import time
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def evaluate(model, y_pred, y_test):
	errors = abs(y_pred - y_test)
	mape = 100 * np.mean(errors / y_test)
	accuracy = 100 - mape
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy 


def random_forest(csvfilename): 

	# Feature settings
	features3 =['Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2',  'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2'] 
	features6 = ['Fun-6', 'Fun-5', 'Fun-4','Fun-3','Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Fun4', 'Fun5', 'Fun6','Chem-6', 'Chem-5', 'Chem-4','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Chem4', 'Chem5', 'Chem6','Hydro-6', 'Hydro-5','Hydro-4','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Hydro4','Hydro5','Hydro6','Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2']
	features9= ['Fun-9','Fun-8','Fun-7','Fun-6', 'Fun-5', 'Fun-4','Fun-3','Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Fun4', 'Fun5', 'Fun6','Fun7','Fun8','Fun9','Chem-9','Chem-8','Chem-7','Chem-6', 'Chem-5', 'Chem-4','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Chem4', 'Chem5', 'Chem6','Chem7','Chem8','Chem9','Hydro-9','Hydro-8','Hydro-7','Hydro-6', 'Hydro-5','Hydro-4','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Hydro4','Hydro5','Hydro6','Hydro7','Hydro8','Hydro9', 'Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2']
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
 
  # Performing training 
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
  accuracy = evaluate(regressor, y_pred, y_test)

	# mean absolute error
	mae = metrics.mean_absolute_error(y_test,y_pred) 
	# mean squared error
	mse = metrics.mean_squared_error(y_test,y_pred)
	# root means square
	rms = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
	
	feature_imp = pd.Series(regressor.feature_importances_, index=features).sort_values(ascending = False)
	feature_imp = feature_imp.nlargest(6)
	x = feature_imp.index
	y = feature_imp

	return x,y,mae,mse,rms

def create_CSV():	

	aminos = ['GLY','ALA','VAL','LEU','ILE','MET','SER','PRO','THR','CYS','ASN','GLN','PHE','TYR','TRP','LYS','HIS','ARG','ASP','GLU']
	f= open("rf_"+str(sys.argv[1])+"_no_vol.csv", 'wt')
	csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	csv_writer.writerow(['Amino','FREQ','F1', 'Val', 'F2', 'Val','F3', 'Val', 'F4', 'Val','F5', 'Val','F6', 'Val', 'MAE', 'MSE','RMS'])
	for i in aminos:
		print("Processing.. "+ str(i))
		s = " SELECT f.frequence FROM frequence_amino_myo as f WHERE f.name ='"+ str(i)+"';"
		freq = db.query(s).getresult()[0][0]
		csvfilename = "random_3_forest_"+str(i)+".csv"
		(x,y,mae,mse,rms) = random_forest(csvfilename)
		csv_writer.writerow([i, freq, x[0], y[0], x[1],y[1],x[2],y[2],x[3],y[3],x[4],y[4],x[5],y[5],mae,mse,rms])
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
	

