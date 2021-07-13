import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from eli5.sklearn import PermutationImportance
import sys
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
if not sys.warnoptions:
    warnings.simplefilter("ignore")


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
	X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.3, random_state = 1)
	X_train = np.array(X_train, dtype='float64')
	X_test = np.array(X_test, dtype='float64')
	y_train = np.ravel(y_train)
	y_test =  np.ravel(y_test)
	print ('Start processing...')
	mlp = MLPRegressor(random_state=1, hidden_layer_sizes= 700,solver = 'adam',activation = 'logistic', max_iter=1000)
	mlp.fit(X_train, y_train)
	accuracy = evaluate(mlp, X_test, y_test)
	#test_accuracy = mlp.score(X_test,np.ravel(y_test))
	#print("Accuracy for test data:", test_accuracy)
	perm = PermutationImportance(mlp).fit(X_test,y_test)
	'''w = eli5.show_weights(perm, feature_names=features, top = 6)
	result = pd.read_html(w.data)[0]
	print(result)'''
	l = list(zip(features, perm.feature_importances_))
	l.sort(key=lambda x: x[1], reverse=True)
	x = list(map(lambda x: x[1], l))
	y = list(map (lambda x: x[0], l))
	sns.barplot(x = x, y = y)
	plt.xlabel('Punteggio di importanza delle caratteristiche')
	plt.ylabel('Caratteristiche')
	plt.title('Visualizzazione caratteristiche più importanti')
	plt.tight_layout()
	plt.savefig('Features_mlp.png')
	print("End creating Feauture importances plot")

if (len(sys.argv) == 2):
	csvfilename = sys.argv[1]
	mlp_regressor(csvfilename)

else:
 	print ('Errore nel numero di parametri')

