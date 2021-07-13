import numpy as np 
import pandas as pd
import sys
import time
from pomegranate import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def dict_BN(pos):
	diction = {0:'Vol-3', 1:'Vol-2', 2:'Vol-1', 3:'Vol1', 4:'Vol2', 5:'Vol3', 6:'Fun-3', 7:'Fun-2', 8:'Fun-1', 9:'Fun1', 10:'Fun2', 11:'Fun3',12:'Chem-3', 13:'Chem-2', 14:'Chem-1',15:'Chem1', 16:'Chem2', 17:'Chem3', 18:'Hydro-3', 19:'Hydro-2', 20:'Hydro-1',21:'Hydro1',22:'Hydro2',23:'Hydro3',24:'Pos_rel'}
	return diction[pos]


def bayesian_network(csvfilename): 

	data = pd.read_csv(csvfilename, sep= ';')
	X = data.iloc[0:200,[24,31,1,2,3,4,7,8,9,10,13,14,15,16,19,20,21,22]].values
	t0 = time.time()
	print('Constructing Bayesian Network....')
	model = BayesianNetwork.from_samples(X,algorithm="chow-liu", state_names = ['Pos_rel','Vol0', 'Vol-2', 'Vol-1','Vol1', 'Vol2', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2'] )
	# Probabilità di X
	p = model.log_probability(X).sum()
	print ("P(D|M) = ", p)
	#belief = model.predict_proba(X)
	
	# Struttura della rete
	struct = model.structure
	print('Number of edge ',model.edge_count())
	print('Number of node ',model.node_count())
	print('Structure: ', struct)
	pos = 0
	for i in struct:
		if i != ():
			print('Parent of '+ dict_BN(pos)+' is '+dict_BN(i[0]))
			pos = pos +1
	# Visualizzazione della rete
	print('Plotting Bayesian Network...')
	plt.figure()
	model.plot()
	plt.savefig("BN.png", dpi= 500)
	t1=time.time()
	print('Time taken ...', (t1-t0), ' sec')

	# Stima del tempo impiegato nell'apprendimento con i vari algoritmi all'aumentare delle variabili
	'''p1, p2,p3,p4 = list(),list(),list(),list()
	t1,t2,t3,t4 =  list(),list(),list(),list()
	n_vars = range (8,19)
	for i in n_vars:
		t0 = time.time()
		X = data.iloc[0:25,0:i].values
		tic = time.time()
		model = BayesianNetwork.from_samples(X, algorithm='exact-dp')
		t1.append(time.time() - tic)
		p1.append(model.log_probability(X).sum())

		tic = time.time()
		model = BayesianNetwork.from_samples(X, algorithm='exact')
		t2.append(time.time() - tic)
		p2.append(model.log_probability(X).sum())

		tic = time.time()
		model = BayesianNetwork.from_samples(X, algorithm='greedy')
		t3.append(time.time() - tic)
		p3.append(model.log_probability(X).sum())

		tic = time.time()
		model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
		t4.append(time.time() - tic)
		p4.append(model.log_probability(X).sum())
		ttot= time.time()
		print('Four BN learned with '+ str(i) +' variables in ' +str(ttot-t0)+' sec')

	plt.figure(figsize=(14, 4))
	plt.subplot(121)
	plt.title("Time to Learn Structure", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=14)
	plt.xlabel("Variables", fontsize=12)
	plt.ylabel("Time (s)", fontsize=12)
	plt.plot(n_vars, t1, c='c', label="Exact Shortest")
	plt.plot(n_vars, t2, c='m', label="Exact A*")
	plt.plot(n_vars, t3, c='g', label="Greedy")
	plt.plot(n_vars, t4, c='r', label="Chow-Liu")
	plt.legend(fontsize=12, loc=2)

	plt.subplot(122)
	plt.title("$P(D|M)$ with Resulting Model", fontsize=14)
	plt.xlabel("Variables", fontsize=12)
	plt.ylabel("Log Probability", fontsize=12)
	plt.plot(n_vars, p1, c='c', label="Exact Shortest")
	plt.plot(n_vars, p2, c='m', label="Exact A*")
	plt.plot(n_vars, p3, c='g', label="Greedy")
	plt.plot(n_vars, p4, c='r', label="Chow-Liu")
	plt.legend(fontsize=12)
	plt.savefig('BN_test.png')'''


if (len(sys.argv) == 2):
	csvfilename = sys.argv[1]
	bayesian_network(csvfilename)
else:
	print ('Errore nel numero di parametri')


