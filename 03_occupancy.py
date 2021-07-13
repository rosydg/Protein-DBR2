#!/usr/bin/python
import csv
import pg
import sys
import math
import numpy 
import matplotlib.pyplot as plt

# Passare a questo programma:
# PRIMO INPUT: tipo di proteina di cui analizzare i volumi M(mioglobine) o S (covid)

username = 'digovannantonio165'
password = 'db_prto45'
dbname = 'pdb165'
hostname = 'localhost'

def createVolume(db):
	
	if (sys.argv[1]=='M'):
		x = 'frequence_amino_myo'
	elif (sys.argv[1]=='S'):
		x = 'frequence_amino_spike'
	else:
		print('Errore nel parametro di ingresso')
		sys.exit()
		
	# Prendo l'amminoacido con frequenza maggiore nel db

	s = "SELECT * FROM "+ str(x) +" AS f order by f.frequence DESC;"
	amino = db.query(s).getresult()[0][0]

	# Creo un file di testo con dati per maria
	s1= "SELECT * FROM occupancy_view AS o JOIN protein AS p ON (p.key = o.protein) WHERE (o.name_amino='"+str(amino)+"') AND (p.type_protein = '"+ sys.argv[1]+"') ORDER BY o.pos_amino;"
	l1 = db.query(s1).getresult()

	logfile = open("volumi_03_"+sys.argv[1]+".txt","w")
	logfile.write('POSIZIONI      VOLUMI')	
	old_pos = -1

	for i in l1:
		if(i[4] == old_pos):
			logfile.write('   ' + str(i[5]))
		else:
			logfile.write('\n'+str(i[4])+'   '+str(i[5]))
			old_pos= i[4]
	logfile.close()	

	# Faccio una query che ritorno la posizione con la media dei volumi dell'amino in quella pos

	s2 = "SELECT o.pos_amino, avg(o.occupancy) FROM occupancy_view AS o JOIN protein AS p ON (p.key = o.protein) WHERE (o.name_amino='"+str(amino)+"') AND (p.type_protein = '"+ sys.argv[1]+"') GROUP BY o.pos_amino;"
	l2 =  db.query(s2).getresult()

	ascissa = list(map(lambda x: int(x[0]),l2))
	ordinata = list(map(lambda x: float(x[1]),l2))
	plt.plot(ascissa, ordinata, 'oy')
	plt.xlabel('Posizioni')
	plt.ylabel('Volumi')
	plt.title('Andamento media dei volumi al variare delle posizioni')
	plt.savefig("Volumi_"+sys.argv[1])
	plt.close()

if len(sys.argv) == 2:
	db = pg.DB(dbname=dbname,host=hostname,user=username,passwd=password)
	createVolume(db)
	db.close()	
else:
	print('Errore nel numero di parametri')













	

