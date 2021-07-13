#!/usr/bin/python
import csv
import pg
import sys
import math
import time
import numpy as np

# NESSUN PARAMETRO da passare a questo programma

def vprod(v,w):
	# prodotto vettoriale 
	val = (v[1]*w[2]-v[2]*w[1],v[2]*w[0]-v[0]*w[2],v[0]*w[1]-v[1]*w[0])
	return val

def createTorsionBending(db):

	logfile =  open("errori_02_load.txt","w")
	# pulizia tabella 
	pul = "DELETE FROM torsion_bending;"
	db.query(pul)
	
	# query che ritorna tutte le proteine con rispettivi domini e lunghezza della sequenza primaria				
	s1 = "SELECT * FROM size_sequence AS s ORDER BY (s.key, s.domain_protein);"
	l1 = db.query(s1).getresult()
	
	# scorriamo tale lista in modo da prendere ogni volta il dominio con rispettiva lunghezza e prendere le posizioni dei carboni 		# alpha degli amminoacidi di quel dominio 
	
	for j in l1:
		t0 = time.time()
		nomeP= str(j[1])	
		print("Processing... "+ nomeP +'  domain  '+str(j[2]))
	
		s = "SELECT * FROM backbone_view AS b WHERE (b.protein = '" +str(j[0])+"') AND (b.domain_protein ='"+ str(j[2])+"');"
		l = db.query(s).getresult()
	
		n = j[3] - 3	#lunghezza dominio - 3 perchè l'ultima quaterna sarà fatta dall'amminoacido in terzultima pos
		
		for i in range (0,n):
			
			x1 = l[i][5]-l[i+1][5]
			y1 = l[i][6]-l[i+1][6]
			z1 = l[i][7]-l[i+1][7]
			x2 = l[i+2][5]-l[i+1][5]
			y2 = l[i+2][6]-l[i+1][6]
			z2 = l[i+2][7]-l[i+1][7]
			x3 = l[i+1][5]-l[i+2][5]
			y3 = l[i+1][6]-l[i+2][6]
			z3 = l[i+1][7]-l[i+2][7]
			x4 = l[i+3][5]-l[i+2][5]
			y4 = l[i+3][6]-l[i+2][6]
			z4 = l[i+3][7]-l[i+2][7]
			
			v1 = (x1,y1,z1)
			normv1 = np.linalg.norm(v1)
			u1 = (x2,y2,z2)
			normu1 = np.linalg.norm(u1)
			v2 = (x3,y3,z3)
			normv2 = np.linalg.norm(v2)
			u2 = (x4,y4,z4)
			normu2 = np.linalg.norm(u2)
			c1 = vprod(v1,u1)
			c2 = vprod(v2,u2)
			t = vprod(c1,c2)
			normc1 = np.linalg.norm(c1)
			alpha = math.asin(normc1/(normv1*normu1))
			normc2 = np.linalg.norm(c2)
			beta = math.asin(normc2/(normv2*normu2))
			normt = np.linalg.norm(t)
			gamma = math.asin(normt/(normc1*normc2))
			try:
				s2 = "SELECT q.quatern_id FROM quatern_amino AS q WHERE q.amino1_id = "+str(l[i][0])+";"
				l2 = db.query(s2).getresult()[0][0]
			
				db.insert('torsion_bending', quatern_id = l2, bending1_x = c1[0], bending1_y = c1[1], bending1_z = c1[2],bending2_x = c2[0], bending2_y = c2[1], bending2_z = c2[2],torsion_x = t[0], torsion_y = t[1], torsion_z = t[2], alpha = alpha, beta = beta, gamma= gamma)
			except Exception as e:
				logfile.write('\nBAD  ' + nomeP)
		t1 = time.time()
		print("Time taken... " + str(t1 - t0) + " sec")
		
	
	logfile.close()


dbname = 'pdb165'
user = 'digiovannantonio165'
passwd = 'db_prto45'
hostname = 'localhost'
if (len(sys.argv) == 1):
	t0 = time.time()
	db = pg.DB(dbname=dbname,host=hostname,user=user,passwd=passwd)
	t1 = time.time()	
	print( "DBsetting... " + str(t1 - t0) + " sec")
	createTorsionBending(db)
	t2 = time.time()
	print("Total time taken..." + str(t2-t0)+ " sec")
	db.close()
else:
	print ('Errore nel numero dei parametri')

