#!/usr/bin/python
import csv
import pg
import time
import sys
import math
import numpy as np

# PRIMO INPUT: H elica o E foglietto


dbname = 'pdb165'
user = 'digiovannantonio165'
passwd = 'db_prto45'
hostname = 'localhost'

def secondaryStructure(csvfilename,db):
	try:
		f = open(csvfilename, 'wt')	
		
		csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		# selezioniamo le proteine spike
		prot = "SELECT p.key, p.name FROM protein AS p WHERE p.type_protein='S';"
		prot = db.query(prot).getresult()
		t0 = time.time()		
		for p in prot:
	
			print("Processing... protein "+ str(p[1]))
			csv_writer.writerow([])
			csv_writer.writerow(['ANALISI PROTEINA ', p[1]])
			# di una singola proteina selezioniamo i domini
			dom = "SELECT DISTINCT p.domain_protein FROM peptid AS p WHERE p.protein = "+str(p[0])+";"
			dom = db.query(dom).getresult()
			
			for d in dom:
				csv_writer.writerow([])
				csv_writer.writerow([])
				csv_writer.writerow([])
				csv_writer.writerow([])
				csv_writer.writerow(['DOMINIO   '+str(d[0])])
				csv_writer.writerow(['C1_x', 'C1_y','C1_z',[],'C2_x','C2_y','C2_z',[],'T_x','T_y','T_z'])
				
				# di un ciascun dominio prendiamo tutti gli amminoacidi
				s = "SELECT * FROM peptid AS p WHERE (p.protein = "+str(p[0])+") AND (p.secondary ='"+sys.argv[1]+"') AND (p.domain_protein ='"+str(d[0])+"') ORDER BY (p.amino_id);"
				l = db.query(s).getresult()
				
				if len(l)<1:
					print('\nNon sono presenti amminoacidi in questo tipo di struttura secondaria\n')
					sys.exit()
						
				pos = list (map(lambda x: x[0],l))
				pos_in = pos[0]
				old_pos= pos[0] - 1
				count = 0
				for k in pos:
					if(k == old_pos + 1):
						old_pos = k
					else:
						n = old_pos - 2
						if (n > 3):
							for j in range (pos_in,n):
							
								s1 = 'SELECT * FROM quatern_amino AS q JOIN torsion_bending as t ON (q.quatern_id = t.quatern_id) JOIN peptid as p ON (q.amino1_id=p.amino_id) WHERE q.amino1_id = '+str(j)+';'		
								l1 = db.query(s1).getresult()
								for i in l1:
									#c1 = (i[7],i[8],i[9])
									#c2 = (i[10],i[11],i[12])
									#tors = (i[13],i[14],i[15])
									csv_writer.writerow([i[7],i[8],i[9],[],i[10],i[11],i[12],[], i[13], i[14], i[15]])
									
						pos_in = k
						old_pos = k
					count = count + 1
		t1 = time.time()
		print("Total time taken... " + str(t1 - t0) + " sec")
	except Exception as e:
		print(e)
	finally:
		f.close()
		
		
if len(sys.argv) == 2:
	if (sys.argv[1] == 'H'):
		csvfilename = "analisi_ALPHA.csv"
	elif (sys.argv[1] == 'E'):
		csvfilename = "analisi_BETA.csv"
	else:
		print('Errore nei parametri di input')
		sys.exit()
	t0 = time.time()
	# Connessione al db
	db = pg.DB(dbname=dbname, host=hostname, user=user,passwd=passwd)
	t1 = time.time()
	print( "DBsetting... " + str(t1 - t0) + " sec")
	secondaryStructure(csvfilename,db)
	db.close()
else:
	print('Errore nel numero di parametri')

	













