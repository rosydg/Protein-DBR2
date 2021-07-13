#!/usr/bin/python
import csv
import pg
import sys
import math
import numpy as np

# Passare a questo programma:
# PRIMO INPUT: H(helix) o E(sheet)
# SECONDO INPUT: M(mioglobine) o S (covid)

dbname = 'pdb'
user = 'rosydg'
passwd = 'rosydg'

def secondaryStructure(db):
	s = "SELECT * FROM peptid AS p JOIN protein as pp ON (p.protein = pp.key) WHERE (pp.type_protein = '"+sys.argv[2]+"') AND (p.secondary ='"+sys.argv[1]+"') ORDER BY (p.amino_id);"
	l = db.query(s).getresult()
	
	if not l:
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
						logfile.write('\n Quaterna   '+str(i[5])  +'nel dominio '+str(i[22])+'  della proteina  '+ str(l[count][7])+' in posizione '+ str(i[23]) )
						tors = (i[13],i[14],i[15])
						c1 = (i[7],i[8],i[9])
						c2 = (i[10],i[11],i[12])
						logfile.write('\n TORSIONE     ' + str(tors) + '          NORMA  ' + str(np.linalg.norm(tors)))
						logfile.write('\n CURVATURA1   ' + str(c1) + '       NORMA  ' +str(np.linalg.norm(c1)))
						logfile.write('\n CURVATURA2   ' + str(c2) + '       NORMA  ' +str(np.linalg.norm(c2)))
						logfile.write('\n ALPHA   ' + str(i[16])+ '        BETA   ' + str(i[17]) + '        GAMMA   ' + str(i[18]))
						logfile.write('\n')
			
			pos_in = k
			old_pos = k
		count = count + 1
	logfile.write('\n *******************************NOTA BENE**********************************\n ALPHA è l angolo che si forma tra v1 e u1 dove: \n v1 = vettore che congiunge la seconda serina con la prima\n u1 = vettore che congiunge la seconda serina con la valina\n\n BETA è l angolo che si forma tra v2 e u2 dove: \n v2 = vettore che congiunge la valina con la seconda serina\n u2 = vettore che congiunge la valina con la leucina\n\n GAMMA è l angolo che si forma tra le due curvature supposto che abbiano lo stesso punto di applicazione')		
	logfile.close()	


if len(sys.argv) == 3:
	if (sys.argv[1] == 'H' and sys.argv[2] == 'M') or (sys.argv[1] == 'H' and sys.argv[2] == 'S'):
		logfile = open("helix_"+sys.argv[2]+"_04.txt","w")
	elif (sys.argv[1] == 'E' and sys.argv[2] == 'M') or (sys.argv[1] == 'E' and sys.argv[2] == 'S'):
		logfile = open("sheet_"+sys.argv[2]+"_04.txt","w")
	else:
		print('Errore nei parametri di input')
		sys.exit()
	# Connessione al db
	db = pg.DB(dbname=dbname,user=user,passwd=passwd)
	secondaryStructure(db)
	db.close()
else:
	print('Errore nel numero di parametri')

	













