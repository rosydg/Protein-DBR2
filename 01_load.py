#!/usr/bin/python
import pg
import time
import os
import math
import numpy
import utils
import sys
import warnings
from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)
from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

# Passare a questo programma:
# PRIMO INPUT: path della cartella contenente i file .ent da processare 
# SECONDO INPUT: tipologia di proteina M (mioglobine) o S (covid)

username = 'digiovannantonio165'
password = 'db_prto45'
dbname = 'pdb165'
hostname = 'localhost'
def insertPDB(filename):
	try:
		# connessione al database
		db = pg.DB(dbname=dbname,host = hostname, user=username,passwd=password)
		# creazione del parser del file pdb
		p = PDBParser(PERMISSIVE = 1)
		structure = p.get_structure("X",filename)

		# creazione del dssp per la lettura della struttura secondaria (eliche e foglietti)
		tupla = dssp_dict_from_pdb_file(filename)
		secondary_tupla = tupla[0]
		# estrazione nome della proteina
		name = p.get_header()['idcode']
		
		# query per controllare se la tabella peptid è gia popolata o meno
		s = "SELECT COALESCE(max(p.amino_id)+1, 1) FROM peptid AS p;"
		counter = db.query(s).getresult()[0][0]
		
		# query per controllare se la tabella quatern_amino è gia popolata o meno
		s1 = "SELECT COALESCE(max(q.quatern_id)+1, 1) FROM quatern_amino AS q;"
		counter1 = db.query(s1).getresult()[0][0]
		
		s2 = "SELECT COALESCE(max(p.key)+1, 1) FROM protein AS p;"
		counter2 = db.query(s2).getresult()[0][0]
		
		db.insert('protein', key = counter2, name = name, type_protein = sys.argv[2])
		
		for model in structure:
			for chain in model:
				dom = chain.get_id()
				primary = list()
				id_list = list()
				for residue in chain:
					if(residue.get_id()[0]==' '): #prendiamo solo i residui che non stanno in HETATOM
						nameAmino = residue.get_resname()
						posAmino = residue.get_id()[1]
						info = secondary_tupla[(dom,(' ',posAmino,' '))]
						db.insert('peptid', amino_id = counter, name_amino = nameAmino , protein = counter2, domain_protein = dom, pos_amino = posAmino, secondary = info[1])
						l = residue["CA"].get_vector()
						db.insert('backbone', amino_id = counter,xpos = l[0], ypos = l[1], zpos = l[2])
						primary.append(residue.get_resname())
						id_list.append(counter)
						coo = list()
						for atom in residue:
							if (atom.get_name() != 'CA'):
								coo.append(atom.get_coord())
						x = list(map(lambda x: x[0], coo))
						y = list(map(lambda x: x[1], coo))
						z = list(map(lambda x: x[2], coo))
						vet_max = [max(x),max(y), max(z)]
						vet_min = [min(x), min(y),min(z)]
						db.insert('occupancy', amino_id = counter, maxx = vet_max[0], minx = vet_min[0], maxy = vet_max[1], miny = vet_min[1],maxz = vet_max[2], minz = vet_min[2])
						counter = counter + 1
				n = len(primary)-3
				
				for i in range (0,n):
					sequenza= str(primary[i])+str(primary[i+1])+str(primary[i+2])+str(primary[i+3])
					db.insert('quatern_amino', quatern_id = counter1, amino1_id = id_list[i], amino2_id= id_list[i+1], amino3_id= id_list[i+2], amino4_id= id_list[i+3], seq = sequenza)
					counter1 = counter1 + 1

		db.close()
		exitcode = True
	except Exception as e:
		db.close()
		print (e)
		exitcode = False
	print(str(name)+str(exitcode))
	return exitcode
										

def worker(filename):
	retval = insertPDB(filename)
	logfile = open("load_01_"+ sys.argv[2]+".txt","a+")
	if retval == True:		
		logfile.write("OK "+ filename+'\n')
	else:
		logfile.write("BAD "+ filename+'\n')
	logfile.close()


if len(sys.argv) == 3:
	tot = 0
	t0 = time.time()
	type = sys.argv[2]
	os.system("rm load_01_"+type+".txt ")
	os.system("touch load_01_"+ type+".txt ")	
	onlyfiles = [(sys.argv[1] + f) for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]
	numfiles = len(onlyfiles)
	esp = sys.argv[2]
	t1 = time.time()
	
	print( "DBsetting... " + str(t1 - t0) + " sec")
	# Core
	count = 1
	for i in onlyfiles:
		t0 = time.time()
		worker(i)
		t1 = time.time()
		tt = t1 - t0
		print ("Processing... " + str(tt) + " sec")
		tot = tot + tt
	print ("Finished in "  + str(tot) + " sec")
else:
	print('Errore nel numero di parametri')
