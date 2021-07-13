#!/usr/bin/python
import csv
import pg
import sys
import math
import random
import time
import numpy as np


def insert_in_CSV(f,amino,db):

	try:
		csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		count_amino = 0
		t0 = time.time()	
		s = " SELECT p.amino_id, p.protein, p.domain_protein, p.pos_amino FROM peptid as p JOIN protein AS pp ON (p.protein = pp.key) WHERE (pp.type_protein = 'M') AND (p.name_amino = '"+str(amino)+"');"
		list_amino = db.query(s).getresult()
		t1 = time.time()
		print("Start processing "+str(amino)+"... "+ str(t1 - t0) + " sec")
		
		t = time.time()
		for i in list_amino:
		
			if (count_amino == 500):
				t1 = time.time()
				print("500 "+str(amino)+" processed in... "+ str(t1 - t) + " sec")
				count_amino = 0
				t = time.time()
				
			count_amino = count_amino + 1
			s1 = "SELECT s.num, s.pos_in, s.pos_fin FROM sequence_view AS s WHERE (s.protein = "+str(i[1])+" ) AND (s.domain_protein = '"+ str(i[2])+"')"
			seq_list = db.query(s1).getresult()
			size = seq_list[0][0]
			pos_in = seq_list[0][1]
			pos_fin = seq_list[0][2]
			
			if (i[3]>= pos_in + 3  and i[3]<= pos_fin-3):
				pos_relativa = round((i[3]/size)*100, 2)
				id_min = i[0] - 3
				id_max = i[0] + 3
				s = " SELECT p.amino_id, p.name_amino FROM peptid AS p WHERE (p.amino_id <= "+str(id_max)+") AND (p.amino_id >= "+str(id_min)+") AND (p.amino_id != "+str(i[0])+") ORDER BY (p.amino_id) ;"
				l = db.query(s).getresult()
				
				list_occ = list()
				list_func = list()
				list_chem = list()
				list_hydro = list()
				list_name = list()
				for j in l:
					s2 = " SELECT o.occupancy FROM occupancy_view AS o WHERE (o.amino_id = "+ str(j[0])+");"
					occ = db.query(s2).getresult()[0][0]
					list_occ.append(occ)
					list_name.append(str(j[1]))
					s3 = "SELECT a.functional, a.chemical, a.hydrophily FROM amino as a WHERE a.name = '"+ str(j[1])+"';"
					prop = db.query(s3).getresult()
					list_func.append(prop[0][0])
					list_chem.append(prop[0][1])
					list_hydro.append(prop[0][2])
				
				s4 = " SELECT q.quatern_id FROM quatern_amino AS q WHERE (q.amino1_id = "+ str(i[0])+") OR (q.amino4_id = "+ str(i[0])+") ORDER BY q.quatern_id ASC;"		
				l4 = db.query(s4).getresult()
				normc1 = list()
				normc2 = list()
				normtors = list()
				for k in l4:
					s5 = "SELECT * FROM torsion_bending as t WHERE (t.quatern_id = "+ str(k[0])+");"
					l5 = db.query(s5).getresult()
					c1 = (l5[0][1],l5[0][2],l5[0][3])
					c2 = (l5[0][4],l5[0][5],l5[0][6])
					tors = (l5[0][7],l5[0][8],l5[0][9])	
					normc1.append(np.linalg.norm(c1))
					normc2.append(np.linalg.norm(c2))
					normtors.append(np.linalg.norm(tors))
				
				s6 = " SELECT o.occupancy FROM occupancy_view AS o WHERE (o.amino_id = "+ str(i[0])+");"
				vol = db.query(s6).getresult()[0][0]
				csv_writer.writerow([list_occ[0], list_occ[1], list_occ[2], list_occ[3], list_occ[4], list_occ[5],list_func[0], list_func[1], list_func[2], list_func[3], list_func[4], list_func[5],list_chem[0], list_chem[1], list_chem[2], list_chem[3], list_chem[4], list_chem[5],list_hydro[0], list_hydro[1], list_hydro[2], list_hydro[3], list_hydro[4], list_hydro[5], pos_relativa, normc1[0], normc1[1], normc2[0], normc2[1], normtors[0],normtors[1], vol])	

	
	except Exception as e:
		print(e)

		
		
def create_table(db):	

	aminos = ['GLY','ALA','VAL','LEU','ILE','MET','SER','PRO','THR','CYS','ASN','GLN','PHE','TYR','TRP','LYS','HIS','ARG','ASP','GLU']
	f= open("random_forest_tot.csv", 'wt')	
	csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	csv_writer.writerow(['Vol-3', 'Vol-2', 'Vol-1', 'Vol1', 'Vol2', 'Vol3','Fun-3', 'Fun-2','Fun-1', 'Fun1', 'Fun2', 'Fun3','Chem-3', 'Chem-2', 'Chem-1','Chem1', 'Chem2', 'Chem3','Hydro-3', 'Hydro-2', 'Hydro-1','Hydro1','Hydro2','Hydro3', 'Pos_rel', 'C1_Q1','C1_Q2','C2_Q1', 'C2_Q2', 'T_Q1', 'T_Q2', 'Vol0'])
	for i in aminos:	
		insert_in_CSV(f,i,db)
	f.close()	

dbname = 'pdb165'
user = 'digiovannantonio165'
passwd = 'db_prto45'
hostname = 'localhost'

if (len(sys.argv) == 1):
	t0 = time.time()
	db = pg.DB(dbname=dbname,host = hostname,user=user,passwd=passwd)
	t1 = time.time()	
	print( "DBsetting... " + str(t1 - t0) + " sec")
	create_table(db)
	t2 = time.time()
	print("Time taken... " + str(t2 - t0) + " sec")
	db.close()
else:
	print ('Errore nel numero dei parametri')
	
