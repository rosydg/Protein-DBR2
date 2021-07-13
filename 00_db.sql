DROP VIEW IF EXISTS frequence_amino_spike;
DROP VIEW IF EXISTS frequence_amino_myo;
DROP VIEW IF EXISTS report_frequence;
DROP VIEW IF EXISTS occupancy_view;
DROP VIEW IF EXISTS backbone_view;
DROP VIEW IF EXISTS sequence_view;
DROP VIEW IF EXISTS peptid_view;
DROP TABLE IF EXISTS torsion_bending;
DROP TABLE IF EXISTS quatern_amino;
DROP TABLE IF EXISTS occupancy;
DROP TABLE IF EXISTS backbone;
DROP TABLE IF EXISTS peptid;
DROP TABLE IF EXISTS protein;
DROP TABLE IF EXISTS amino;


CREATE TABLE amino(
	name CHAR(3) PRIMARY KEY,
	functional INT,
	chemical INT,
	hydrophily INT
);

CREATE TABLE protein (
	key BIGINT PRIMARY KEY,
	name VARCHAR(30) NOT NULL ,
	type_protein VARCHAR(30) NOT NULL		   
);

CREATE TABLE peptid(
	amino_id INTEGER PRIMARY KEY,
	name_amino CHAR(3) REFERENCES amino(name),
	protein BIGINT REFERENCES protein(key),
	domain_protein CHAR,
	pos_amino BIGINT,
	secondary CHAR
);

CREATE TABLE backbone(
	amino_id bigint REFERENCES peptid(amino_id),
	xpos DOUBLE PRECISION,
	ypos DOUBLE PRECISION,
	zpos DOUBLE PRECISION,
	PRIMARY KEY (amino_id)
);

CREATE TABLE occupancy(
	amino_id bigint REFERENCES peptid(amino_id),
	maxx REAL,
	minx REAL,
	maxy REAL,
	miny REAL,
	maxz REAL,
	minz REAL,
	PRIMARY KEY (amino_id)
);

CREATE TABLE quatern_amino(
	quatern_id BIGINT PRIMARY KEY,
	amino1_id BIGINT,
	amino2_id BIGINT,
	amino3_id BIGINT,
	amino4_id BIGINT,	
	seq CHAR(15)
);


CREATE TABLE torsion_bending( 
	quatern_id BIGINT REFERENCES quatern_amino(quatern_id),		 
	bending1_x DOUBLE PRECISION, 
	bending1_y DOUBLE PRECISION,
	bending1_z DOUBLE PRECISION, 
	bending2_x DOUBLE PRECISION, 
	bending2_y DOUBLE PRECISION, 
	bending2_z DOUBLE PRECISION, 
	torsion_x DOUBLE PRECISION, 
	torsion_y DOUBLE PRECISION, 		
	torsion_z DOUBLE PRECISION,
	alpha DOUBLE PRECISION,
	beta DOUBLE PRECISION,
	gamma DOUBLE PRECISION,	 
	PRIMARY KEY (quatern_id)
);

/*creiamo una vista nella quale inseriamo le quaterne di amminoacidi con la rispettiva frequenza*/

CREATE VIEW report_frequence(sequence, frequence) AS(
	
	SELECT q.seq, count(*) 
	FROM quatern_amino AS q
	GROUP BY q.seq
);
/*Vista che ricava le proprietà degli amminoacidi dalla prima tabella*/

CREATE VIEW peptid_view(amino, amino_id, protein,position,func,chem,hydro) AS (
	SELECT a.name, p.amino_id, p.protein, p.pos_amino, a.functional, a.chemical, a.hydrophily
	FROM peptid AS p JOIN amino AS a ON (a.name = p.name_amino)
);

/*Vista che ricava le posizioni dei carboni alpha degli amminoacidi di una proteina*/

CREATE VIEW backbone_view(amino_id, name_amino, protein, domain_protein, pos_amino, xpos, ypos, zpos) AS (
	SELECT p.amino_id, p.name_amino, p.protein, p.domain_protein, p.pos_amino, b.xpos, b.ypos, b.zpos 	
	FROM backbone AS b JOIN peptid AS p ON (p.amino_id=b.amino_id) 
	ORDER BY (p.protein, p.domain_protein, p.pos_amino)	
);

/*Vista che ricava i volumi degli amminoacidi*/

CREATE VIEW occupancy_view(amino_id, name_amino, protein, domain_protein, pos_amino, occupancy) AS (
	SELECT p.amino_id, p.name_amino, p.protein, p.domain_protein, p.pos_amino, ((o.maxx-o.minx)*(o.maxy-o.miny)*(o.maxz-o.minz))	
	FROM occupancy AS o JOIN peptid AS p ON (p.amino_id=o.amino_id) 
	ORDER BY (p.amino_id)	
);



/*Vista che ricava la lunghezza della sequenza, la posizione inziale e finale degli amminoacidi di un dominio della proteina*/

CREATE VIEW sequence_view(protein, domain_protein, num , pos_in, pos_fin) AS (
	SELECT p.protein, p.domain_protein, count(*), min(p.pos_amino), max(p.pos_amino)
	FROM peptid as p
	GROUP BY p.protein, p.domain_protein	
);


/*Vista che trova l'amminoacido più ripetuto nelle mioglobine*/

CREATE VIEW frequence_amino_myo(name, frequence) AS (
	SELECT p.name_amino, count(*) 
	FROM peptid AS p JOIN protein AS pp ON (p.protein=pp.key)
	WHERE (pp.type_protein = 'M')
	GROUP BY p.name_amino
);

/*Vista che trova l'amminoacido più ripetuto nelle spike*/

CREATE VIEW frequence_amino_spike(name, frequence) AS (
	SELECT p.name_amino, count(*) 
	FROM peptid AS p JOIN protein AS pp ON (p.protein=pp.key)
	WHERE (pp.type_protein = 'S')
	GROUP BY p.name_amino
);



INSERT INTO amino VALUES 
('ALA',1,5,-1), 
('ARG',4,2,1),
('ASN',3,6,1),
('ASP',2,1,1),
('CYS',3,8,1),
('GLN',3,6,1),
('GLU',2,1,1),
('GLY',3,5,1),
('HIS',4,2,1),
('ILE',1,5,-1),
('LEU',1,5,-1),
('LYS',4,2,1),
('MET',1,8,-1),
('PHE',1,7,-1),
('PRO',1,4,-1),
('SER',3,3,1),
('THR',3,3,1),
('TRP',1,7,-1),
('TYR',3,7,1),
('VAL',1,5,-1);











