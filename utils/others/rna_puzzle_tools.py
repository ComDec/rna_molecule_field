from Bio.PDB import PDBParser
import os
standard_res = ['A', 'T', 'C', 'G', 'U']

def extract_sequence_from_pdb(pdb_path):
    p = PDBParser(PERMISSIVE=1)
    filename = pdb_path
    structure_id = filename.split('.')[0]
    s = p.get_structure(structure_id, filename)
    model = s[0]
    chain = model.get_list()
    residue = chain[0].get_list()
    seq = []
    for x in residue:
        res_name = x.resname
        for pair_name in standard_res:
            if pair_name in res_name:
                seq.append(pair_name)
                break
            
    return ''.join(seq).strip('\n')

def add_missing_atoms(pdb_path):
    with open("gen.inp", "w") as f:
        f.write(f"source leaprc.RNA.ROC\n")
        f.write(f"a = loadPdb {pdb_path}\n")
        f.write(f"savePDB a {pdb_path}\n")
        f.write(f"quit\n")
    os.system("tleap -f gen.inp")
    os.remove("gen.inp")

if __name__ == '__main__':
    pdb_folder = '/home/MD/RNA/rna_puzzle_dataset/pdb'
    fasta_folder = '/home/MD/RNA/rna_puzzle_dataset/fasta'
    pdb_list = os.listdir(pdb_folder)
    for item in pdb_list:
        pdb_path = os.path.join(pdb_folder, item)
        add_missing_atoms(pdb_path=pdb_path)
        # file_name = item.split('.')[0]
        # seq = extract_sequence_from_pdb(os.path.join(pdb_folder, item))
        # with open(os.path.join(fasta_folder, file_name + '.fasta'), 'w') as f:
        #     f.write(f'>{file_name}')
        #     f.write('\n')
        #     f.write(seq)
        #     f.write('\n')