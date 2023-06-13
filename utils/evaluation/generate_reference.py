import sys
from Bio import SeqIO
import openmm.unit as unit
import os

from Bio.PDB import PDBParser
from openmm import app


def genCGReference_from_fasta(fasta_path, save_name):

    seq_reader = SeqIO.parse(fasta_path, 'fasta')
    for record in seq_reader:
        # id = str(record.id).strip('\n')
        seq = str(record.seq).strip('\n')

    text = [i for i in seq]
    text[0] = f"{text[0]}5"
    text[-1] = f"{text[-1]}3"
    seq = " ".join(text)

    with open("gen.inp", "w") as f:
        f.write(f"source leaprc.RNA.ROC\n")
        f.write(f"seq = sequence {{ {seq} }}\n")
        f.write(f"savePDB seq {save_name}.pdb\n")
        f.write(f"quit\n")
    os.system("tleap -f gen.inp")
    os.remove("gen.inp")

def genCGReference(cg_pdb, save_name):
    p = PDBParser(PERMISSIVE=1)
    filename = cg_pdb
    structure_id = filename.split('.')[0]
    s = p.get_structure(structure_id, filename)
    model = s[0]
    chains = model.get_list()
    seq_chains = []
    for chain in chains:
        seq_chains.append([])
        for residue in chain.get_list():
            seq_chains[-1].append(residue.resname[1])

    print("GENERATE REFERENCE")
    for nc, chain in enumerate(seq_chains):
        text = [i for i in chain]
        text[0] = f"{text[0]}5"
        text[-1] = f"{text[-1]}3"
        seq = " ".join(text)
        with open("gen.inp", "w") as f:
            f.write(f"source leaprc.RNA.ROC\n")
            f.write(f"seq = sequence {{ {seq} }}\n")
            if len(seq_chains) > 1:
                f.write(f"savePDB seq _chain{nc}.pdb\n")
            else:
                f.write(f"savePDB seq {save_name}.pdb\n")
            f.write(f"quit\n")
        os.system("tleap -f gen.inp")
        os.remove("gen.inp")
    if len(seq_chains) > 1:
        pdblist = []
        modeller = app.Modeller(app.Topology(), [])
        for nc in range(len(seq_chains)):
            pname = f"_chain{nc}.pdb"
            pdb = app.PDBFile(pname)
            pos = pdb.getPositions(asNumpy=True)
            pos[:, 2] = pos[:, 2] + 1.0 * nc * unit.nanometer
            modeller.add(pdb.topology, pos)
        with open(f'{save_name}.pdb', "w") as f:
            app.PDBFile.writeFile(modeller.getTopology(),
                                  modeller.getPositions(), f)
        os.system("rm _chain*.pdb")


if __name__ == '__main__':
    genCGReference_from_fasta(sys.argv[1], save_name=sys.argv[2])