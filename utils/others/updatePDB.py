import openmm as mm
import openmm.app as app 
import sys

def updatePDB(inp, out):

    pdb = app.PDBFile(inp)
    for atom in pdb.topology.atoms():
        if atom.name == "BP":
            atom.element = app.element.phosphorus
        elif atom.name in ["SCD", "SCA"]:
            atom.element = app.element.nitrogen
        else:
            atom.element = app.element.carbon
        if atom.name == "SC1":
            rname = atom.residue.name[1]
            atom.name = f"S{rname}1"
        if atom.name == "B1":
            rname = atom.residue.name[1]
            if rname == "A":
                atom.element = app.element.oxygen
            elif rname == "C":
                atom.element = app.element.sulfur
            elif rname == "U":
                atom.element = app.element.carbon
            elif rname == "G":
                atom.element = app.element.fluorine
    with open(out, "w") as f:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, f)