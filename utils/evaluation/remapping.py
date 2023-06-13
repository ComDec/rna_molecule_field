import codecs
import json
import numpy as np
import copy
import yaml
from Bio.PDB import PDBParser
import openmm.app as app
import openmm as mm
import openmm.unit as unit
import os
import sys


def convert_yaml_dict(yaml_path):
    with codecs.open(yaml_path, 'r', 'utf-8') as f_yaml:
        yaml_mapping = yaml.safe_load(f_yaml)

    json_dict = {}
    for item in yaml_mapping:
        top_keys = item['from_residue']
        sec_key = item['to_residue']
        yaml_atom = item['atom_mapping']
        for key in top_keys:
            temp_dict = {
                key: {
                    sec_key: yaml_atom
                }
            }
            json_dict.update(temp_dict)

    return json_dict

def getAA_idx_from_CG(aa_pdb, cg_pdb, mapping_file):

    try:
        with codecs.open(mapping_file, 'r', 'utf-8') as f:
            mapping = json.load(f)
    except:
        mapping = convert_yaml_dict(mapping_file)

    p = PDBParser(PERMISSIVE=1)
    filename = aa_pdb
    structure_id = filename.split('.')[0]
    s = p.get_structure(structure_id, filename)
    model = s[0]
    chain = model.get_list()
    residue = chain[0].get_list()

    mapping_atom_slice = []
    sequences = []

    # parsing residues
    for i in range(len(residue)):
        atom = residue[i]
        residue_name = residue[i].resname
        sequences.append(residue_name)
        try:
            cg_residue_name = [x for x in mapping[residue_name].keys()][0]
        except KeyError:
            print("{} residues name is not defined in mapping files at position {}".format(
                residue_name, i))
            continue

        residue_mapping = mapping[residue_name][cg_residue_name]
        atom_slice_dict = copy.deepcopy(mapping[residue_name][cg_residue_name])

        # parsing mapping dicts
        for keys in residue_mapping.keys():
            temp_atom_slice = []
            mapping_atoms = residue_mapping[keys]
            mapping_nums = len(mapping_atoms)

            # check for validation, ignore missing atoms in AA structures
            fix_mapping_atoms = []
            for sub_idx in range(mapping_nums):
                try:
                    temp_atom = atom[mapping_atoms[sub_idx]]
                    fix_mapping_atoms.append(mapping_atoms[sub_idx])
                except KeyError:
                    print('missing {} at residue {}'.format(
                        mapping_atoms[sub_idx], i))

            # new mapping atoms
            mapping_atoms = fix_mapping_atoms
            mapping_nums = len(mapping_atoms)

            # calculate coords of CG beads
            for sub_idx in range(mapping_nums):
                temp_atom = atom[mapping_atoms[sub_idx]]
                temp_atom_slice.append(temp_atom.serial_number - 1)

            atom_slice_dict[keys] = temp_atom_slice

        mapping_atom_slice.append(atom_slice_dict)

    filename = cg_pdb
    structure_id = filename.split('.')[0]
    s = p.get_structure(structure_id, filename)
    model = s[0]
    chain = model.get_list()
    residue = chain[0].get_list()
    coors = []

    for atoms in residue:
        for atom in atoms:
            coors.append(atom.get_coord())

    position_offset = np.mean(np.array(coors), axis=0)
    CG_coords = copy.deepcopy(mapping_atom_slice)

    for i, beads in enumerate(CG_coords):
        CG_groups = residue[i]
        for CG_atoms in beads.keys():
            try:
                CG_coords[i][CG_atoms] = (CG_groups[CG_atoms].get_coord()-position_offset).tolist()
            except KeyError:
                CG_coords[i][CG_atoms] = []

    # refinement
    return mapping_atom_slice, CG_coords, sequences

def targetFold(ref_pdb, atom_idx, group_COM, save_path):
    ref = app.PDBFile(ref_pdb)

    ff = app.ForceField(f'amber14/RNA.OL3.xml')
    system = ff.createSystem(ref.topology,
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=2 * unit.amu)
    for nforce in range(system.getNumForces())[::-1]:
        force = system.getForce(nforce)
        if isinstance(force, mm.NonbondedForce):
            system.removeForce(nforce)
        if isinstance(force, mm.PeriodicTorsionForce):
           system.removeForce(nforce)

    posforce = mm.CustomCentroidBondForce(1, "k*((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)")
    posforce.addGlobalParameter("k", 0.0)
    posforce.addPerBondParameter("x0")
    posforce.addPerBondParameter("y0")
    posforce.addPerBondParameter("z0")

    for i in range(len(atom_idx)):
        idx_dict = atom_idx[i]
        com_dict = group_COM[i]
        for key in idx_dict.keys():
            if len(idx_dict[key]) == 0:
                continue
            ngrp = posforce.addGroup(idx_dict[key])
            posforce.addBond([ngrp], [x*0.1 for x in com_dict[key]])
    system.addForce(posforce)

    integ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                        5.0 / unit.picosecond,
                                        1.0 * unit.femtosecond)
    simulation = app.Simulation(ref.topology, system, integ)

    simulation.reporters.append(
        app.StateDataReporter(sys.stdout,
                              1000,
                              step=True,
                              potentialEnergy=True,
                              speed=True,
                              remainingTime=True,
                              totalSteps=70000))
    simulation.context.setPositions(ref.getPositions())
    simulation.minimizeEnergy(maxIterations=200)
    for k in np.linspace(0.0, 50.0, 200):
        simulation.context.setParameter("k", k)
        simulation.step(100)
        simulation.minimizeEnergy(maxIterations=100)
    for k in np.linspace(50.0, 500.0, 200):
        simulation.context.setParameter("k", k)
        simulation.step(100)
        simulation.minimizeEnergy(maxIterations=50)
    for k in np.linspace(500.0, 5000.0, 200):
        simulation.context.setParameter("k", k)
        simulation.step(100)
    for k in np.linspace(5000.0, 50000.0, 100):
        simulation.context.setParameter("k", k)
        simulation.step(100)
    simulation.context.setParameter("k", 100000.0)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)

    ff = app.ForceField(f'amber14/RNA.OL3.xml', f"implicit/gbn2.xml")
    system = ff.createSystem(ref.topology,
                             nonbondedMethod=app.CutoffNonPeriodic,
                             nonbondedCutoff=1.6 * unit.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=2 * unit.amu)

    posforce = mm.CustomCentroidBondForce(1, "k*((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)")
    posforce.addGlobalParameter("k", 0.0)
    posforce.addPerBondParameter("x0")
    posforce.addPerBondParameter("y0")
    posforce.addPerBondParameter("z0")

    for i in range(len(atom_idx)):
        idx_dict = atom_idx[i]
        com_dict = group_COM[i]
        for key in idx_dict.keys():
            if len(idx_dict[key]) == 0:
                continue
            ngrp = posforce.addGroup(idx_dict[key])
            posforce.addBond([ngrp], [x*0.1 for x in com_dict[key]])
    system.addForce(posforce)

    integ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin,
                                        5.0 / unit.picosecond,
                                        2.0 * unit.femtosecond)
    simulation = app.Simulation(ref.topology, system, integ)
    simulation.reporters.append(
        app.StateDataReporter(sys.stdout,
                              10000,
                              step=True,
                              speed=True,
                              potentialEnergy=True,
                              remainingTime=True,
                              totalSteps=100000))
    simulation.context.setPositions(state.getPositions())
    simulation.minimizeEnergy()
    simulation.step(100000)
    state = simulation.context.getState(getPositions=True)
    with open(save_path, "w") as f:
        app.PDBFile.writeFile(ref.topology, state.getPositions(), f)


if __name__ == "__main__":
    os.chdir('/home/MD/RNA')
    ref_pdb = './evaluation/ref_rp9.pdb'
    atom_idx, group_COM, sequences = getAA_idx_from_CG(ref_pdb, './rna_puzzle_9_cg.pdb', '../mapping_rna_test.yml')
    targetFold(ref_pdb, atom_idx, group_COM)
