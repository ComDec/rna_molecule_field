import openmm as mm 
import openmm.app as app
app.Topology.loadBondDefinitions("RNA-top.xml")
import openmm.unit as unit
import mdtraj as md
import sys

xml = sys.argv[1]
path = sys.argv[2]

pdb = app.PDBFile(f"{path}/mol_cg_fixed.pdb")
traj = md.load(f"{path}/mol_cg.dcd", top=f"{path}/mol_cg_fixed.pdb")
traj = traj.superpose(traj[0])
ff = app.ForceField(xml)
system = ff.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.2 * unit.nanometer)
integ = mm.VerletIntegrator(1e-10)
ctx = mm.Context(system, integ)
elist = []
for frame in traj:
    ctx.setPositions(frame.xyz[0,:,:] * unit.nanometer)
    state = ctx.getState(getEnergy=True)
    ener = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    elist.append(ener)
    
print("Mean Energy: ", sum(elist) / len(elist))