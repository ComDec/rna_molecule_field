import sys

import openmm as mm
import openmm.app as app 
import openmm.unit as unit 
app.Topology.loadBondDefinitions("RNA-top-test.xml")


pdb = app.PDBFile(sys.argv[1])
ff = app.ForceField(sys.argv[2])
system = ff.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=0.8*unit.nanometer, constraints=app.HBonds)
integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 5/unit.picosecond, 2*unit.femtosecond)
sim = app.Simulation(pdb.topology, system, integ)

nstep = 50 * 1000 * 500

sim.reporters.append(app.StateDataReporter(sys.argv[1].split('.')[0] + '.out', 500, step=True, potentialEnergy=True, temperature=True, speed=True, totalSteps=nstep, remainingTime=True))
sim.reporters.append(app.DCDReporter(sys.argv[1].split('.')[0] + '.dcd', 500))
sim.context.setPositions(pdb.positions)
sim.minimizeEnergy()
sim.step(nstep)