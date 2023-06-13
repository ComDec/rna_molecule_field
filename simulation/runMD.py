import openmm as mm
import openmm.app as app 
import openmm.unit as unit 


pdb = app.PDBFile("simulation/1ubq_solv.pdb")
ff = app.ForceField("amber14-all.xml", "tip3p.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9*unit.nanometer, constraints=app.HBonds)
system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, 300*unit.kelvin, 50))
integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 5/unit.picosecond, 2*unit.femtosecond)
sim = app.Simulation(pdb.topology, system, integ)

nstep = 100 * 1000 * 500

sim.reporters.append(app.StateDataReporter("run.out", 50000, step=True, potentialEnergy=True, temperature=True, speed=True, totalSteps=nstep, remainingTime=True))
sim.reporters.append(app.DCDReporter("run.dcd", 50000))
sim.context.setPositions(pdb.positions)
sim.minimizeEnergy()
sim.step(nstep)