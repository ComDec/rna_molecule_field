import openmm as mm
import openmm.app as app
import openmm.unit as unit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
opt = parser.parse_args()


total_step = int(500*1000*500)
pdb = app.PDBFile(f"{opt.input}")
modeller = app.Modeller(pdb.topology, pdb.positions)
ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
modeller.addSolvent(ff, padding=0.8*unit.nanometer)

with open("solv.pdb", "w") as f:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer)
integ = mm.LangevinMiddleIntegrator(298.15*unit.kelvin, 5.0/unit.picosecond, 2.0*unit.femtosecond)
sim = app.Simulation(modeller.topology, system, integ)
sim.reporters.append(
    app.StateDataReporter(f"{opt.output}.out", 10000,
                        totalSteps=total_step, 
                        speed=True,
                        remainingTime=True, 
                        step=True, 
                        time=True,
                        potentialEnergy=True, 
                        volume=True,
                        temperature=True))
sim.reporters.append(
    app.DCDReporter(f"{opt.output}.dcd", 10000)
)
sim.context.setPositions(modeller.getPositions())
sim.minimizeEnergy()
sim.step(total_step)
