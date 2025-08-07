from ase.io import read 
from ase import units 
from ase.md import velocitydistribution 
from bam_qml.jase.calculator_base import BaseCalculator
from bam_qml.jase.ase_md_npt import NPT3
from bam_qml.jase.ase_md_logger import MDLogger3
#from bam.util import find_input_json
import json
import jax 
import getopt 
import sys 
#jax.config.update('jax_platform_name', 'cpu')

argv = sys.argv[1:]
opts, args = getopt.getopt (
        argv, "hi:", ["help=", "input="]
    )
    
fname_json = "input_qml.json"
for opt, arg in opts:
    if opt in ("-h", "--help"):
        print("python -m bam_qml.training.race_trainer -i <input_file.json>")
        sys.exit(1)
    elif opt in ("-i", "--input"):
        fname_json = arg
print ('fname_json', fname_json)


with open(fname_json) as f:
    json_data = json.load(f)

if json_data['float64']:
    jax.config.update ("jax_enable_x64", True)
    print ("running with float64")
else:
    jax.config.update ("jax_enable_x64", False)
    print ("running with float32")

atoms = read (json_data['MD']['fname_xyz'], index=-1)
atoms.calc = BaseCalculator (json_data) 

dt_fs = 1.0*units.fs
ttime = 25.0*units.fs
ptime = 100.0*units.fs
bulk_modulus = 137.0
pfactor = (ptime**2)*bulk_modulus * units.GPa
temperature_K = float(json_data['MD']['temperature'])
temperature = temperature_K * units.kB
external_stress = 0.01 * units.GPa 

l_init_vel = True
if l_init_vel:
     velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
     velocitydistribution.Stationary(atoms)
     velocitydistribution.ZeroRotation(atoms)
    
fixed_temperature = True
fixed_pressure = False

if not fixed_temperature:
    ttime=None

if not fixed_pressure:
    pfactor = None 

anisotropic = False

dyn = NPT3 (atoms,
            dt_fs,
            temperature=temperature, 
            externalstress=external_stress,
            ttime=ttime,
            pfactor=pfactor,
            anisotropic=anisotropic,
            trajectory=json_data['MD']['fname_traj'],
            logfile=None,
            append_trajectory=True,
            loginterval=50)

logger = MDLogger3 (dyn=dyn, 
                    atoms=atoms, 
                    logfile=json_data['MD']['fname_log'], 
                    stress=False)
dyn.attach (logger, 10)
dyn.run (100000,l_pbc_wrapper=True)
