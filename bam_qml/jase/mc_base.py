from ase.io import read 
from ase import units 
from bam_qml.jase.calculator_base_energy import BaseCalculator
from bam_qml.jase.ase_mc_nvt import MonteCarlo
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

trained_model = json_data['NN']['fname_pkl']

atoms = read (json_data['MC']['fname_xyz'], index=-1)
atoms.calc = BaseCalculator (json_data, model = trained_model) 

dr = 0.015  # A
temperature_K = float(json_data['MC']['temperature'])
temperature = temperature_K * units.kB

mc = MonteCarlo(atoms,
              dr,
              temperature=temperature, 
              trajectory=json_data['MC']['fname_traj'],
              logfile=json_data['MC']['fname_log'],
              append_trajectory=True,
              loginterval=10,
              traj_interval=200)

mc.run (500000, l_pbc_wrapper=True)
