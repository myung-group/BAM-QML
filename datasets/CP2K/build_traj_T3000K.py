from ase.io import write
from ase.atoms import Atoms
import ase.units as units
import numpy as np 
from ase.calculators.singlepoint import SinglePointCalculator
import gzip 

rng = np.random.default_rng(seed=1)

enr_list = []
xyz_list = []
frc_list = []
with gzip.open('Si-pos_T3000K.xyz.gz', 'rt') as f:
    lines = f.readlines()

nconf = 1880
natoms = 64
for ic in range (nconf):
    l0 = ic*(natoms+2)
    words = lines[l0+1].split()
    enr_list.append (float(words[-1]))
    for ia in range (natoms):
        words = lines[l0+2+ia].split()
        xyz_list.append([float(words[1]), float(words[2]), float(words[3])])

with gzip.open('Si-frc_T3000K.xyz.gz', 'rt') as f:
    lines = f.readlines()
    
for ic in range (nconf):
    l0 = ic*(natoms+2)
    for ia in range (natoms):
        words = lines[l0+2+ia].split()
        frc_list.append([float(words[1]), float(words[2]), float(words[3])])


# CP2K energy unit : Hartree
#      length unit : Ang
enr_list = np.array (enr_list)*units.Hartree
xyz_list = np.array(xyz_list).reshape(-1, natoms, 3)
print (xyz_list.shape)
frc_list = np.array(frc_list).reshape(-1,natoms, 3)*units.Hartree/units.Ang # units.Ang = 1


box = 10.862*np.ones(3)
numbers = 14*np.ones(natoms, dtype=np.int32) # Si's atomic number

idx = np.arange (nconf)
idx = rng.permutation (idx)
idx = rng.permutation (idx)
idx = rng.permutation (idx)
idx = rng.permutation (idx)

atoms_list = []
for ic in idx:
    crds = np.mod (xyz_list[ic], box)
    atoms = Atoms (numbers=numbers, 
                   positions=crds,
                   cell=box,
                   pbc=[True, True, True])
    atoms.calc = SinglePointCalculator(atoms, 
                                 energy=enr_list[ic],
                                 forces=frc_list[ic])
    atoms_list.append (atoms)

write ('si_train_T3000K.xyz', atoms_list[:900])
write ('si_val_T3000K.xyz', atoms_list[900:1000])
write ('si_test_T3000K.xyz', atoms_list[1000:])

