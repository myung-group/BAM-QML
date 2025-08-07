"""Molecular Carlo."""

import warnings
import numpy as np

from ase.optimize.optimize import Dynamics
from ase.io.trajectory import Trajectory
from ase import units
import datetime 


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)


class MonteCarlo(Dynamics):
    """Base-class for all MC classes."""

    def __init__(self, atoms, dx, temperature, trajectory, logfile=None,
                 loginterval=1, traj_interval=1, append_trajectory=False):
        """Molecular Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean (optional)
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.
        """

        Dynamics.__init__(self, atoms, logfile=None, trajectory=None)

        self.masses = self.atoms.get_masses()
        self.max_steps = None

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (-1, 1)

        self.dx = dx
        self.beta = 1.0/temperature
        self.natoms = len (self.atoms)
        self.n_mc_move = min (4, self.natoms//20)

        # Trajectory is attached here instead of in Dynamics.__init__
        # to respect the loginterval argument.
        self.trajectory = trajectory
        self.traj_interval = traj_interval
        self.logfile = logfile
        self.loginterval = loginterval
        np.random.seed (1)
        crds = self.atoms.get_positions()
        self.save_trajectory (crds, mode="w")
        
        self.log ("###Start_MC_RUN  Epot  DX  Ratio###", mode="w")

    def todict(self):
        return {'type': 'monte-carlo',
                'mc-type': self.__class__.__name__,
                }

    def log (self, mssge, mode="a"):
        if self.logfile:
            with open(self.logfile, mode) as f:
                f.write ("{} {}\n".format (date(), mssge))


    def run(self, steps=50, l_pbc_wrapper=True):
        """ Call Dynamics.run and adjust max_steps """
        self.max_steps = steps + self.nsteps

        naccpt = 0
        ntrial = 0
        invmasses = 1.0/self.atoms.get_masses()

        for i in range (steps):
            '''
            if l_pbc_wrapper and (i+1)%10 == 0:
                q = self.atoms.get_scaled_positions ()
                q = np.mod (q, np.ones(3))
                crds_old = self.atoms.cell.cartesian_positions (q)
                self.atoms.set_positions (crds_old)
            '''
            crds_old = self.atoms.get_positions()
            ener_old = self.atoms.get_potential_energy ()

            crds_new = crds_old.copy()
            o = np.random.randint (self.natoms, size=5)
            d_crds = np.random.normal (loc=0.0, scale=self.dx, size=(5,3))
            crds_new[o] = crds_new[o] + d_crds #np.einsum('ij,i->ij', d_crds, invmasses[o])

            self.atoms.set_positions (crds_new)
            ener_new = self.atoms.get_potential_energy ()

            if ener_new < ener_old:
                # accept
                crds = crds_new
                naccpt += 1
            elif np.random.random_sample () < np.exp (-self.beta * (ener_new - ener_old)):
                # accept
                crds = crds_new
                naccpt += 1
            else:
                # reject
                crds = crds_old

            self.atoms.set_positions (crds)

            self.nsteps += 1
            ntrial += 1
            #self.call_observers ()

            if (i+1)%self.loginterval == 0:
                self.log (f"{i+1:5d} {ener_old:12.4f} {self.dx:8.6f} {naccpt/ntrial:6.4f}")

            if (i+1)%self.traj_interval == 0:
                self.save_trajectory (crds, mode="a")

            if (i+1)%2000 == 0:
                ratio = naccpt / 2000
                #scale = 1.0 + 0.1*(ratio - 0.5)
                self.dx = self.dx * (0.5+ratio)
                naccpt = 0
                ntrial = 0

    def save_trajectory(self, crds, mode):

        with open(self.trajectory, mode) as f:
            print (f"{self.natoms:5d} ", file=f)
            print ("TEST", file=f)
            for xyz in crds:
                print (f"Si   {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}", file=f)

