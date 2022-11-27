import torch as tc
from torch import nn, optim
import numpy as np
from lbfgs import LBFGS
from solver import DFTSolver
from cell import RealCell
from utils import A_per_B, plot3d

# Double precision is critical

tc.set_default_dtype(tc.float64)
tc.set_printoptions(precision=9, linewidth=10000)
tc.autograd.set_detect_anomaly(True)


# Set up parameters

# This example is geometry optimization of the H3+ molecule
# (https://en.wikipedia.org/wiki/Trihydrogen_cation)

# The only things to specify are:
# - atomic numbers of constituent natoms (Z=1) 
# - number of electrons in system (2)
# - number of orbitals to construct (1, i.e. only ground state)
# - size and spacing of cell

# Using only this info, the program finds the electron density, 
# equilibrium bond lengths, and LUMO energy

r = 16.
sample = 64

cell = RealCell(r, sample)
norbitals, f = 1, 2
solver = DFTSolver(cell, norbitals, f)

plot = False
save = True
load = False

# You can load a previously saved weight config to optimise later on

if load:
    H3plus_W = tc.load('H3plusweightGO.pt')
    solver.initW(H3plus_W)
    with tc.no_grad():
        solver.x = nn.Parameter(tc.Tensor([0.91721907]) + 1j * tc.Tensor([0.]))
        solver.y = nn.Parameter(tc.Tensor([1.39855908]) + 1j * tc.Tensor([0.]))


# Autograd optimization phase
# Can use any pytorch optimizer (SGD/LBFGS recommended for geometry optimization)

optimizer = LBFGS(solver.parameters(), 
                    lr=0.2,
                    history_size=5, 
                    max_iter=10, 
                    line_search_fn="strong_wolfe")
steps = 60

def closure():
    optimizer.zero_grad()
    E = solver.forward()
    E.backward()
    return E

for i in range(steps):
    print(f'{i+1}/{steps} complete: E = ', end='')
    print(solver.forward().detach().numpy())
    optimizer.step(closure)

# Optimization complete

psi, epsilon = solver.psi(solver.W)

energy = solver.forward().detach().numpy()
xopt = solver.x.detach().numpy()
yopt = solver.y.detach().numpy()

print('ϵ =', epsilon.detach())
print('Total energy =', energy, 'H')

print('Optimized geometry (x, y) =', xopt, 'B', yopt, 'B')
print('Bond length 1 =', xopt * 2 * A_per_B, 'Å')
print('Bond length 2 =', np.sqrt(xopt * xopt + yopt * yopt) * A_per_B, 'Å')

# Both bond lengths around 0.90 Å according to NIST, i.e. atoms in equilateral triangle

if plot:
    for i in range(solver.norbitals):
        orb = cell.cI(psi[..., i])
        plot3d(tc.real(orb.conj() * orb), cell)    

if save:
    tc.save(solver.W, 'H3plusweightGO.pt')