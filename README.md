# diff-dft

## A fully differentiable Density Functional Theory solver 

### Features:
- Based on Kohn-Sham approach to density functional theory, using fictitious orbitals and potentials to iteratively compute electron density for electronic structure calculations 
- Energy calculation is entirely differentiable allowing use of PyTorch to both minimise energy, and optimise or differentiate wrt any parameter - including ion positions, lattice basis vectors, volume, etc - to compute and visualise unknown thermodynamic and microscopic properties. 
- Modular structure means solver can be treated as any other machine learning model
- Works with plane-wave basis set using PyTorch differentiable FFTs
- Support for Vosko-Wilk-Nusair exchange-correlation functional 
- Ionic potentials use Gaussian-based Ewald summation for inter-ion and self-energy
- In-built support for GPU acceleration

### Example:

See [diff-dft/H3_optimization.py](https://github.com/nkarve/diff-dft/blob/main/diff-dft/H3_optimization.py) - finds the energy, electron density, nuclei positions and bond angles in the ground state of the H<sub>3</sub><sup>+</sup> ion.

<img src="/diff-dft/h3densitymap.png">

### Extensions:
Currently in proof-of-concept stage. Incoming features include:
- Callback-style functional builder
- Pseudopotentials for higher-Z atoms
- Templates for computation of elastic, response and electric properties
- Support for other lattice types
- Finite-temperature effects
- Pretty printing

### Recommended Resources:
- Tomás Arias DFT++ lectures
