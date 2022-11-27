import torch as tc
from torch import nn
from functionals import IonicPotential, excpVWN, excVWN

class DFTSolver(nn.Module):
    def __init__(self, cell, norbitals, f):
        super().__init__()
        
        self.cell = cell
        self.norbitals = norbitals
        self.Vdual = tc.zeros_like(self.cell.dr)
        self.f = f
        self.initW()
    
    def initW(self, W0=None):
        if W0 is None: 
            shape = self.cell.sample, self.cell.sample, self.cell.sample, self.norbitals
            self.W = tc.rand(*shape) + 1j * tc.rand(*shape)
            self.W = nn.Parameter(self.orthogonalise(self.W).requires_grad_())
        else:
            self.W = nn.Parameter(W0.requires_grad_())
        
        self.x = nn.Parameter((tc.Tensor([1.5]) + 1j * tc.Tensor([0.])).requires_grad_())
        self.y = nn.Parameter((tc.Tensor([1.5]) + 1j * tc.Tensor([0.])).requires_grad_())

    def getVdual(self):
        ion0 = tc.Tensor([[0, 0, 0]])
        ion1 = tc.real(self.x) * tc.Tensor([[1, 0, 0]])
        ion2 = tc.real(self.y) * tc.Tensor([[0, 1, 0]])
        ions = tc.cat((ion0 + ion2 - ion1, ion0, ion0 + ion2 + ion1), 0)
        self.potential = IonicPotential(self.cell, ions, Z=1)
        return self.potential.Vdual

    def forward(self):
        return tc.real(self.energy(self.W))

    def add_potential(self, V):
        self.Vdual += self.cell.cJdagOcJ(V)

    def orthogonalise(self, W):
        norm = tc.einsum('ijkl,ijkm', W.conj(), self.cell.O(W))
        U, S, V = tc.linalg.svd(norm)
        invsqnorm = U @ tc.diag((S ** -0.5).type(tc.complex128)) @ V
        return W @ invsqnorm

    def density(self, Y):
        IY = self.cell.cI(Y)
        n = tc.sum(IY.conj() * IY, axis=-1)
        return self.f * n

    # Action of hamiltonian on unorthonormalised orbital (possibly redundant - optimise later)
    def H(self, W, orth=False):
        Y = self.orthogonalise(W) if orth else W
        n = self.density(Y)

        exc = excVWN(n)
        excp = excpVWN(n)
        phiexc = self.cell.reciprocal_poisson_solve(n) + self.cell.cJ(exc)
 
        Veff = self.getVdual() + self.cell.cJdag(self.cell.O(phiexc)) + excp * self.cell.cJdagOcJ(n)
        H = -0.5 * self.cell.L(Y)
        H += self.cell.cIdag(tc.einsum('ijk,ijkl->ijkl', Veff, self.cell.cI(Y)))
        
        return H

    def energy(self, W):
        norm = tc.einsum('ijkl,ijkm', W.conj(), self.cell.O(W))
        invn = tc.inverse(norm)
        IW = self.cell.cI(W)
        n = self.f * tc.sum((IW @ invn).conj() * IW, axis=-1)
        
        Veff = self.getVdual()
        phi = .5 * self.cell.reciprocal_poisson_solve(n) + self.cell.cJ(excVWN(n))

        KE = -0.5 * self.f * tc.einsum('ijkl,ijk,ijkl', W.conj(), self.cell.Lm, (W @ invn))
        PE = tc.sum(Veff * n) + tc.sum(n.conj() * self.cell.cJdag(self.cell.O(phi)))

        return KE + PE + self.potential.ewald()

    def psi(self, W):
        Y = self.orthogonalise(W)
        mu = tc.einsum('ijkl,ijkm', Y.conj(), self.H(Y))
        epsilon, D = tc.linalg.eigh(mu)

        return Y @ D, tc.real(epsilon)