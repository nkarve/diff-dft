import numpy as np
import torch as tc
from torch.fft import fftn, ifftn, fftfreq

class RealCell:
    def __init__(self, r, sample):
        self.r = r
        self.sample = sample
        self.samples = sample ** 3
        self.vol = r ** 3
        self.samplevol = self.vol / self.samples
        self.rspac = r / sample
        self.box_vecs = tc.diag(tc.tensor([r, r, r]))
        self.dx3 = tc.as_tensor(np.mgrid[0:r:self.rspac, 0:r:self.rspac, 0:r:self.rspac])
        self.dr = tc.sqrt(tc.sum(tc.square(self.dx3 - r / 2.), axis=0))

        j0 = fftfreq(sample) * sample
        j1 = fftfreq(sample) * sample
        j2 = fftfreq(sample) * sample
              
        self.ns = tc.as_tensor(np.array(np.meshgrid(j0, j1, j2, indexing='ij')))
        self.k = 2 * tc.pi * tc.einsum('ij,jklm', tc.linalg.inv(self.box_vecs.T), self.ns)
        self.Lm = -self.vol * tc.sum(self.k ** 2, axis=0)
        self.Linvm = tc.zeros_like(self.Lm)
        self.Linvm[self.Lm != 0] = 1. / self.Lm[self.Lm != 0]

    def intg(self, x):
        return tc.sum(x) * self.samplevol

    def O(self, x):
        return self.vol * x
    
    def cJdag(self, x):
        norm = 1. / self.samples
        return norm * self.samples * ifftn(x, dim=(0,1,2))
    
    def cJ(self, x):
        norm = 1. / self.samples
        return norm * fftn(x, dim=(0,1,2))
    
    def cJdagOcJ(self, x):
        return self.samplevol * x

    def cIdag(self, x):
        return fftn(x, dim=(0,1,2))
    
    def cI(self, x):
        return self.samples * ifftn(x, dim=(0,1,2))

    def L(self, x):
        return tc.einsum('ijk,ijkl->ijkl', self.Lm, x)

    def Linv(self, x):
        return tc.einsum('ijk,ijk->ijk', self.Linvm, x)
    
    def reciprocal_poisson_solve(self, x):
        return -4. * tc.pi * self.Linv(self.O(self.cJ(x)))
    
    def real_poisson_solve(self, x):
        return self.cI(self.reciprocal_poisson_solve(x))

