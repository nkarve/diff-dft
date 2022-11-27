import numpy as np
import torch as tc

class IonicPotential():
    def __init__(self, cell, X, Z):
        self.cell = cell
        self.X = X
        self.Z = Z
        self.sf = tc.sum(tc.exp(-1j * tc.einsum('ij,jklm->iklm', self.X, cell.k)), axis=0)
        self.Vk = 4 * np.pi * Z * cell.vol * cell.Linvm * self.sf 
        self.Vdual = tc.real(cell.cJdag(self.Vk))

    def ewald(self, sigma=0.25):
        g1 = self.Z * Gaussian(sigma)(self.cell.dr)
        nz = tc.real(self.cell.cI(self.cell.cJ(g1) * self.sf))
        phi = self.cell.real_poisson_solve(nz)

        Uself = self.Z ** 2 / (2 * np.sqrt(np.pi)) * (1/sigma) * self.X.shape[0]
        Unum = 0.5 * self.cell.intg(phi * nz)

        ewald = Unum - Uself
        return ewald

class Gaussian:
    def __init__(self, sigma):
        self.const1 = (1 / (2 * tc.pi * sigma * sigma)) ** 1.5
        self.const2 = 1 / (2 * sigma * sigma)

    def __call__(self, r):
        return self.const1 * tc.exp(-r * r * self.const2)


def excVWN(n):
    X1 = 0.75*(3.0/(2.0*np.pi))**(2.0/3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4*c-b*b)
    X0 = x0*x0+b*x0+c

    rs=(4*np.pi/3*n)**(-1./3.)
  
    x = tc.sqrt(rs)
    X = x*x+b*x+c

    return -X1/rs + A*(tc.log(x * x / X) +2 * b / Q * tc.arctan(Q/(2 * x+b)) - (b*x0)/X0*(tc.log((x-x0)*(x-x0)/X)+2*(2*x0+b)/Q*tc.arctan(Q/(2*x+b))))

def excpVWN(n):
    X1 = 0.75*(3.0/(2.0*np.pi))**(2.0/3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4.*c-b*b)
    X0 = x0*x0+b*x0+c

    rs=(4.*np.pi/3. * n)**(-1./3.)

    x=tc.sqrt(rs)
    X=x*x+b*x+c

    dx=0.5/x

    return (-rs/(3.*n))* dx * (2.*X1/(rs*x)+A*(2./x-(2.*x+b)/X-4.*b/(Q*Q+(2.*x+b)*(2.*x+b))-(b*x0)/X0*(2./(x-x0)-(2.*x+b)/X-4.*(2.*x0+b)/(Q*Q+(2*x+b)*(2*x+b)))))