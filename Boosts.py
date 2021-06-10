from math import sqrt
'''
Functions for defining the lab-frame N energy and the boost between rest/lab frames
'''

def ENLab(mK, mmu, mN):
    """Lab-frame energy of N in the decay K -> mu N. Can also apply to K -> pi S with proper replacement"""
    return (mK**2 - mmu**2 + mN**2)/(2.0*mK)
def gN(mK, mmu, mN):
    """Boost between lab/rest frames"""
    EN0 = ENLab(mK, mmu, mN)
    return EN0/mN
def bN(mK, mmu, mN):
    """beta between lab/rest frames"""
    gN0 = gN(mK, mmu, mN)
    return sqrt(1.0 - 1.0/gN0**2)
def BoostMat(beta, gamma):
    """Boost matrix between lab/rest frames (assumed to be in z-direction)"""
    return [[gamma, 0, 0, beta*gamma], [0, 1, 0, 0], [0, 0, 1, 0], [beta*gamma, 0, 0, gamma]]