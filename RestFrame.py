import numpy as np
from math import sqrt, cos, sin
'''
Code for determining the four-vectors of the final-state particles (electrons, positrons, neutrinos)
in the rest-frame of the decaying particle (HNL N or HPS S)
'''

def RF4vecs(kins, masses):
    """Given the parameters of the final-state (invariant masses and angles), determine the rest-frame four-vectors of the outgoing neutrino and charged-lepton pair
    kins: set of kinematical variables in the rest frame:
        zll = m_{\ell\ell}^2/m_N^2: reduced invariant mass of the charged lepton pair
        znum = m_{\nu m}^2/m_N^2: reduced invariant mass of the neutrino/negatively-charged-lepton
        ctll: cosine of the angle between the charged-lepton-pair (sum of the four-vectors) and the z-axis
        gamll: rotation angle about the direction of the charged-lepton pair
    masses: [mN, mm, mp] the HNL and daughter charged-lepton masses (mm: negatively-charged, mp: positively-charged)
    """
    zll, znum, ctll, gamll = kins
    mN, mm, mp = masses
    mllsq, mnumsq = zll*mN**2, znum*mN**2

    Em = 1.0/(2.0*mN)*(mllsq + mnumsq - mp**2)
    Enu = 0.5*mN*(1.0 - zll)

    p3m = sqrt(Em**2 - mm**2)
    cqmnu = (Em*(mN**2 - mllsq) - mN*(mnumsq - mm**2))/((mN**2 - mllsq)*p3m)
    sqmnu = sqrt(1.0-cqmnu**2)
    stll = sqrt(1.0-ctll**2)

    phi = np.random.uniform(0.0, 2.0*np.pi)
    pnuRF = [Enu, -Enu*stll*np.sin(phi), -Enu*stll*np.cos(phi), -Enu*ctll]
    pmRF = [Em, p3m*(sqmnu*cos(phi)*sin(gamll) - sin(phi)*(sqmnu*cos(gamll)*ctll + cqmnu*stll)), p3m*(-cos(phi)*(sqmnu*cos(gamll)*ctll + cqmnu*stll) - sin(phi)*sqmnu*sin(gamll)), p3m*(-cqmnu*ctll + sqmnu*cos(gamll)*stll)]
    ppRF = [mN - pnuRF[0] - pmRF[0], -pnuRF[1] - pmRF[1], -pnuRF[2] - pmRF[2], -pnuRF[3] - pmRF[3]]
    
    return [pnuRF, pmRF, ppRF]

def RFHPS(masses, NSamp, VB=False):
    """Generate a sample of rest-frame Higgs Portal Scalar decays into electron/positron pairs
    masses: [mS, me] the scalar and daughter charged-lepton masses
    NSamp: length of sample to return (randomly drawn cos(theta) and phi in the S rest-frame)
    VB: (bool) whether to print information while running
    """
    mS, me = masses

    if me >= mS/2.0:
        print("Scalar too light to decay to charged lepton pair")
        return None

    toret = []
    for j in range(NSamp):
        if VB and j % 10000 == 0:
            print([j, NSamp, j/NSamp])
        
        ct = np.random.uniform(-1.0, 1.0)
        phi = np.random.uniform(0.0, 2.0*np.pi)

        Ee = mS/2.0
        pe = sqrt(Ee**2 - me**2)
        p1RF = [Ee, pe*sqrt(1 - ct**2)*sin(phi), pe*sqrt(1 - ct**2)*cos(phi), pe*ct]
        p2RF = [Ee, -pe*sqrt(1 - ct**2)*sin(phi), -pe*sqrt(1 - ct**2)*cos(phi), -pe*ct]
        toret.append([p1RF, p2RF])

    return toret