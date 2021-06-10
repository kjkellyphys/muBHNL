import numpy as np
from math import sin, cos, sqrt
import vegas as vg
import functools

def CDgLgR(gLgR): #Dirac HNL Decay
    """Coefficients of Lorentz-Invariant Objects given couplings gL and gR (Dirac HNL)
       See arXiv:2104.05719 (Table 3) for more details
       Valid as long as the HNL is decaying into a neutrino and identical final-state charged leptons
    """
    gL, gR = gLgR
    return [64.0*gL*gR, 64.0*gL**2, 64.0*gR**2, -64.0*gL*gR, -64.0*gL**2, -64.0*gR**2]
def CMgLgR(gLgR): #Majorana HNL Decay
    """Coefficients of Lorentz-Invariant Objects given couplings gL and gR (Majorana HNL)
       See arXiv:2104.05719 (Table 3) for more details
       Valid as long as the HNL is decaying into a neutrino and identical final-state charged leptons
    """
    gL, gR = gLgR
    return [128.0*gL*gR, 64.0*(gL**2 + gR**2), 64.0*(gL**2 + gR**2), 0.0, 64.0*(gR**2 - gL**2), 64.0*(gL**2 - gR**2)]

def znumMinMax(zll, s, d):
    """Dalitz Region for the charged-lepton-pair invariant mass m_{ll}^2/m_N^2
       Depends on the dimensionless sum/differences s = (mm + mp)/mN, d = (mm - mp)/mN
    """
    st = 2.0*(1.0-zll)*sqrt((zll - d**2)*(zll - s**2))/(4.0*zll)
    bt = (d**2*zll + 2.0*s*d + zll*(2.0 - 2.0*zll + s**2))/(4.0*zll)
    return [bt - st, bt + st]

def LIKins(varth, Ms, Pol):
    """Calculate Lorentz-Invariants K1, K4, K5, K8, K9, K10 -- Needed to determine matrix-element-squared
       Inputs are the kinematical quantities (z_{ll}, z_{\nu m}, \cos\theta_{ll}, and \gamma_{ll}),
       as well as the masses (N, daughter charged leptons) and the N polarization [-1, 1]
    """
    zll, znum, ctll, gamll = varth
    mN, mm, mp = Ms

    sT = (mm+mp)/mN
    dT = (mm-mp)/mN

    znumLL, znumUL = znumMinMax(zll,sT,dT)
    if znum < znumLL or znum > znumUL: #Return 0 if outside the Dalitz Region
        return 0

    mllsq, mnumsq = zll*mN**2, znum*mN**2
    Em = 1.0/(2.0*mN)*(mllsq + mnumsq - mp**2)
    p3m = sqrt(Em**2 - mm**2)
    cqmnu = (Em*(mN**2 - mllsq) - mN*(mnumsq - mm**2))/((mN**2 - mllsq)*p3m)
    sqmnu = sqrt(1.0-cqmnu**2)
    stll = sqrt(1.0-ctll**2)

    K1 = 0.5*mm*mp*(mN**2 - mllsq)
    K4 = 0.25*(mnumsq - mm**2)*(mN**2 + mp**2 - mnumsq)
    K5 = 0.25*(mN**2 + mm**2 - mnumsq - mllsq)*(mllsq + mnumsq - mp**2)

    K8 = 0.5*Pol*mm*mp*(mN**2 - mllsq)*ctll
    K9 = 0.5*Pol*mN*(mnumsq - mm**2)*(p3m*(cos(gamll)*stll*sqmnu - ctll*cqmnu) - (mN**2 - mllsq)/(2.0*mN)*ctll)
    K10 = 0.5*Pol*mN*p3m*(mN**2 + mm**2 - mnumsq - mllsq)*(ctll*cqmnu - sqmnu*cos(gamll)*stll)

    return [K1,K4,K5,K8,K9,K10]

def MSqDM_VG(MsGsPolDM, varth):
    """ Obtain the matrix-element-squared of the N decay
        Requires the N/charged-lepton masses, gL and gR, N polarization, and whether it is Dirac or Majorana
        varth is a vector of the final-state kinematic measurables (invariant masses and angles) in the N rest frame
    """
    Ms, gLgR, pol, DM = MsGsPolDM
    if DM == 0:
        CVec = CDgLgR(gLgR)
    elif DM == 1:
        CVec = CMgLgR(gLgR)
    else:
        return 0.0
    SISD = LIKins(varth, Ms, pol)
    if SISD == 0: #Check whether the point was allowed in the Dalitz region. If not, return 0.
        return 0.0
    else:
        return np.dot(CVec, SISD)

def NPol(mM, ml, mN):
    """    Polarization of a HNL N emerging from the two-body meson decay M -> l N.
    Arguments:
        mM {float} -- Mass of Meson decaying at rest
        ml {float} -- Mass of charged-lepton being produced along with HNL
        mN {float} -- Heavy Neutral Lepton mass
    Returns:
        [float] -- Polarization of outgoing N in the range [-1, 1]
    """
    yl, yN = ml/mM, mN/mM
    numterm = (yl**2 - yN**2)*sqrt(yN**4 + (1.0 -yl**2)**2 - 2.0*yN**2*(1.0+yl**2))
    denterm = yl**4 + yN**4 - 2.0*yl**2*yN**2 - yl**2 - yN**2
    return numterm/denterm

def RetSampDM(Ms, DecayInfo, gLgR, DM, VB, Short=False):
    """Returns a sample of N decay events in terms of the (reduced) invariant masses and rotation angles, as well as weights
       Ms includes the N and daughter charged lepton masses
       DecayInfo includes the parent meson mass and the charged lepton with which N is produced (two body decay Mes -> Lep + N)
       gLgR includes couplings
       DM is 0 for Dirac N, 1 for Majorana
       If true, VB prints more information as this runs
    """
    mN, mm, mp = Ms
    mMes, mLep = DecayInfo

    igrange = [[(mm + mp)**2/mN**2, 1.0], [mm**2/mN**2, (mN-mp)**2/mN**2], [-1.0, 1.0], [0.0, 2.0*np.pi]]
    DecPol = NPol(mMes, mLep, mN)

    integrand = vg.Integrator(igrange)
    #For more events to be generated, increase nitn. For more precision, increase the entries in nstrat
    if Short:
        resu = integrand(functools.partial(MSqDM_VG, [Ms,gLgR,DecPol,DM]), nitn=20, nstrat=[12,12,4,4])
    else:
        resu = integrand(functools.partial(MSqDM_VG, [Ms,gLgR,DecPol,DM]), nitn=20, nstrat=[20,20,6,6])

    if VB:
        print(resu.summary())
        print('result = %s  Q = %.2f' % (resu, resu.Q))

    integral, pts = 0.0, []
    for x, wgt in integrand.random():
        integral += wgt*MSqDM_VG([Ms,gLgR,DecPol,DM], x)
    if VB:
        print(integral)

    #For more events to be generated, increase NSamp
    NSamp = 30
    for kc in range(NSamp):
        if np.mod(kc, 10) == 0 and VB:
            print(kc)
        for x, wgt in integrand.random():
            M0 = MSqDM_VG([Ms,gLgR,DecPol,DM], x)*wgt*integral
            if M0 != 0.0:
                pts.append([x[0], x[1], x[2], x[3], M0])

    return pts


def GetPts(Dist, npts):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
    ret = []
    MW = np.max(np.transpose(Dist)[4])

    while len(ret) < npts:
        pt = Dist[np.random.randint(0,len(Dist))]
        WT = np.random.uniform()*MW
        if WT < pt[4]:
            ret.append(pt[0:4])

    return ret