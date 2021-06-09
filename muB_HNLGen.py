import os
import numpy as np
from math import sin, cos, sqrt
import vegas as vg
import functools
from scipy.optimize import minimize
import itertools
from multiprocessing import Pool, Value, Lock
import sys

#Coefficients of Lorentz-Invariant Objects given couplings gL and gR
#See arXiv:2104.05719 (Table 3) for more details
#Valid as long as the HNL is decaying into a neutrino and identical final-state charged leptons
def CDgLgR(gLgR): #Dirac HNL Decay
    gL, gR = gLgR
    return [64.0*gL*gR, 64.0*gL**2, 64.0*gR**2, -64.0*gL*gR, -64.0*gL**2, -64.0*gR**2]
def CMgLgR(gLgR): #Majorana HNL Decay
    gL, gR = gLgR
    return [128.0*gL*gR, 64.0*(gL**2 + gR**2), 64.0*(gL**2 + gR**2), 0.0, 64.0*(gR**2 - gL**2), 64.0*(gL**2 - gR**2)]

def znumMinMax(zll, s, d):
    #Dalitz Region for the charged-lepton-pair invariant mass m_{ll}^2/m_N^2
    #Depends on the dimensionless sum/differences s = (mm + mp)/mN, d = (mm - mp)/mN
    st = 2.0*(1.0-zll)*sqrt((zll - d**2)*(zll - s**2))/(4.0*zll)
    bt = (d**2*zll + 2.0*s*d + zll*(2.0 - 2.0*zll + s**2))/(4.0*zll)
    return [bt - st, bt + st]

#Calculate Lorentz-Invariants K1, K4, K5, K8, K9, K10 -- Needed to determine matrix-element-squared
#Inputs are the kinematical quantities (z_{ll}, z_{\nu m}, \cos\theta_{ll}, and \gamma_{ll}),
#as well as the masses (N, daughter charged leptons) and the N polarization [-1, 1]
def LIKins(varth, Ms, Pol):
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

#Polarization of a HNL N emerging from the two-body meson decay M -> l N.
def NPol(mM, ml, mN):
    yl, yN = ml/mM, mN/mM
    numterm = (yl**2 - yN**2)*np.sqrt(yN**4 + (1.0 -yl**2)**2 - 2.0*yN**2*(1.0+yl**2))
    denterm = yl**4 + yN**4 - 2.0*yl**2*yN**2 - yl**2 - yN**2
    return numterm/denterm

def RetSampDM(Ms, DecayInfo, gLgR, DM, VB):
    mN, mm, mp = Ms
    mMes, mLep = DecayInfo

    igrange = [[(mm + mp)**2/mN**2, 1.0], [mm**2/mN**2, (mN-mp)**2/mN**2], [-1.0, 1.0], [0.0, 2.0*np.pi]]
    DecPol = NPol(mMes, mLep, mN)

    integrand = vg.Integrator(igrange)
    #resu = integrand(functools.partial(MSqDM_VG, [Ms,gLgR,DecPol,DM]), nitn=20, nstrat=[25,25,8,8])
    resu = integrand(functools.partial(MSqDM_VG, [Ms,gLgR,DecPol,DM]), nitn=20, nstrat=[20,20,6,6])

    if VB:
        print(resu.summary())
        print('result = %s  Q = %.2f' % (resu, resu.Q))

    integral, pts = 0.0, []
    for x, wgt in integrand.random():
        integral += wgt*MSqDM_VG([Ms,gLgR,DecPol,DM], x)
    if VB:
        print(integral)

    for kc in range(30):
        if np.mod(kc, 10) == 0 and VB:
            print(kc)
        for x, wgt in integrand.random():
            M0 = MSqDM_VG([Ms,gLgR,DecPol,DM], x)*wgt*integral
            if M0 != 0.0:
                pts.append([x[0], x[1], x[2], x[3], M0])

    return pts

def GetPts(Dist, npts):
    ret = []
    MW = np.max(np.transpose(Dist)[4])

    while len(ret) < npts:
        pt = Dist[np.random.randint(0,len(Dist))]
        WT = np.random.uniform()*MW
        if WT < pt[4]:
            ret.append(pt[0:4])

    return ret

def ENLab(mK, mmu, mN):
    return (mK**2 - mmu**2 + mN**2)/(2.0*mK)
def gN(mK, mmu, mN):
    EN0 = ENLab(mK, mmu, mN)
    return EN0/mN
def bN(mK, mmu, mN):
    gN0 = gN(mK, mmu, mN)
    return np.sqrt(1.0 - 1.0/gN0**2)
def BoostMat(beta, gamma): #Boost in z-direction
    return [[gamma, 0, 0, beta*gamma], [0, 1, 0, 0], [0, 0, 1, 0], [beta*gamma, 0, 0, gamma]]

def RF4vecs(kins, masses):
    zll, znum, ctll, gamll = kins
    mN, mm, mp = masses
    mllsq, mnumsq = zll*mN**2, znum*mN**2

    Em = 1.0/(2.0*mN)*(mllsq + mnumsq - mp**2)
    Enu = 0.5*mN*(1.0 - zll)

    p3m = sqrt(Em**2 - mm**2)
    cqmnu = (Em*(mN**2 - mllsq) - mN*(mnumsq - mm**2))/((mN**2 - mllsq)*p3m)
    sqmnu = sqrt(1.0-cqmnu**2)
    stll = sqrt(1.0-ctll**2)

    pnuRF = [Enu, 0, -Enu*stll, -Enu*ctll]
    pmRF = [Em, p3m*sqmnu*np.sin(gamll), -p3m*(cqmnu*stll + sqmnu*np.cos(gamll)*ctll), p3m*(sqmnu*np.cos(gamll)*stll - cqmnu*ctll)]
    ppRF = [mN - pnuRF[0] - pmRF[0], -pnuRF[1] - pmRF[1], -pnuRF[2] - pmRF[2], -pnuRF[3] - pmRF[3]]
    
    return [pnuRF, pmRF, ppRF]

def LFEvts(Dist, Masses, LFInfo):
    mN, mm, mp = Masses
    mK, mmu = LFInfo
    gN0 = gN(mK, mmu, mN)
    bN0 = bN(mK, mmu, mN)
    BM = BoostMat(bN0, gN0)

    EsAngles = []
    for evt in Dist:
        RFp = RF4vecs(evt[0:4], Masses)
        wgt = evt[-1]
        pmLF = np.dot(BM, RFp[1])
        ppLF = np.dot(BM, RFp[2])
        #Em, Ep = pmLF[0], ppLF[0]            
        #ctpppm = np.dot(pmLF[1:], ppLF[1:])/np.sqrt(np.dot(pmLF[1:], pmLF[1:])*np.dot(ppLF[1:], ppLF[1:]))
        #ttpmz = np.sqrt(pmLF[1]**2 + pmLF[2])/pmLF[3]
        #ttppz = np.sqrt(ppLF[1]**2 + ppLF[2])/ppLF[3]
        EsAngles.append([pmLF[0], pmLF[1], pmLF[2], pmLF[3], ppLF[0], ppLF[1], ppLF[2], ppLF[3], wgt])
        #EsAngles.append([Em, Ep, ctpppm, ttpmz, ttppz, wgt])
    return EsAngles

def LFAnalysis(Evts):
    toret = []
    for evti, evt in enumerate(Evts):
        if evti % 10000 == 0:
            print([evti, len(Evts), evti/len(Evts)])
        pmE, pmX, pmY, pmZ, ppE, ppX, ppY, ppZ, wgt = evt
        ctpppm = (pmX*ppX + pmY*ppY + pmZ*ppZ)/np.sqrt((pmX**2 + pmY**2 + pmZ**2)*(ppX**2 + ppY**2 + ppZ**2))
        ttpmz = np.sqrt(pmX**2 + pmY**2)/pmZ
        ttppz = np.sqrt(ppX**2 + ppY**2)/ppZ
        toret.append([pmE, ppE, ctpppm, ttpmz, ttppz, wgt])
    return toret

EMin = 0.010
OACut = 10*np.pi/180.0
CTMax = np.cos(OACut)
def CutAnalysis(Evts):
    #Return Fraction of Events Passing Cuts
    #Kinematics of Events Passing Cuts: Energies, Opening Angle, Leading Electron Angle
    toret = []
    for evti, evt in enumerate(Evts):
        if evti % 10000 == 0:
            print([evti, len(Evts), evti/len(Evts)])
        pmE, ppE, ctpppm, ttpmz, ttppz, wgt = evt

        if pmE > EMin and ppE > EMin and np.abs(ctpppm) < CTMax:
            if pmE >= ppE:
                toret.append([pmE, ppE, ctpppm, ttpmz, wgt])
            else:
                toret.append([pmE, ppE, ctpppm, ttppz, wgt])
    if len(toret) == 0:
        return [0.0, None]
    passwgt = np.sum(np.transpose(toret)[4])
    totwgt = np.sum(np.transpose(Evts)[5])
    return [passwgt/totwgt, toret]

sw2 = 0.223
meT = 0.000511
muBMasses = [0.001023, 0.003, 0.009, 0.027, 0.054, 0.081, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]
mKT, mmuT = 0.493677, 0.105658
gLgRTrue = [0.5*(1.0 - 2.0*sw2), sw2] #Only Z-contribution

'''
means = []
cutpassfracs = []
for mmi in range(len(muBMasses)):
    mNT = muBMasses[mmi]
    print([mmi, mNT])

    Dist0 = RetSampDM([mNT, meT, meT], [mKT, mmuT], gLgRTrue, 1, False)

    LF = LFEvts(Dist0, [mNT, meT, meT], [mKT, mmuT])
    np.save("muBLFOutputs/LabFrameEvts"+str(mmi), LF)

    Analysis = LFAnalysis(LF)
    np.save("muBLFOutputs/LabFrameDistributions"+str(mmi), Analysis)

    CutResults = CutAnalysis(Analysis)

    EffCuts = CutResults[0]

    cutpassfracs.append([mNT, EffCuts])
    np.savetxt("CutsPassedFrac.dat", cutpassfracs)

    PC = CutResults[1]
    if PC is None:
        continue
    mean = np.average(np.transpose(PC)[2], weights=np.transpose(PC)[4])
    means.append([mNT, mean])

    np.save("muBLFOutputs/PassCuts"+str(mmi), PC)
    np.savetxt("MeansCT_PassedCuts_HNL.dat", means)
'''

mNT = 0.040
Dist0 = RetSampDM([mNT, meT, meT], [mKT, mmuT], gLgRTrue, 1, False)

LF = LFEvts(Dist0, [mNT, meT, meT], [mKT, mmuT])
Analysis = LFAnalysis(LF)
CutResults = CutAnalysis(Analysis)
PC = CutResults[1]

np.save("muBLFOutputs/PassCuts_40MeV", PC)