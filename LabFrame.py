from Boosts import gN, bN, BoostMat
from RestFrame import RF4vecs, RFHPS
import numpy as np
from math import sin, cos, sqrt

"""
Code for boosting rest-frame events (HNL N or HPS S) into the lab-frame.
HNL events are assumed to have weights, HPS are equally-weighted

LFSmear() is used to add in angular uncertainty to the truth information,
    using some angular resolution "Res". Energy uncertainty can be incorporated in this step as well.

LFAnalysis() takes events and maps onto certain observables:
    Energy of the two charged leptons, their opening angle, and each track's
    angle with respect to the incoming N/S direction.

CutAnalysis() performs a cut-based analysis on truth or reconstructed events,
    and returns the fraction of events (and the events themselves) that pass cuts.
"""

def LFEvts(Dist, Masses, LFInfo):
    """Take the rest-frame four-vectors and transform them into the lab frame
       Returns the lab-frame charged lepton four-vectors
    """
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
        EsAngles.append(np.concatenate([pmLF, ppLF, [wgt]]))
    return EsAngles

def LFEvtsHPS(masses, LFInfo, NSamp):
    """Generates a set of lab-frame HPS events
       masses: [mS, me] masses of HPS S and daughter charged-leptons
       LFInfo: [mK, mpi] masses of parent particle and other particle produced with S (K -> pi S)
       NSamp: Desired length of samples
    """
    mS, me = masses
    mK, mpi = LFInfo
    gN0 = gN(mK, mpi, mS)
    bN0 = bN(mK, mpi, mS)
    BM = BoostMat(bN0, gN0)

    EsAngles = []
    RFs = RFHPS(masses, NSamp)
    for ei in RFs:
        pmLF = np.dot(BM, ei[0])
        ppLF = np.dot(BM, ei[1])
        EsAngles.append(np.concatenate([pmLF, ppLF, [1.0]]))
    return EsAngles

def LFSmear(Evts, Res, VB=False):
    """Smear lab-frame events given angular resolution (energy is assumed to be perfectly measured)
       Res: angular uncertainty for a given track in degrees

       If energy uncertainty is desired, it can be implemented by rescaling {pmE, pmX, pmY, pmZ} and {ppE, ppX, ppY, ppZ} accordingly
    """
    toret = []
    for evti, evt in enumerate(Evts):
        if VB and evti % 100000 == 0:
            print([evti, len(Evts), evti/len(Evts)])
        DTpm, DTpp = np.random.normal(scale=Res, size=2)*np.pi/180.0
        phipm, phipp = np.random.uniform(0.0, 2.0*np.pi, size=2)

        pmE, pmX, pmY, pmZ, ppE, ppX, ppY, ppZ, wgt = evt

        pTm = sqrt(pmX**2 + pmY**2)
        p3m = sqrt(pmX**2 + pmY**2 + pmZ**2)
        pmXR = pmX*cos(DTpm) + sin(DTpm)/pTm*(-pmX*pmZ*cos(phipm) + p3m*pmY*sin(phipm))
        pmYR = pmY*cos(DTpm) - sin(DTpm)/pTm*(pmY*pmZ*cos(phipm) + p3m*pmX*sin(phipm))
        pmZR = pmZ*cos(DTpm) + pTm*cos(phipm)*sin(DTpm)

        pTp = sqrt(ppX**2 + ppY**2)
        p3p = sqrt(ppX**2 + ppY**2 + ppZ**2)
        ppXR = ppX*cos(DTpp) + sin(DTpp)/pTp*(-ppX*ppZ*cos(phipp) + p3p*ppY*sin(phipp))
        ppYR = ppY*cos(DTpp) - sin(DTpp)/pTp*(ppY*ppZ*cos(phipp) + p3p*ppX*sin(phipp))
        ppZR = ppZ*cos(DTpp) + pTp*cos(phipp)*sin(DTpp)

        toret.append([pmE, pmXR, pmYR, pmZR, ppE, ppXR, ppYR, ppZR, wgt])
    return toret

def LFAnalysis(Evts, VB=False):
    """Perform analyses on the lab-frame events.
       Evts: list of events after processing into lab frame (either directly from LFEvts or after smearing in LFSmear)
       VB: (optional) print progress

       Returns: Energies of each of the charged leptons 
                (cosine) of the opening angle between charged leptons
                (tangent) of the angle between each charged lepton and incoming N/S direction (assumed to be z-direction)
    """
    toret = []
    for evti, evt in enumerate(Evts):
        if VB and evti % 100000 == 0:
            print([evti, len(Evts), evti/len(Evts)])
        pmE, pmX, pmY, pmZ, ppE, ppX, ppY, ppZ, wgt = evt
        ctpppm = (pmX*ppX + pmY*ppY + pmZ*ppZ)/sqrt((pmX**2 + pmY**2 + pmZ**2)*(ppX**2 + ppY**2 + ppZ**2))
        ttpmz = sqrt(pmX**2 + pmY**2)/pmZ
        ttppz = sqrt(ppX**2 + ppY**2)/ppZ
        toret.append([pmE, ppE, ctpppm, ttpmz, ttppz, wgt])
    return toret

def CutAnalysis(Evts, EMin=0.010, OACut=10):
    """Take the lab-frame information and perform some cuts:
        EMin: minimum energy (GeV) of each electron/positron track to identify as a distinct track
            default: 10 MeV = 0.010 GeV
        OACut: minimum opening angle (degrees) to identify pair of tracks
            default: 10 degrees

        Returns: *fraction of events that passes cuts
                 *Kinematics of passing events: energies, opening angle, leading electron angle
    """
    CTMax = np.cos(OACut*np.pi/180.0)

    toret = []
    for evti, evt in enumerate(Evts):
        if evti % 100000 == 0:
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