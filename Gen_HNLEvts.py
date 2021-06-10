import HNLGen, LabFrame
import sys
import numpy as np

'''
Example Usage:
python3 Gen_HNLEvts.py <m_N> <DM> "/path/to/filename"

m_N: Mass of HNL to decay into a neutrino and an electron/positron pair
DM: 0 or 1 -- 0 is for a Dirac HNL, 1 is for Majorana

Saves two files:
"/path/to/filenameTruth.npy" -- Truth-level event distributions for an HNL with mass m_N decaying to nu e+ e-
"/path/to/filenameReco.npy" -- Reco-level event distributions for an HNL with mass m_N decaying to nu e+ e-

--Can increase statistics by changing final argument of Dist0 from "True" to "False"
--Can change angular resolution of detector by changing "DetAngUncert"
--Can change cuts on final-state particles (energy of e+ and e-, opening angle) with "EMinT" and "OACutT"
'''

sw2 = 0.223
mKT, mmuT, meT = 0.493677, 0.105658, 0.000511

#gL and gR couplings if N decays only via Z-Boson
#See arXiv:2104.05719 for more variations.
#If interested in electron-mixing-only, gL changes to 0.5*(1.0 + 2.0*sw2), gR is unchanged
gLgRTrue = [0.5*(1.0 - 2.0*sw2), sw2]

mNT = float(sys.argv[1]) #user-input mass in GeV for HNL Mass
DMT = int(sys.argv[2]) #user-input choice: Dirac (0) or Majorana (1) HNL
Dist0 = HNLGen.RetSampDM([mNT, meT, meT], [mKT, mmuT], gLgRTrue, 1, True, True) #Generate sample of events
LF = LabFrame.LFEvts(Dist0, [mNT, meT, meT], [mKT, mmuT])

DetAngUncert = 3.0 #Angular uncertainty for electron tracks
LFS = LabFrame.LFSmear(LF, DetAngUncert)

Analysis = LabFrame.LFAnalysis(LF)
AnalysisS = LabFrame.LFAnalysis(LFS)

EMinT, OACutT = 0.010, 10.0 #Minimum visible energy (10 MeV) and opening angle (10 degrees) for identification
CutResults = LabFrame.CutAnalysis(Analysis, EMinT, OACutT)
CutResultsS = LabFrame.CutAnalysis(AnalysisS, EMinT, OACutT)

PC, PCS = CutResults[1], CutResultsS[1]
fnameSave0 = str(sys.argv[3])
np.save(fnameSave0+"Truth", PC)
np.save(fnameSave0+"Reco", PCS)