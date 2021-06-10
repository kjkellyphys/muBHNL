import HNLGen, LabFrame
import sys
import numpy as np

'''
Example Usage:
python3 Scan_HNLEvts_AvgCT.py <DM> "/path/to/filename"

DM: 0 or 1 -- 0 is for a Dirac HNL, 1 is for Majorana

Scans over the masses of HNL given in "muBMasses" (from the MicroBooNE HPS analysis) and saves two files:
"/path/to/filename_CutPassFraction.dat" -- Fraction of events that pass energy/angular cuts as a function of m_N
"/path/to/filename_AvgCosTh.dat" -- Average cosine of the opening angle between final-state electron/positron pairs that pass cuts

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

DMT = int(sys.argv[1]) #user-input choice: Dirac (0) or Majorana (1) HNL
EMinT, OACutT = 0.010, 10.0 #Minimum visible energy (10 MeV) and opening angle (10 degrees) for identification
DetAngUncert = 3.0 #Angular uncertainty for electron tracks

#Masses given in MicroBooNE HPS Analysis Efficiency Tables
muBMasses = [0.001023, 0.003, 0.009, 0.027, 0.054, 0.081, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]

cutpassfracs, means = [], []
for mmi in range(len(muBMasses)):
    mNT = muBMasses[mmi]
    print([mmi, mNT])

    Dist0 = HNLGen.RetSampDM([mNT, meT, meT], [mKT, mmuT], gLgRTrue, 1, True, True) #Generate sample of events
    LF = LabFrame.LFEvts(Dist0, [mNT, meT, meT], [mKT, mmuT])

    LFS = LabFrame.LFSmear(LF, DetAngUncert)
    AnalysisS = LabFrame.LFAnalysis(LFS)

    CutResultsS = LabFrame.CutAnalysis(AnalysisS, EMinT, OACutT)
    EffCuts = CutResultsS[0]

    cutpassfracs.append([mNT, EffCuts])

    PCS = CutResultsS[1]
    if PCS is None:
        continue
    mean = np.average(np.transpose(PCS)[2], weights=np.transpose(PCS)[4])
    means.append([mNT, mean])

fnameSave0 = str(sys.argv[2])
np.savetxt(fnameSave0+"_CutPassFraction.dat", cutpassfracs)
np.savetxt(fnameSave0+"_AvgCosTh.dat", means)