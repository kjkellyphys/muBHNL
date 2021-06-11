import LabFrame
import sys
import numpy as np

'''
Example Usage:
python3 Scan_HPSEvts_AvgCT.py "/path/to/filename"

Scans over the masses of HPS given in "muBMasses" (from the MicroBooNE HPS analysis) and saves two files:
"/path/to/filename_CutPassFraction.dat" -- Fraction of events that pass energy/angular cuts as a function of m_S
"/path/to/filename_AvgCosTh.dat" -- Average cosine of the opening angle between final-state electron/positron pairs that pass cuts

--Can increase statistics by changing NS
--Can change angular resolution of detector by changing "DetAngUncert"
--Can change cuts on final-state particles (energy of e+ and e-, opening angle) with "EMinT" and "OACutT"
'''

#Check input arguments
args = sys.argv
if len(args) != 2:
    print("Error: 1 command-line argument expected (filename)")
    print("Using default options instead.")
    fnameSave0 = "TestHPSScan"
else:
    fnamesave0 = str(args[1]) #base filename for saving outputs
mKT, mmuT, mpiT, meT = 0.493677, 0.105658, 0.13957, 0.000511

muBMasses = [0.001023, 0.003, 0.009, 0.027, 0.054, 0.081, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]
NS = int(1e6) #Number of points to sample for HPS Distribution
EMinT, OACutT = 0.010, 10.0 #Minimum visible energy (10 MeV) and opening angle (10 degrees) for identification
DetAngUncert = 3.0 #Angular uncertainty for electron tracks

cutpassfracs, means = [], []
for mmi in range(len(muBMasses)):
    mST = muBMasses[mmi]
    print([mmi, mST])

    LF = LabFrame.LFEvtsHPS([mST, meT], [mKT, mpiT], NS) #Generate lab-frame events
    LFS = LabFrame.LFSmear(LF, DetAngUncert) #Smeared lab-frame events

    #Perform angular analyses of lab-frame events (reco)
    AnalysisS = LabFrame.LFAnalysis(LFS)

    #Perform cut analyses of lab-frame events (reco)
    CutResultsS = LabFrame.CutAnalysis(AnalysisS, EMinT, OACutT)
    EffCuts = CutResultsS[0]

    cutpassfracs.append([mST, EffCuts])
    PCS = CutResultsS[1]
    if PCS is None:
        continue
    mean = np.average(np.transpose(PCS)[2], weights=np.transpose(PCS)[4])
    means.append([mST, mean])

np.savetxt(fnameSave0+"_CutPassFraction.dat", cutpassfracs)
np.savetxt(fnameSave0+"_AvgCosTh.dat", means)