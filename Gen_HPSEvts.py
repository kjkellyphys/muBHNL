import LabFrame
import sys
import numpy as np

'''
Example Usage:
python3 Gen_HPSEvts.py <m_S> "/path/to/filename"

Saves two files:
"/path/to/filenameTruth.npy" -- Truth-level event distributions for an HPS with mass m_S decaying to e+/e-
"/path/to/filenameReco.npy" -- Reco-level event distributions for an HPS with mass m_S decaying to e+/e-

--Can increase statistics by changing NS
--Can change angular resolution of detector by changing "DetAngUncert"
--Can change cuts on final-state particles (energy of e+ and e-, opening angle) with "EMinT" and "OACutT"
'''

#Check input arguments
args = sys.argv
if len(args) != 3:
    print("Error: 2 command-line arguments expected (m_S, filename)")
    print("Using default options instead.")
    mST, fnameSave0 = 0.050, "TestHPSEvts_"
else:
    mST = float(args[1]) #mST is the HPS mass (GeV)
    fnamesave0 = str(args[2]) #base filename for saving outputs

mKT, mmuT, mpiT, meT = 0.493677, 0.105658, 0.13957, 0.000511

NS = int(1e5) #Number of points to sample for HPS Distribution
LF = LabFrame.LFEvtsHPS([mST, meT], [mKT, mpiT], NS) #Generate lab-frame events

DetAngUncert = 3.0 #Angular uncertainty for electron tracks
LFS = LabFrame.LFSmear(LF, DetAngUncert) #Smeared lab-frame events

#Perform angular analyses of lab-frame events (truth and reco)
Analysis = LabFrame.LFAnalysis(LF)
AnalysisS = LabFrame.LFAnalysis(LFS)

EMinT, OACutT = 0.010, 10.0 #Minimum visible energy (10 MeV) and opening angle (10 degrees) for identification
#Perform cut analyses of lab-frame events (truth and reco)
CutResults = LabFrame.CutAnalysis(Analysis, EMinT, OACutT)
CutResultsS = LabFrame.CutAnalysis(AnalysisS, EMinT, OACutT)

#Save the events that pass cuts, user-input filename base string
PC, PCS = CutResults[1], CutResultsS[1]
np.save(fnameSave0+"Truth", PC)
np.save(fnameSave0+"Reco", PCS)