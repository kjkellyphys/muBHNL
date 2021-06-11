# muBHNL
Python-based simulation code for the three-body decay of a heavy neutral lepton (HNL). Includes tools for determining the lab-frame event kinematics of the final-state particles (assumed to be identical charged leptons that are produced along with a light neutrino). This assumes that the production of the HNL is from a two-body meson decay into a charged lepton and an HNL.

Also included: code for simulating two-body decays of Higgs-Portal Scalars (HPS), for comparison against MicroBooNE's analysis of this model.

## Requirements:

-- numpy

-- vegas

-- functools


## Code Included:

### "HNLGen.py"
-- Code for generating a sample of events in an HNL rest-frame

### "Boosts.py"
-- Functions for translating between rest-frame and lab-frame

### "RestFrame.py"
-- Functions for translating between vegas final-state parameters and visible particle four-vectors. Also includes code for generating HPS events in the scalar rest-frame.

### "LabFrame.py"
-- Functions for transforming a distribution of events between rest-frame and lab-frame (both HNL and HPS).

## Sample Scripts:

### "Gen_HNLEvts.py" 
-- Code for generating a distribution of truth and reconstructed HNL events in the lab-frame, including performing some rudimentary analyses.

-- Usage: "python3 Gen_HNLEvts.py mN DM filename" -- mN is the HNL mass in GeV, DM is Dirac (0) vs. Majorana (1) HNL decay, filename is the root of the files to be saved using this script.

### "Gen_HPSEvts.py"
-- Code for generating a distribution of truth and reconstructed HPS events in the lab-frame, including performing some rudimentary analyses.

-- Usage: "python3 Gen_HPSEvts.py mS filename" -- mS is the HPS mass in GeV, filename is the root of the files to be saved using this script.
  
### "Scan_HNLEvts_AvgCT.py"
-- Code for scanning over a range of HNL masses to determine the average lab-frame opening angles of electron/positron pairs, as well as the fraction that pass different cuts.

-- Usage: "python3 Scan_HNLEvts_AvgCT.py DM filename" -- DM is Dirac (0) vs. Majorana (1) HNL decay, filename is the root of the text files to be saved using this script.
