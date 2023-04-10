# TEST SCRIPT TO WRITE A .ROOT FILE FROM A .TXT FILE


import ROOT

# Create a ROOT file
file = ROOT.TFile("/afs/cern.ch/work/p/pvidrier/private/GITHUB/FITTING/BDTmassroot.root", "RECREATE")

# Create a TTree with one branch, "data"
data = ROOT.vector('float')()
tree = ROOT.TTree("DecayTree", "mytree")
tree.Branch("Jpsi_M", data)

# Read data from the text file and fill the TTree
with open("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT_B2JpsiKRD/data_Jpsi_M.txt") as f:
    for line in f:
        value = float(line.strip())
        data.push_back(value)
        tree.Fill()

# Write the TTree to the ROOT file and close the file
tree.Write()
file.Close()