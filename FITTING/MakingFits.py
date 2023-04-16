import uproot
from simpleFit import simpleFit
import ROOT
from FIT import ballFit
from FIT_for_data import ballFitdata
from FIT_for_data_with_restrictions import ballFitdatarest
from FIT_for_data2000to3500 import ballFitdata200035000
from FIT_for_data_with_restrictions2000to3500 import ballFitdatarest20003500
import matplotlib.pyplot as plt

# Trying the simple fit on our mc dataset
file = ROOT.TFile("/afs/cern.ch/work/p/pvidrier/private/roots/mc/00170282/0000/00170282_00000001_1.tuple_bu2kee.root")
tree = file.Get("Tuple/DecayTree;1")
#simpleFit(tree,"",3000., xmin = 1000, xmax = 4000)

# Fit with RooCrystalBall as bkg for our mc dataset
file = ROOT.TFile("/afs/cern.ch/work/p/pvidrier/private/roots/mc/00170282/0000/00170282_00000001_1.tuple_bu2kee.root")
tree = file.Get("Tuple/DecayTree;1")
#ballFit(tree,"",3100., xmin = 0, xmax = 4000)

# Test for our written .root with the data DOES NOT WORK
#file = ROOT.TFile("/afs/cern.ch/work/p/pvidrier/private/GITHUB/FITTING/BDTmassroot.root")
#tree = file.Get("DecayTree")
#ballFit(tree,"",3100., xmin = 0, xmax = 4000)

# Plotting the data from our written .root file DOES WORK
mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/GITHUB/FITTING/BDTmassroot.root")["DecayTree;1"]

mcpd=mc.arrays("Jpsi_M",library="pd")

plt.hist(mcpd, bins=100, alpha=0.5)
plt.title("Jpsi_M")
plt.xlabel("Jpsi_M")
plt.savefig("FITTING/Jpsi_Mfromroot.png")
plt.close()

# WE USE DIRECLY THE .TXT FILE TO MAKE THE FIT
# All parameters can vary
#ballFitdata("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT_B2JpsiKRD/data_Jpsi_M.txt","",3100., xmin = 0, xmax = 4000)
#ballFitdata200035000("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT_B2JpsiKRD/data_Jpsi_M.txt","",3100., xmin = 2000, xmax = 3500)

# Only the mean and sigma of both the gaussian (signal) and RooCrystalBall (background) can vary
# The fixed values taken from the mc ballfit
#ballFitdatarest("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT_B2JpsiKRD/data_Jpsi_M.txt","",3100., xmin = 0, xmax = 4000)
ballFitdatarest20003500("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT_B2JpsiKRD/data_Jpsi_M.txt","",3100., xmin = 2000, xmax = 3500)