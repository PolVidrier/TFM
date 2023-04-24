from root_numpy import tree2array, array2root
import uproot
import ROOT

mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root")["Tuple/DecayTree;1"]

old_variables=mc.keys()
new_variables=[]

for variable in old_variables:
    newvariable=variable.replace("L1", "ep")
    newvariable=newvariable.replace("L2", "em")
    new_variables.append(newvariable)

rfile = ROOT.TFile("/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root")
intree = rfile.Get("Tuple/DecayTree;1")

# and convert the TTree into an array
array = tree2array(intree)

# Rename the fields
array.dtype.names = new_variables

# Or write directly into a ROOT file without using PyROOT
array2root(array, "/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root", "DecayTree",mode="RECREATE")