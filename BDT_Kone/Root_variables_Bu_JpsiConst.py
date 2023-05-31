from root_numpy import tree2array, array2root
import uproot
import ROOT
import numpy as np


rfile = ROOT.TFile("/eos/lhcb/user/p/pvidrier/roots/mc_Kone_with_cuts.root")
intree = rfile.Get("DecayTree;1")

# and convert the TTree into an array
array = tree2array(intree)


new_dtype = np.dtype(array.dtype.descr + [('Bu_DTFPV_JpsiConstr_MASS', 'float32')])

# Create new array with new data type
new_data = np.zeros(array.shape, dtype=new_dtype)

# Copy data from old array to new array
for name in array.dtype.names:
    new_data[name] = array[name]

# Assign values to new "Bu_JpsiConstr_MASS" variable
new_data['Bu_DTFPV_JpsiConstr_MASS'] = array["Bu_M"]-array["Jpsi_M"]+3096.9

print(new_data["Kp_PT"])
print(new_data["Bu_DTFPV_JpsiConstr_MASS"])
print(new_data["Bu_DTFPV_JpsiConstr_MASS"].dtype)


# Or write directly into a ROOT file without using PyROOT
array2root(new_data, "/eos/lhcb/user/p/pvidrier/roots/mc_Kone_with_cuts_Bu_JpsiConstr.root", "DecayTree",mode="RECREATE")
