import uproot
import pandas as pd
import matplotlib.pyplot as plt
import yaml

mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/00170282/0000/00170282_00000001_1.tuple_bu2kee.root")["Tuple/DecayTree;1"]
data=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root")["tuple_B2JpsiKRD/DecayTree;1"]

features=[]

with open("private/BDT_B2JpsiKRD/Allvariables.yaml","r") as file:
    dictionary=yaml.safe_load(file)

for var in dictionary:
    if var != "Bu_BPVLTIME":
        features.append(var)

mcpd=mc.arrays(features,"Bu_BKGCAT==0",library="pd")
datapd=data.arrays(features,"Jpsi_M > 3200",library="pd")

for var in dictionary:
    if var != "Bu_BPVLTIME":
        plt.hist(mcpd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
        plt.hist(datapd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
        plt.title("{}".format(var))
        plt.xlabel("{}".format(var))
        plt.legend(["MC","data"])
        plt.savefig("private/BDT_B2JpsiKRD/plots/Allvariables/{}.png".format(var))
        plt.close()
