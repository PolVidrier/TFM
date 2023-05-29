import uproot
import pandas as pd
import matplotlib.pyplot as plt
import yaml

mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root")["DecayTree;1"]
data=uproot.open(r"/eos/lhcb/user/p/pvidrier/roots/B2JpsiKLFU_MagAll_presel.root")["DecayTree;1"]

features=[]

with open("BDT_Kone/data/Allvariables.yaml","r") as file:
    dictionary=yaml.safe_load(file)

for var in dictionary:
     features.append(var)


mcpd=mc.arrays(features,"Bu_BKGCAT==0",library="pd")
datapd=data.arrays(features,"Jpsi_M > 3200",library="pd")

for var in dictionary:
        if type(dictionary[var][0])!=bool:
            plt.hist(mcpd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
            plt.hist(datapd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
            plt.title("{}".format(var))
            plt.xlabel("{} [MeV/c]".format(var))
            plt.legend(["MC","data"])
            plt.savefig("BDT_Kone/plots/Allvariables/{}.png".format(var))
            plt.close()
        else:
            plt.hist(mcpd[var], range=(0,1),bins=2, density=True, alpha=0.5)
            plt.hist(datapd[var], range=(0,1), bins=2, density=True, alpha=0.5)
            plt.title("{}".format(var))
            plt.xlabel("{}".format(var))
            plt.legend(["MC","data"])
            plt.savefig("BDT_Kone/plots/Allvariables/{}.png".format(var))
            plt.close()
