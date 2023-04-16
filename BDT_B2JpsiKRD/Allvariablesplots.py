import uproot
import pandas as pd
import matplotlib.pyplot as plt
import yaml

mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root")["Tuple/DecayTree;1"]
data=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root")["tuple_B2JpsiKRD/DecayTree;1"]

features=[]
featuresmc=[]

with open("BDT_B2JpsiKRD/Allvariables.yaml","r") as file:
    dictionary=yaml.safe_load(file)

for var in dictionary:
    varmc=var.replace("ep","L1")
    varmc=varmc.replace("em","L2")
    features.append(var)
    featuresmc.append(varmc)

mcpd=mc.arrays(featuresmc,"Bu_BKGCAT==0",library="pd")
datapd=data.arrays(features,"Jpsi_M > 3200",library="pd")

for var in dictionary:
    print("Plotting",var)
    varmc=var.replace("ep","L1")
    varmc=varmc.replace("em","L2")
    if type(dictionary[var][0])!=bool:
        plt.hist(mcpd[varmc], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
        plt.hist(datapd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
        plt.title("{}".format(var))
        plt.xlabel("{}".format(var))
        plt.legend(["MC","data"])
        plt.savefig("BDT_B2JpsiKRD/plots/Allvariables/{}.png".format(var))
        plt.close()
    else:
        plt.hist(mcpd[varmc], range=(0,1),bins=2, density=True, alpha=0.5)
        plt.hist(datapd[var], range=(0,1), bins=2, density=True, alpha=0.5)
        plt.title("{}".format(var))
        plt.xlabel("{}".format(var))
        plt.legend(["MC","data"])
        plt.savefig("BDT_B2JpsiKRD/plots/Allvariables/{}.png".format(var))
        plt.close()
