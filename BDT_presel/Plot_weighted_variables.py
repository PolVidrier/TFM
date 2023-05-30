
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import yaml

data=uproot.open(r"/eos/lhcb/user/p/pvidrier/roots/data_presel_with_cuts.root")["DecayTree"]
weights=uproot.open(r"BDT_presel/weights/weights.root")["RooTreeDataStore_ds_data_ds_data"]
mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root")["DecayTree;1"]

features=[]
featureweights=weights.keys()
print(featureweights)
xmin=4900
xmax=5600

with open("BDT_Kone/data/Allvariables.yaml","r") as file:
    dictionary=yaml.safe_load(file)

for var in dictionary:
     features.append(var)


weightspd=weights.arrays(featureweights,library="pd")
datapd=data.arrays(features,"(4900<Bu_DTFPV_JpsiConstr_MASS) & (Bu_DTFPV_JpsiConstr_MASS<5600)",library="pd")
mcpd=mc.arrays(features,"(Bu_BKGCAT==0) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005) & (acos((ep_PX*Kp_PX+ep_PY*Kp_PY+ep_PZ*Kp_PZ)/(ep_P*Kp_P))>0.0005) & (acos((Kp_PX*em_PX+Kp_PY*em_PY+Kp_PZ*em_PZ)/(Kp_P*em_P))>0.0005) & (Bu_BPVIPCHI2<9)",library="pd")



for var in dictionary:
        if var!="sBu_BPVIPCHI2":
            if type(dictionary[var][0])!=bool:
                plt.hist(mcpd[var],bins=100, density=True, alpha=0.5)
                plt.hist(datapd[var], weights=weightspd["sig_n_sw"], bins=100, density=True, alpha=0.5)
                plt.title("{}".format(var))
                plt.xlabel("{} [MeV/c]".format(var))
                plt.legend(["MC","Weighted data"])
                plt.savefig("BDT_presel/plots/Weighted_plots/{}.png".format(var))
                plt.close()
            else:
                plt.hist(mcpd[var],range=(0,1), bins=2, density=True, alpha=0.5)
                plt.hist(datapd[var],weights=weightspd["sig_n_sw"], range=(0,1), bins=2, density=True, alpha=0.5)
                plt.title("{}".format(var))
                plt.xlabel("{}".format(var))
                plt.legend(["MC","Weighted data"])
                plt.savefig("BDT_presel/plots/Weighted_plots/{}.png".format(var))
                plt.close()
