import yaml
import uproot
import numpy as np

#tupla=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root")["tuple_B2JpsiKRD/DecayTree;1"]
tupla=uproot.open(r"/eos/lhcb/user/p/pvidrier/roots/B2JpsiKLFU_MagAll_presel.root")["DecayTree;1"]
#tupla=uproot.open(r"/eos/lhcb/user/a/alopezhu/EMTF/data/Jpsi2ee_v10Alig/B2JpsiKLFU_MagDown_presel.root")["DecayTree;1"]
tuplamc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root")["DecayTree"]

feature_list=tupla.keys()
vararray=tupla.arrays(feature_list,library="pd") # type: ignore
dictionary={}

feature_listmc=tuplamc.keys()
vararraymc=tuplamc.arrays(feature_listmc,"Bu_BKGCAT==0",library="pd") #type: ignore

for varmc in feature_listmc:
    for var in feature_list: 
        if var==varmc:
            if var in ["ep_CLUSTERMATCH","Bu_M","Jpsi_M","Jpsi_PZ","Bu_MIN_PT","Jpsi_PT","Bu_BPVIP","em_ETA","ep_ETA","Jpsi_BPVIPCHI2","Bu_BPVIPCHI2","Kp_PT","Kp_ETA"]:
                dictionary[var]=[min([min(vararray[var]),min(vararraymc[var])]),max([max(vararray[var]),max(vararraymc[var])])]

with open("BDT_Ktwo/data/variableselection.yaml","w") as file:
    documents=yaml.dump(dictionary,file)