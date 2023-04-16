import yaml
import uproot
import numpy as np

tupla=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root")["tuple_B2JpsiKRD/DecayTree;1"]
tuplamc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/00170282/0000/00170282_00000001_1.tuple_bu2kee.root")["Tuple/DecayTree;1"]

feature_list=tupla.keys()
vararray=tupla.arrays(feature_list,library="pd") # type: ignore
dictionary={}

feature_listmc=tuplamc.keys()
vararraymc=tuplamc.arrays(feature_listmc,"Bu_BKGCAT==0",library="pd") #type: ignore

for varmc in feature_listmc:
    varmc=varmc.replace("L1","ep")
    varmc=varmc.replace("L2","em")
    for var in feature_list: 
        if var==varmc:
            if var in ["ep_PZ","ep_PID_E","ep_NDOF","ep_HCALPIDE","ep_ETA","ep_ECALPIDE","ep_CLUSTERMATCH","ep_BPVIP","em_BPVIP","em_PZ","em_PID_E","em_NDOF","em_HCALPIDE","em_ETA","em_ECALPIDE","Bu_MAX_PT","Jpsi_BPVIPCHI2","Jpsi_BPVIP","Bu_BPVIPCHI2","Bu_BPVIP","Bu_MAXDOCA","Bu_MIN_PT","Bu_MIN_P","Bu_ETA","Jpsi_ETA","Bu_M","Jpsi_M","Bu_MIN_PT","Bu_SUM_PT","Jpsi_PZ"]:
                varmc=varmc.replace("ep","L1")
                varmc=varmc.replace("em","L2")
                dictionary[var]=[min([min(vararray[var]),min(vararraymc[varmc])]),max([max(vararray[var]),max(vararraymc[varmc])])]

with open("BDT_B2JpsiKRD/variableselection.yaml","w") as file:
    documents=yaml.dump(dictionary,file)