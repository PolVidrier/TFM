import yaml
import uproot
import numpy as np
import pandas as pd

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
            varmc=varmc.replace("ep","L1")
            varmc=varmc.replace("em","L2")
            if not np.isnan(min(vararraymc[varmc])) and not np.isnan(max(vararraymc[varmc])) and not np.isnan(min(vararray[var])) and not np.isnan(max(vararray[var])):
                dictionary[var]=[min([min(vararray[var]),min(vararraymc[varmc])]),max([max(vararray[var]),max(vararraymc[varmc])])]

with open("BDT_B2JpsiKRD/Allvariables.yaml","w") as file:
    documents=yaml.dump(dictionary,file)