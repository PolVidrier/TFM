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

for var in feature_list:
    for varmc in feature_listmc: 
        if var==varmc:
            dictionary[var]=[min([min(vararray[var]),min(vararraymc[var])]),max([max(vararray[var]),max(vararraymc[var])])]

with open("private/BDT_B2JpsiKRD/Allvariables.yaml","w") as file:
    documents=yaml.dump(dictionary,file)