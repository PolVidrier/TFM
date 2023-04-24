import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import yaml
from root_numpy import array2root



def BDT_Applying(model,variables,file,tree,cuts,name):
    # model= name of the BDT model file, a .json file
    # variables= variables for the training, in .yaml file
    # file= name of the .root file
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # name= name for the BDT_output

    # We load the model
    xgb_classifier = xgb.XGBClassifier(max_depth=6) # the default 
    xgb_classifier.load_model(model)

    data=uproot.open(file)[tree]
    features=[]

    with open(variables,"r") as file:
        dictionary=yaml.safe_load(file)

    for var in dictionary:
        features.append(var)

    if cuts=="":
        fulldatapd=data.arrays(features,library="pd")
    else:
        fulldatapd=data.arrays(features,cuts,library="pd")

    del fulldatapd["Jpsi_M"]   #we delete the mass
    del fulldatapd["Bu_M"]

    data=fulldatapd.to_numpy()

    datapred_prob = xgb_classifier.predict_proba(data)  #probability of predictions  

    datapred_prob=datapred_prob[:,1]
    datapred_prob.dtype=[("BDT_output",np.float32)]

    # We write the output in a .root file
    output_file="BDT/data/BDT_output_%s.root" % name
    output_tree="DecayTree"
    array2root(datapred_prob,output_file,output_tree,mode="RECREATE")
    print("BDT applied to %s" % name)


# FOR THE DATA
model="BDT/data/BDT.json"
variables="BDT/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
cuts="Jpsi_M>100"
name="data"

BDT_Applying(model,variables,file,tree,cuts,name)


# FOR THE MC
model="BDT/data/BDT.json"
variables="BDT/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root"
tree="DecayTree"
cuts="Bu_BKGCAT==0"
name="mc"

BDT_Applying(model,variables,file,tree,cuts,name)