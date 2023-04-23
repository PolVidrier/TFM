import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import yaml
import ROOT
import time

def BDT_Applying(model,variables,file,tree,Has_ep_em,cuts,name):
    xgb_classifier = xgb.XGBClassifier(max_depth=6) # the default 
    xgb_classifier.load_model(model)

    data=uproot.open(file)[tree]
    features=[]
    featuresmc=[]

    with open(variables,"r") as file:
        dictionary=yaml.safe_load(file)

    for var in dictionary:
        varmc=var.replace("ep","L1")
        varmc=varmc.replace("em","L2")
        features.append(var)
        featuresmc.append(varmc)

    if Has_ep_em==True:
        if cuts=="":
            fulldatapd=data.arrays(features,library="pd")
        else:
            fulldatapd=data.arrays(features,cuts,library="pd")
    else:
        if cuts=="":
            fulldatapd=data.arrays(featuresmc,library="pd")
        else:
            fulldatapd=data.arrays(featuresmc,cuts,library="pd")

    del fulldatapd["Jpsi_M"]   #we delete the mass
    del fulldatapd["Bu_M"]

    data=fulldatapd.to_numpy()

    datapred_prob = xgb_classifier.predict_proba(data)  #probability of predictions  

    datapred_prob=datapred_prob[:,1]
    print(datapred_prob)

    with open('BDT/BDT_output_%s.txt'% name, 'w') as f:
        for i in range(len(datapred_prob)):
            f.write(str(datapred_prob[i])+"\n")
    f.close

    # ATTEMPT WITH ROOT
    # create a new ROOT file and a TTree object
    output_file = ROOT.TFile("BDT/BDT_output_%s.root" % name, "RECREATE")
    tree = ROOT.TTree("DecayTree", "My Tree")

    #tarray = ROOT.TArrayF(datapred_prob.size)
    #for i in range(datapred_prob.size):
    #    tarray[i] = datapred_prob[i]

    # create a branch in the tree for your numpy array
    branch = tree.Branch("BDT_output", datapred_prob, "BDT_output/F")

    # loop over the numpy array and fill the tree with the values
    for i in range(len(datapred_prob)):
        branch.Fill()
        if i < 5:  # Print the values for the first five events
            print("output:", tree.BDT_output)

    # write the tree to the ROOT file and close the file
    tree.Write()
            
    output_file.Close()


# FOR THE DATA
model="BDT/BDT.json"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
Has_ep_em=True
cuts="Jpsi_M>100"
name="data"

BDT_Applying(model,variables,file,tree,Has_ep_em,cuts,name)


# FOR THE MC
model="BDT/BDT.json"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root"
tree="Tuple/DecayTree;1"
Has_ep_em=False
cuts="Bu_BKGCAT==0"
name="mc"

BDT_Applying(model,variables,file,tree,Has_ep_em,cuts,name)