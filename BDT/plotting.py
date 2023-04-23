import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import yaml
import ROOT
import time

def plotting(bestcut,variables,file,tree,Has_ep_em,cuts,name):

    with open(bestcut,"r") as f:
        bestcut=float(f.read())
    f.close

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


    data=fulldatapd.to_numpy()

    with open('BDT/BDT_output_%s.txt'% name, 'r') as f:
        datapred_prob=[float(output) for output in f.readlines()]
    f.close

    mas_test = fulldatapd[["Jpsi_M"]]   #for the prediction of mass
    mas_test_array = mas_test.to_numpy()
    mas_pred, mas_test = [], mas_test_array[:,0].tolist()

    for i in range(len(datapred_prob)):
        if datapred_prob[i] >= bestcut:    #Signal
            mas_pred.append(mas_test[i])

    plt.hist(mas_pred,label="Prediction", bins=100)
    plt.title("Jpsi_M %s Prediction" % name)
    plt.xlabel("Jpsi_M")
    plt.legend(loc="upper left")
    plt.savefig("BDT/plots/Jpsi_M_%s.png" % name)
    plt.close()



# FOR THE DATA
bestcut="BDT/BDT_Best_cut.txt"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
Has_ep_em=True
cuts="Jpsi_M>100"
name="data"

plotting(bestcut,variables,file,tree,Has_ep_em,cuts,name)

# FOR THE MC
bestcut="BDT/BDT_Best_cut.txt"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root"
tree="Tuple/DecayTree;1"
Has_ep_em=False
cuts="Bu_BKGCAT==0"
name="mc"

plotting(bestcut,variables,file,tree,Has_ep_em,cuts,name)

