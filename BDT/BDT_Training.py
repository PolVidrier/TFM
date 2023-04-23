import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import yaml
import time

def BDT_Training(mcfile,mctree,cutsmc,datafile,datatree,cutsdata,variables,name):
    mc=uproot.open(mcfile)[mctree]
    data=uproot.open(datafile)[datatree]

    features=[]
    featuresmc=[]

    with open(variables,"r") as file:
        dictionary=yaml.safe_load(file)

    for var in dictionary:
        varmc=var.replace("ep","L1")
        varmc=varmc.replace("em","L2")
        features.append(var)
        featuresmc.append(varmc)

    if cutsmc=="":
        mcpd=mc.arrays(featuresmc,library="pd")
    else:
        mcpd=mc.arrays(featuresmc,cutsmc,library="pd")
    if cutsdata=="":
        datapd=data.arrays(features,library="pd")
    else:
        datapd=data.arrays(features,cutsdata,library="pd")

    del mcpd["Jpsi_M"]   
    del datapd["Jpsi_M"]
    del mcpd["Bu_M"]
    del datapd["Bu_M"]

    signal=mcpd.to_numpy()
    bkg=datapd.to_numpy()

    print("Signal shape: ",signal.shape[0])
    print("Background shape: ",bkg.shape[0])

    X=np.concatenate((signal,bkg))
    y=np.concatenate((np.ones(signal.shape[0]),np.zeros(bkg.shape[0])))

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3)
    print("Train size: ", X_train.shape[0])
    print("Test size: ", X_test.shape[0])
    print("Training...")
    start_time = time.time()
    # training the model
    xgb_classifier = xgb.XGBClassifier(max_depth=6) # the default 
    xgb_classifier.fit(X_train,y_train)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time:.2f} seconds")
    xgb_classifier.save_model("BDT/%s.json" % name)

    pred_prob = xgb_classifier.predict_proba(X_test)  #probability of predictions  

    # SIGNALS PREDICTED
    pred_probsignal=pred_prob[:,1][y_test==1] #just half of it, it is doubled
    # BACKGROUND PREDICTED
    pred_probbkg=pred_prob[:,1][y_test==0] 

    # FIGURE OF MERIT PLOT
    print("Computing best cut")
    a=0.1
    fom=[]
    errorFoM=[]
    cuts=np.linspace(0,0.99,1000)
    for cut in cuts:
        S=0
        B=0
        for varsig in pred_probsignal:
            if varsig>=cut:
                S+=1
        for varbck in pred_probbkg:
            if varbck>=cut:
                B+=1
        fom.append(a*S/np.sqrt(a*S+B))
        errorFoM.append(np.sqrt(((a*(a*S+2*B)/(2*((a*S+B)**(3./2))))*np.sqrt(S))**2.+((a*S/(2*((B+a*S)**(3./2))))*np.sqrt(B))**2.))

    bestcut=cuts[fom.index(max(fom))]

    with open('BDT/%s_Best_cut.txt' % name, 'w') as f:
            f.write(str(bestcut))
    f.close


mcfile="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root"
mctree="Tuple/DecayTree;1"
cutsmc="Bu_BKGCAT==0"
datafile="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
datatree="tuple_B2JpsiKRD/DecayTree;1"
cutsdata="Jpsi_M > 3200"
variables="BDT/variableselection.yaml"
name="BDT"

BDT_Training(mcfile,mctree,cutsmc,datafile,datatree,cutsdata,variables,name)