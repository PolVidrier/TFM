import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split 
import yaml
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def BDT_GridSearch(mcfile,mctree,cutsmc,datafile,datatree,cutsdata,variables,name):
    # mcfile= name of the .root file for the mc
    # mctree= name of the tree inside the mcfile
    # cutsmc= cuts applied to the mc, "" if none
    # datafile= name of the .root file for the data
    # datatree= name of the tree inside the datafile
    # cutsdata= cuts applied to the data, "" if none
    # variables= variables for the training, in .yaml file
    # name= name for the BDT model

    # We open the files
    mc=uproot.open(mcfile)[mctree]
    data=uproot.open(datafile)[datatree]

    features=[]

    with open(variables,"r") as file:
        dictionary=yaml.safe_load(file)

    for var in dictionary:
        features.append(var)

    if cutsmc=="":
        mcpd=mc.arrays(features,library="pd")
    else:
        mcpd=mc.arrays(features,cutsmc,library="pd")
    if cutsdata=="":
        datapd=data.arrays(features,library="pd")
    else:
        datapd=data.arrays(features,cutsdata,library="pd")

    featuresminusM=features.copy()
    # We delete the masses
    del mcpd["Jpsi_M"]   
    del datapd["Jpsi_M"]
    del mcpd["Bu_M"]
    del datapd["Bu_M"]
    featuresminusM.remove("Jpsi_M")
    featuresminusM.remove("Bu_M")

    signal=mcpd.to_numpy()
    bkg=datapd.to_numpy()

    print("Signal shape: ",signal.shape[0])
    print("Background shape: ",bkg.shape[0])

    X=np.concatenate((signal,bkg))
    y=np.concatenate((np.ones(signal.shape[0]),np.zeros(bkg.shape[0])))

    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3)
    print("Train size: ", X_train.shape[0])
    print("Test size: ", X_test.shape[0])

    # A parameter grid for XGBoost
    #params = {
    #        'eta':[0.1, 0.3, 0.5],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'min_child_weight': [0, 1, 2],    # the larger the more conservative, default=1
    #        'gamma': [0, 0.5, 1],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [4, 5, 6, 7],    # depth of the tree, more more overfitt, default=6
    #        'n_estimators':[10,25,50,75,100,150]   # number of trees, the fewer the less overtraining, default=100
    #        }
    
    #params = {
    #        'eta':[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'min_child_weight': [0, 0.5, 1, 1.5, 2],    # the larger the more conservative, default=1
    #        'gamma': [0, 0.25, 0.5, 0.75, 1],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [3, 4, 5, 6, 7],    # depth of the tree, more more overfitt, default=6
    #        'n_estimators':[50, 75, 100, 125, 150, 175]   # number of trees, the fewer the less overtraining, default=100
    #        }
    
    #params = {
    #        'eta':[0.05, 0.1, 0.2, 0.3, 0.4],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'min_child_weight': [0, 0.5, 1, 1.5],    # the larger the more conservative, default=1
    #        'gamma': [0, 0.5, 0.75, 1],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [2, 3, 4, 5, 6],    # depth of the tree, more more overfitt, default=6
    #        'n_estimators':[ 75, 100, 125, 150, 175, 200]   # number of trees, the fewer the less overtraining, default=100
    #        }

    #params = {
    #        'eta':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'gamma': [0, 0.5, 0.75, 1, 1.25, 1.5],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],    # depth of the tree, more more overfitt, default=6
    #        'min_child_weight': [0, 0.5, 1, 1.5, 2],    # the larger the more conservative, default=1
    #        'max_delta_step': [0, 1, 2, 3, 4, 5],      # 0 no constrain, other makes more conservative, default=0
    #        'subsample': [0.1, 0.25, 0.5, 0.75, 1],    # can prevent overtraining, default=1
    #        'n_estimators': [50, 75, 100, 125, 150, 175, 200, 225, 250]   # number of trees, the fewer the less overtraining, default=100
    #        } # TOO LONG

    #params = {
    #        'eta':[0.3, 0.4, 0.5, 0.6],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'gamma': [0, 0.5, 0.75],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [1, 2, 3, 4, 5],    # depth of the tree, more more overfitt, default=6
    #        'min_child_weight': [0, 0.5, 1, 2],    # the larger the more conservative, default=1
    #        'max_delta_step': [0, 1, 2, 4],      # 0 no constrain, other makes more conservative, default=0
    #        'subsample': [0.25, 0.75, 1],    # can prevent overtraining, default=1
    #        'n_estimators': [150, 175, 200, 225, 250]   # number of trees, the fewer the less overtraining, default=100
    #        }
    
    #params = {
    #        'eta':[0.1, 0.2, 0.3, 0.4],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'gamma': [0, 0.25, 0.5],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [1, 2, 3, 4],    # depth of the tree, more more overfitt, default=6
    #        'min_child_weight': [0, 0.25, 0.5],    # the larger the more conservative, default=1
    #        'max_delta_step': [0, 1, 2, 4],      # 0 no constrain, other makes more conservative, default=0
    #        'subsample': [0.25, 0.5, 0.75, 1],    # can prevent overtraining, default=1
    #        'n_estimators': [200, 225, 250, 275, 300, 325, 350]   # number of trees, the fewer the less overtraining, default=100
    #        } ERROR IN THE NIGHT

    #params = {
    #        'eta':[0.1, 0.2, 0.3, 0.4],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'gamma': [0, 0.25, 0.5],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [1, 2, 3, 4],    # depth of the tree, more more overfitt, default=6
    #        'min_child_weight': [0, 0.25],    # the larger the more conservative, default=1
    #        'max_delta_step': [0, 1, 2],      # 0 no constrain, other makes more conservative, default=0
    #        'subsample': [0.5, 0.75, 1],    # can prevent overtraining, default=1
    #        'n_estimators': [225, 250, 275, 300, 325, 350]   # number of trees, the fewer the less overtraining, default=100
    #        }
    
    #params = {
    #        'eta':[0.1, 0.2, 0.3, 0.4],   # learning rate, the smaller prevents overfitting, default=0.3
    #        'gamma': [0, 0.25, 0.5],      # min_split_loss, the larger the more conservative, default=0
    #        'max_depth': [1, 2, 3, 4, 5, 6],    # depth of the tree, more more overfitt, default=6
    #        'min_child_weight': [0],    # the larger the more conservative, default=1
    #        'max_delta_step': [0, 1, 2, 3, 4],      # 0 no constrain, other makes more conservative, default=0
    #        'subsample': [0.75],    # can prevent overtraining, default=1
    #        'n_estimators': [225, 250, 275, 300]   # number of trees, the fewer the less overtraining, default=100
    #        }

    params = {
            'eta':[0.2],   # learning rate, the smaller prevents overfitting, default=0.3
            'gamma': [0, 0.25, 0.5],      # min_split_loss, the larger the more conservative, default=0
            'max_depth': [1, 2, 3, 4, 5, 6],    # depth of the tree, more more overfitt, default=6
            'min_child_weight': [0],    # the larger the more conservative, default=1
            'max_delta_step': [0, 1, 2, 3],      # 0 no constrain, other makes more conservative, default=0
            'subsample': [0.75],    # can prevent overtraining, default=1
            'n_estimators': [225, 250, 275, 300, 325]   # number of trees, the fewer the less overtraining, default=100
            }


    print("Training...")
    start_time = time.time()

    # Training the model
    xgb_classifier = xgb.XGBClassifier(max_depth=6) # the default 
    grid_search = GridSearchCV(estimator=xgb_classifier,param_grid=params,scoring = 'roc_auc',n_jobs = -1,cv = 5,verbose=3) #n-jobs is the parallel procress cpu things, cv is crossval 5 is default and verbose is to show the results
    #grid_search = GridSearchCV(estimator=xgb_classifier,param_grid=params,scoring = 'neg_log_loss',n_jobs = -1,cv = 5,verbose=3) #n-jobs is the parallel procress cpu things, cv is crossval 5 is default and verbose is to show the results
    grid_search.fit(X_train, y_train)
    print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_search.best_score_)  #mean cros validated score
    print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time:.2f} seconds")


mcfile="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root"
mctree="DecayTree"
cutsmc="Bu_BKGCAT==0"
cutsmcplus="(Bu_BKGCAT==0) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005)"
datafile="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
datatree="tuple_B2JpsiKRD/DecayTree;1"
cutsdata="Jpsi_M > 3200"
cutsdataplus="(Jpsi_M>3200) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005) & ((Hlt1_Hlt1DisplacedDielectronDecision==1) | (Hlt1_Hlt1DisplacedLeptonsDecision==1) | (Hlt1_Hlt1LowMassNoipDielectronDecision==1) | (Hlt1_Hlt1SingleHighEtDecision==1) | (Hlt1_Hlt1SingleHighPtElectronDecision==1) | (Hlt1_Hlt1TrackElectronMVADecision==1))"
variables="BDT/data/variableselection.yaml"
name="BDT"

BDT_GridSearch(mcfile,mctree,cutsmcplus,datafile,datatree,cutsdataplus,variables,name)