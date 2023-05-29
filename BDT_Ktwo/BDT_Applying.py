import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import yaml
from root_numpy import array2root,tree2array
import ROOT
import os


def BDT_Applying(model,bestcut,variables,file,tree,cuts,name):
    # model= name of the BDT model file, a .json file
    # variables= variables for the training, in .yaml file
    # file= name of the .root file
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # name= name for the BDT_output

    # We load the model
    xgb_classifier = xgb.XGBClassifier(max_depth=6) # the default 
    xgb_classifier.load_model(model)

    datauproot=uproot.open(file)[tree]
    features=[]

    with open(variables,"r") as f:
        dictionary=yaml.safe_load(f)
    f.close()

    for var in dictionary:
        features.append(var)

    if cuts=="":
        fulldatapd=datauproot.arrays(features,library="pd")
    else:
        fulldatapd=datauproot.arrays(features,cuts,library="pd")

    del fulldatapd["Jpsi_M"]   #we delete the mass
    del fulldatapd["Bu_M"]

    data=fulldatapd.to_numpy()

    datapred_prob = xgb_classifier.predict_proba(data)  #probability of predictions  

    datapred_prob=datapred_prob[:,1]

    # We write the output in a .root file
    datapred_prob2=datapred_prob.copy()
    datapred_prob2.dtype=[("BDT_output",np.float32)]

    output_file="BDT_Ktwo/data/BDT_output_%s.root" % name
    output_tree="DecayTree"
    array2root(datapred_prob2,output_file,output_tree,mode="RECREATE")
    
    # We read the best cut
    with open(bestcut,"r") as f:
        bestcut=float(f.read())
    f.close

    # We save the result in an new root file with the best cut applied
    allfeatures=datauproot.keys()
    if cuts=="":
        alldatapd=datauproot.arrays(allfeatures,library="pd")
    else:
        alldatapd=datauproot.arrays(allfeatures,cuts,library="pd")

    output_file="/eos/lhcb/user/p/pvidrier/roots/%s_with_cuts.root" % name
    output_tree="DecayTree"
    
    #os.remove("/eos/lhcb/user/p/pvidrier/roots/%s_with_cuts.root" % name) # in case it exists

    for var in allfeatures:
        alldata=alldatapd[[var]]
        array=alldata.to_numpy()
        arraycut=[]
        for i in range(len(datapred_prob)):
            if datapred_prob[i] >= bestcut:    #Signal
                arraycut.append(array[i])
        arraycut=np.array(arraycut)
        arraycut.dtype=[("%s" % var, np.result_type(arraycut))]
        if var==allfeatures[0]:
            array2root(arraycut,output_file,output_tree,mode="RECREATE") # We create an empy root so we don't add every time we execute this script
        else:
            array2root(arraycut,output_file,output_tree,mode="UPDATE")  # It will add all the variables

    # FAILED ATTEMPT WITH TREE2ARRAY
    #rfile = ROOT.TFile.Open(file,"READ")
    #treee = rfile.Get(tree)
    #if (cuts!=""): treee = treee.CopyTree(cuts)
    #branch_list = treee.GetListOfBranches()
    #allvariables=[]

    #for i in range(branch_list.GetEntries()):
    #    branch = branch_list.At(i)
    #    allvariables.append(branch.GetName())

    #array = tree2array(treee)
    #arraycuts=[]
    

    #for var in allvariables:
    #    arraycut=[]
    #    for i in range(len(datapred_prob)):
    #        if datapred_prob[i] >= bestcut:    #Signal
    #            arraycut.append(array[var][i])
    #    arraycut=np.array(arraycut)
    #    arraycut.dtype=[("%s" % var, np.result_type(array[var]))]
    #    arraycuts.append(arraycut)

    #arraycuts=np.array(arraycuts)
    #array2root(arraycuts,output_file,output_tree,mode="RECREATE")
    print("BDT applied to %s" % name)


# FOR THE DATA
model="BDT_Ktwo/data/BDT.json"
bestcut="BDT_Ktwo/data/BDT_Best_cut.txt"
variables="BDT_Ktwo/data/variableselection.yaml"
file="/eos/lhcb/user/p/pvidrier/roots/B2JpsiKLFU_MagAll_presel.root"
tree="DecayTree;1"
cutsplus="Jpsi_M>100"
#cutsplus="(Jpsi_M>100) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005) & ((Hlt1_Hlt1DisplacedDielectronDecision==1) | (Hlt1_Hlt1DisplacedLeptonsDecision==1) | (Hlt1_Hlt1LowMassNoipDielectronDecision==1) | (Hlt1_Hlt1SingleHighEtDecision==1) | (Hlt1_Hlt1SingleHighPtElectronDecision==1) | (Hlt1_Hlt1TrackElectronMVADecision==1))"
cutsforROOT="Jpsi_M>100 && angle(em,ep)>0.0005 && (Hlt1_Hlt1DisplacedDielectronDecision || Hlt1_Hlt1DisplacedLeptonsDecision || Hlt1_Hlt1LowMassNoipDielectronDecision || Hlt1_Hlt1SingleHighEtDecision || Hlt1_Hlt1SingleHighPtElectronDecision || Hlt1_Hlt1TrackElectronMVADecision)"
name="data_Ktwo"  

BDT_Applying(model,bestcut,variables,file,tree,cutsplus,name)


# FOR THE MC/SIGNAL
model="BDT_Ktwo/data/BDT.json"
bestcut="BDT_Ktwo/data/BDT_Best_cut.txt"
variables="BDT_Ktwo/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root"  # a vegades em surt error i haig de borrar el document abans
tree="DecayTree"
cuts="Bu_BKGCAT==0" 
cutsplus="(Bu_BKGCAT==0) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005)" 
name="mc_Ktwo"

BDT_Applying(model,bestcut,variables,file,tree,cutsplus,name)











