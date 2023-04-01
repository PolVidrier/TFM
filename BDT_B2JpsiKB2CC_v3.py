import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split 
import yaml

mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/00170282/0000/00170282_00000001_1.tuple_bu2kee.root")["Tuple/DecayTree;1"]
data=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root")["tuple_B2JpsiKB2CC/DecayTree;1"]

features=[]

with open("private/BDT/variableselectionB2JpsiKB2CC_v3.yaml","r") as file:
    dictionary=yaml.safe_load(file)

for var in dictionary:
    features.append(var)

mcpd=mc.arrays(features,"Bu_BKGCAT==0",library="pd")
datapd=data.arrays(features,"Jpsi_M > 3200",library="pd")

for var in dictionary:
    plt.hist(mcpd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
    plt.hist(datapd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
    plt.title("{}".format(var))
    plt.xlabel("{}".format(var))
    plt.legend(["MC","data"])
    plt.savefig("private/plots/plotsBDT/B2JpsiKB2CC_v3/variables/{}.png".format(var))
    plt.close()


del mcpd["Bu_END_VX"]
del datapd["Bu_END_VX"]
del mcpd["Bu_END_VY"]
del datapd["Bu_END_VY"]
del mcpd["Bu_END_VZ"]
del datapd["Bu_END_VZ"]



#we delete the mass
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
# training the model
xgb_classifier = xgb.XGBClassifier() 
xgb_classifier.fit(X_train,y_train)
print("Ready")

predictions = xgb_classifier.predict(X_test)  #predictions
pred_prob = xgb_classifier.predict_proba(X_test)  #probability of predictions  

predictionstrain = xgb_classifier.predict(X_train)  #predictions of the training
pred_probtrain = xgb_classifier.predict_proba(X_train)  #probability of predictions of the training 

# SIGNALS PREDICTED
pred_probsignal=pred_prob[:,1][y_test==1] #just half of it, it is doubled
# BACKGROUND PREDICTED
pred_probbkg=pred_prob[:,1][y_test==0] 


# SIGNALS PREDICTED
pred_probsignaltrain=pred_probtrain[:,1][y_train==1] 
# BACKGROUND PREDICTED
pred_probbkgtrain=pred_probtrain[:,1][y_train==0] 

# PLOTING ALL PREDICTIONS
print("Ploting predictions")
bin=np.linspace(0,1,50)
plt.hist(pred_probsignal,bins=bin,density=True,label="Sig Test",histtype="step",color="Blue") 
plt.hist(pred_probbkg,bins=bin,density=True,label="Bkg Test",histtype="step",color="Red") 
plt.hist(pred_probsignaltrain,bins=bin,density=True, alpha=0.5,label="Sig Train",color="Green")  
plt.hist(pred_probbkgtrain,bins=bin,density=True,alpha=0.5,label="Bkg Train",color="Orange")  
plt.title("Predictions")
plt.legend(loc="upper center")
plt.xlabel("BDT Output")
plt.ylabel("Arbitrary units")
plt.savefig("private/plots/plotsBDT/B2JpsiKB2CC_v3/predictions.png")
plt.close()

# ROC CURVE
print("Ploting ROC curve")
truething=pd.Series(y_test).values

from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(truething, pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange",lw=2, label="XGB-AUC= %0.5f" % roc_auc,)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.legend(loc="lower right")
plt.savefig("private/plots/plotsBDT/B2JpsiKB2CC_v3/ROC.png")
plt.close()

# FIGURE OF MERIT PLOT
print("Ploting FoM")
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
plt.figure()
plt.errorbar(cuts,fom,yerr=errorFoM,label="FoM, Best Cut=%f" % bestcut,elinewidth=0.5,errorevery=10)
plt.xlabel("BDT Cut")
plt.ylabel("FoM")
plt.title("FoM")
plt.legend(loc="lower center")
plt.savefig("private/plots/plotsBDT/B2JpsiKB2CC_v3/FoM.png")
plt.close()

# PREDICTION OF THE MASS
print("Prediction of the mass")
# Data without the window cut
fulldatapd=data.arrays(features,library="pd")

mas_test = fulldatapd[["Jpsi_M"]]   #for the prediction of mass
mas_test_array = mas_test.to_numpy()

del fulldatapd["Bu_END_VX"]
del fulldatapd["Bu_END_VY"]
del fulldatapd["Bu_END_VZ"]


del fulldatapd["Jpsi_M"]   #we delete the mass
del fulldatapd["Bu_M"]

data=fulldatapd.to_numpy()

mas_pred, mas_test = [], mas_test_array[:,0].tolist()

datapredictions = xgb_classifier.predict(data)  #predictions
datapred_prob = xgb_classifier.predict_proba(data)  #probability of predictions  

datapred_prob=datapred_prob[:,1]

for i in range(0,len(datapred_prob)):
    if datapred_prob[i] >= bestcut:    #Signal
        mas_pred.append(mas_test[i])

plt.hist(mas_pred,label="Prediction", bins=100, density=True)
plt.title("Jpsi_M")
plt.xlabel("Jpsi_M")
plt.legend(loc="upper right")
plt.savefig("private/plots/plotsBDT/B2JpsiKB2CC_v3/Jpsi_M.png")
plt.close()
