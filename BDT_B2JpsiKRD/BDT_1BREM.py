import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split 
import yaml

mc=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/mc/00170282/0000/00170282_00000001_1.tuple_bu2kee.root")["Tuple/DecayTree;1"]
data=uproot.open(r"/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root")["tuple_B2JpsiKRD/DecayTree;1"]

features=[]
featuresmc=[]

with open("BDT_B2JpsiKRD/variableselection.yaml","r") as file:
    dictionary=yaml.safe_load(file)

for var in dictionary:
    varmc=var.replace("ep","L1")
    varmc=varmc.replace("em","L2")
    features.append(var)
    featuresmc.append(varmc)

mcpd=mc.arrays(featuresmc,"(Bu_BKGCAT==0) & (((L1_HASBREM==True) & (L2_HASBREM==False))|((L1_HASBREM==False) & (L2_HASBREM==True)))",library="pd")
datapd=data.arrays(features,"(Jpsi_M > 3200) & (((ep_HASBREM==True) & (em_HASBREM==False))|((ep_HASBREM==False) & (em_HASBREM==True)))",library="pd")

#for var in dictionary:
#    varmc=var.replace("ep","L1")
#    varmc=varmc.replace("em","L2")
#    if type(dictionary[var][0])!=bool:
#        plt.hist(mcpd[varmc], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
#        plt.hist(datapd[var], range=(dictionary[var][0],dictionary[var][1]), bins=100, density=True, alpha=0.5)
#        plt.title("{}".format(var))
#        plt.xlabel("{}".format(var))
#        plt.legend(["MC","data"])
#        plt.savefig("BDT_B2JpsiKRD/plots/Usedvariables/{}.png".format(var))
#        plt.close()
#    else:
#        plt.hist(mcpd[varmc], range=(0,1),bins=2, density=True, alpha=0.5)
#        plt.hist(datapd[var], range=(0,1), bins=2, density=True, alpha=0.5)
#        plt.title("{}".format(var))
#        plt.xlabel("{}".format(var))
#        plt.legend(["MC","data"])
#        plt.savefig("BDT_B2JpsiKRD/plots/Usedvariables/{}.png".format(var))
#        plt.close()

featuresminusM=features.copy()
# for later
mcmas_test = mcpd[["Jpsi_M"]]   #for the prediction of mass
mcmas_test_array = mcmas_test.to_numpy()
mcmas_test = mcmas_test_array[:,0].tolist()

#we delete the mass
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
print("Training...")
# training the model
xgb_classifier = xgb.XGBClassifier() 
xgb_classifier.fit(X_train,y_train)
print("Ready")

# FEATURE IMPORTANCE
print("Plotting feature importance")
xgb_classifier.get_booster().feature_names = featuresminusM
xgb.plot_importance(xgb_classifier.get_booster())
plt.yticks(fontsize=4.5)
plt.ylabel(None)
plt.grid(False)
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/featureimportance.png", dpi=300)
plt.close()


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
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/predictions.png")
plt.close()

# ROC CURVE
print("Ploting ROC curve")
truething=pd.Series(y_test).values
truethingtrain=pd.Series(y_train).values

from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(truething, pred_prob[:,1])
roc_auc = auc(fpr, tpr)

fprtrain, tprtrain, thresholdtrain = roc_curve(truethingtrain, pred_probtrain[:,1])
roc_auctrain = auc(fprtrain, tprtrain)

plt.figure()
plt.plot(fpr, tpr, color="darkorange",lw=2, label="XGB-AUC_test= %0.5f" % roc_auc,)
plt.plot(fprtrain, tprtrain, color="darkgreen",lw=2, label="XGB-AUC_train= %0.5f" % roc_auctrain,)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.legend(loc="lower right")
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/ROC.png")
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
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/FoM.png")
plt.close()

# PREDICTION OF THE JPSI MASS
print("Prediction of the Jpsi mass")
# Data without the window cut
fulldatapd=data.arrays(features,"(Jpsi_M>100) & (((ep_HASBREM==True) & (em_HASBREM==False))|((ep_HASBREM==False) & (em_HASBREM==True)))",library="pd")

mas_test = fulldatapd[["Jpsi_M"]]   #for the prediction of mass
mas_test_array = mas_test.to_numpy()

masB_test = fulldatapd[["Bu_M"]]   #for the prediction of mass
masB_test_array = masB_test.to_numpy()


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

plt.hist(mas_pred,label="Prediction", bins=100)
plt.title("Jpsi_M Prediction")
plt.xlabel("Jpsi_M")
plt.legend(loc="upper left")
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/Jpsi_M.png")
plt.close()

with open('BDT_B2JpsiKRD/data_Jpsi_M_1BREM.txt', 'w') as f:
    for item in mas_pred:
        f.write("%s\n" % item)
f.close

plt.hist(mas_pred,label="Prediction", bins=100,density=True,alpha=0.5)
plt.hist(mcmas_test, label="mc",bins=100,density=True, alpha=0.5)
plt.title("Jpsi_M Comparison")
plt.xlabel("Jpsi_M")
plt.legend(loc="upper left")
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/Jpsi_M_mc_comparison.png")
plt.close()

# PREDICTION OF THE B MASS
print("Prediction of the B mass")

masB_pred, masB_test = [], masB_test_array[:,0].tolist()


for i in range(0,len(datapred_prob)):
    if datapred_prob[i] >= bestcut:    #Signal
        masB_pred.append(masB_test[i])

plt.hist(masB_pred,label="Prediction", bins=100)
plt.title("Bu_M Prediction")
plt.xlabel("Bu_M")
plt.legend(loc="upper left")
plt.savefig("BDT_B2JpsiKRD/plots/1BREM/Bu_M.png")
plt.close()