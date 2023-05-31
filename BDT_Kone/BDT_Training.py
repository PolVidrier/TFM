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


def BDT_Training(mcfile,mctree,cutsmc,datafile,datatree,cutsdata,variables,name):
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
    print("Training...")
    start_time = time.time()

    # Training the model
    #xgb_classifier = xgb.XGBClassifier(max_depth=6) # the default 
    #xgb_classifier = xgb.XGBClassifier(n_estimators=25) # let's try to reduce overtraining, it works
    #xgb_classifier=xgb.XGBClassifier(eta=0.1,gamma=1,max_depth=4,min_child_weight=0,n_estimators=150) #optimized with neg_log_loss
    #xgb_classifier=xgb.XGBClassifier(eta=0.1,gamma=1,max_depth=4,min_child_weight=2,n_estimators=150) #optimized with roc_auc THE 2n BEST
    #xgb_classifier=xgb.XGBClassifier(eta=0.1,gamma=0.5,max_depth=5,min_child_weight=1,n_estimators=150) #optimized with roc_auc, 2n try it is worse
    #xgb_classifier=xgb.XGBClassifier(eta=0.2,gamma=0.75,max_depth=3,min_child_weight=0,n_estimators=175) #optimized with roc_auc, changing values, THE 3rd BEST
    #xgb_classifier=xgb.XGBClassifier(eta=0.4,gamma=0,max_depth=2,min_child_weight=0,n_estimators=200) #optimized with roc_auc, changing values2, THE BEST
    xgb_classifier=xgb.XGBClassifier(eta=0.2,gamma=0.25,max_delta_step=0,max_depth=2,min_child_weight=0,n_estimators=275,subsample=0.75) # THE BEST

    xgb_classifier.fit(X_train,y_train)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time:.2f} seconds")
    xgb_classifier.save_model("BDT_Kone/data/%s.json" % name)

    pred_prob = xgb_classifier.predict_proba(X_test)  #probability of predictions  
    pred_probtrain = xgb_classifier.predict_proba(X_train)  #probability of predictions of the training 

    # SIGNALS PREDICTED
    pred_probsignal=pred_prob[:,1][y_test==1] #just half of it, it is doubled
    # BACKGROUND PREDICTED
    pred_probbkg=pred_prob[:,1][y_test==0] 

    # SIGNALS PREDICTED FOR THE TRAINING
    pred_probsignaltrain=pred_probtrain[:,1][y_train==1] 
    # BACKGROUND PREDICTED FOR THE TRAINING
    pred_probbkgtrain=pred_probtrain[:,1][y_train==0] 

    # POSITIVES AND NEGATIVES
    positives=len(pred_probsignal)
    negatives=len(pred_probbkg)
    positivestrain=len(pred_probsignaltrain)
    negativestrain=len(pred_probbkgtrain)

    # one negative classified as positive, one positive classified as negative
    fprtest=[1./negatives,0.]
    tprtest=[1.,(positives-1)/float(positives)]
    area=1-(tprtest[0]-tprtest[1])*(fprtest[0]-fprtest[1])/2.
    fprtesttrain=[1./negativestrain,0.]
    tprtesttrain=[1.,(positivestrain-1)/float(positivestrain)]
    areatrain=1-(tprtesttrain[0]-tprtesttrain[1])*(fprtesttrain[0]-fprtesttrain[1])/2.

    # ACCURACY SCORE (Needed?)
    # make predictions for test data
    y_pred = xgb_classifier.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


    # FEATURE IMPORTANCE
    print("Plotting Feature Importance")
    xgb_classifier.get_booster().feature_names = featuresminusM
    xgb.plot_importance(xgb_classifier.get_booster())
    plt.yticks(fontsize=4.5)
    plt.ylabel(None)
    plt.grid(False)
    plt.savefig("BDT_Kone/plots/BDT/Feature_Importance_%s.png" % name, dpi=300)
    plt.close()


    # PLOTING ALL PREDICTIONS
    print("Plotting Predictions")
    bin=np.linspace(0,1,50)
    plt.hist(pred_probsignal,bins=bin,density=True,label="Sig Test",histtype="step",color="Blue") 
    plt.hist(pred_probbkg,bins=bin,density=True,label="Bkg Test",histtype="step",color="Red") 
    plt.hist(pred_probsignaltrain,bins=bin,density=True, alpha=0.5,label="Sig Train",color="Green")  
    plt.hist(pred_probbkgtrain,bins=bin,density=True,alpha=0.5,label="Bkg Train",color="Orange")  
    plt.title("Predictions")
    plt.legend(loc="upper center")
    plt.xlabel("BDT Output")
    plt.ylabel("Arbitrary units")
    plt.savefig("BDT_Kone/plots/BDT/Predictions_%s.png" % name)
    plt.close()


    # ROC CURVE
    print("Plotting ROC curve")
    truething=pd.Series(y_test).values
    truethingtrain=pd.Series(y_train).values

    fpr, tpr, threshold = roc_curve(truething, pred_prob[:,1])
    roc_auc = auc(fpr, tpr)

    fprtrain, tprtrain, thresholdtrain = roc_curve(truethingtrain, pred_probtrain[:,1])
    roc_auctrain = auc(fprtrain, tprtrain)

    # calculate the acceptance (recall or sensitivity) at a specific FPR value
    fpr_threshold = 0.05  # set a specific FPR value
    idx = np.argmin(np.abs(fpr - fpr_threshold))
    acceptance = tpr[idx]
    print("Acceptance: %.2f%%" % (acceptance * 100.0))

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange",lw=2, label="XGB-AUC_test= %0.8f" % roc_auc,)
    plt.plot(fprtrain, tprtrain, color="darkgreen",lw=2, label="XGB-AUC_train= %0.8f" % roc_auctrain,)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig("BDT_Kone/plots/BDT/ROC_%s.png" % name)
    plt.close()

    # MORE FOCUSED ROC
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange",lw=2, label="XGB-AUC_test= %0.8f" % roc_auc,)
    plt.plot(fprtrain, tprtrain, color="darkgreen",lw=2, label="XGB-AUC_train= %0.8f" % roc_auctrain,)
    plt.xlim([0.0, 0.25])
    plt.ylim([0.75, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig("BDT_Kone/plots/BDT/ROC_corner_%s.png" % name)
    plt.close()

    # DELTA AUC
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange",lw=2, label="XGB-AUC_test= %0.8f" % roc_auc,)
    plt.plot(fprtest, tprtest, color="red",lw=2, label="1 misclassified, AUC_test= %0.8f" % area)
    plt.plot(fprtesttrain, tprtesttrain, color="blue",lw=2, label="1 misclassified, AUC_train= %0.8f" % areatrain)
    plt.plot(fprtrain, tprtrain, color="darkgreen",lw=2, label="XGB-AUC_train= %0.8f" % roc_auctrain,)
    plt.xlim([0.0, 0.01])
    plt.ylim([0.99, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig("BDT_Kone/plots/BDT/ROC_error_%s.png" % name)
    plt.close()

    # FIGURE OF MERIT
    print("Plotting Figure of Merit")
    a=0.1
    fom=[]
    errorFoM=[]
    cuts=np.linspace(0,0.9999,1000)
    for cut in cuts:
        S=0
        B=0
        for varsig in pred_probsignal:
            if varsig>=cut:
                S+=1
        for varbck in pred_probbkg:
            if varbck>=cut:
                B+=1
        try:
            fom.append(a*S/np.sqrt(a*S+B))
        except ZeroDivisionError:
            errorFoM.append(0)
        try:
            errorFoM.append(np.sqrt(((a*(a*S+2*B)/(2*((a*S+B)**(3./2))))*np.sqrt(S))**2.+((a*S/(2*((B+a*S)**(3./2))))*np.sqrt(B))**2.))
        except ZeroDivisionError:
            errorFoM.append(0)
        

    # We obtain the best cut
    bestcut=cuts[fom.index(max(fom))]

    plt.figure()
    plt.errorbar(cuts,fom,yerr=errorFoM,label="FoM, Best Cut=%f" % bestcut,elinewidth=0.5,errorevery=10)
    plt.xlabel("BDT Cut")
    plt.ylabel("FoM")
    plt.title("FoM")
    plt.legend(loc="lower center")
    plt.savefig("BDT_Kone/plots/BDT/FoM_%s.png" % name)
    plt.close()

    # MORE FOCUSED FOM
    # FIGURE OF MERIT
    print("Plotting Figure of Merit")
    a=0.1
    fom=[]
    errorFoM=[]
    cuts=np.linspace(0.7,0.9999,1000)
    for cut in cuts:
        S=0
        B=0
        for varsig in pred_probsignal:
            if varsig>=cut:
                S+=1
        for varbck in pred_probbkg:
            if varbck>=cut:
                B+=1
        try:
            fom.append(a*S/np.sqrt(a*S+B))
        except ZeroDivisionError:
            errorFoM.append(0)
        try:
            errorFoM.append(np.sqrt(((a*(a*S+2*B)/(2*((a*S+B)**(3./2))))*np.sqrt(S))**2.+((a*S/(2*((B+a*S)**(3./2))))*np.sqrt(B))**2.))
        except ZeroDivisionError:
            errorFoM.append(0)
    
    # We obtain the best cut and save it in a .txt file
    bestcut=cuts[fom.index(max(fom))]

    with open('BDT_Kone/data/%s_Best_cut.txt' % name, 'w') as f:
            f.write(str(bestcut))
    f.close
    
    plt.figure()
    plt.errorbar(cuts,fom,yerr=errorFoM,label="FoM, Best Cut=%f" % bestcut,elinewidth=0.5,errorevery=10)
    plt.xlabel("BDT Cut")
    plt.ylabel("FoM")
    plt.title("FoM")
    plt.legend(loc="lower center")
    plt.savefig("BDT_Kone/plots/BDT/FoM_peak_%s.png" % name)
    plt.close()



mcfile="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root"
mctree="DecayTree"
cutsmc="Bu_BKGCAT==0"
cutsmcplus="(Bu_BKGCAT==0) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005) & (acos((ep_PX*Kp_PX+ep_PY*Kp_PY+ep_PZ*Kp_PZ)/(ep_P*Kp_P))>0.0005) & (acos((Kp_PX*em_PX+Kp_PY*em_PY+Kp_PZ*em_PZ)/(Kp_P*em_P))>0.0005) & (Bu_BPVIPCHI2<9)"
datafile="/eos/lhcb/user/p/pvidrier/roots/B2JpsiKLFU_MagAll_presel.root"
datatree="DecayTree;1"
cutsdata="Jpsi_M > 3200"
variables="BDT_Kone/data/variableselection.yaml"
name="BDT"

BDT_Training(mcfile,mctree,cutsmcplus,datafile,datatree,cutsdata,variables,name)
