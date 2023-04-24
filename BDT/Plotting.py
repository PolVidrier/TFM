import uproot
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def Plotting(bestcut,variables,file,tree,cuts,name):
    # bestcut= .txt file with the best cut of the BDT
    # variables= variables, in .yaml file
    # file= name of the .root file
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # name= name for the Jpsi_M plot

    # We read the best cut
    with open(bestcut,"r") as f:
        bestcut=float(f.read())
    f.close

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

    data=fulldatapd.to_numpy()

    # We read the BDT output
    bdt_out=uproot.open("BDT/data/BDT_output_%s.root" % name)["DecayTree"]
    bdt_output=bdt_out.arrays(["BDT_output"],library="pd")
    datapred_prob=bdt_output.to_numpy()

    mas_test = fulldatapd[["Jpsi_M"]]   #for the prediction of mass
    mas_test_array = mas_test.to_numpy()
    mas_pred, mas_test = [], mas_test_array[:,0].tolist()

    for i in range(len(datapred_prob)):
        if datapred_prob[i] >= bestcut:    #Signal
            mas_pred.append(mas_test[i])

    # Plotting the Jpsi_M
    print("Plotting Jpsi_M_%s" % name)
    plt.hist(mas_pred,label="Prediction", bins=100)
    plt.title("Jpsi_M %s Prediction" % name)
    plt.xlabel("Jpsi_M")
    plt.legend(loc="upper left")
    plt.savefig("BDT/plots/Jpsi_M_%s.png" % name)
    plt.close()



# FOR THE DATA
bestcut="BDT/data/BDT_Best_cut.txt"
variables="BDT/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
cuts="Jpsi_M>100"
name="data"

Plotting(bestcut,variables,file,tree,cuts,name)

# FOR THE MC
bestcut="BDT/data/BDT_Best_cut.txt"
variables="BDT/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root"
tree="DecayTree"
cuts="Bu_BKGCAT==0"
name="mc"

Plotting(bestcut,variables,file,tree,cuts,name)

