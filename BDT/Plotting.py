import uproot
import pandas as pd
import matplotlib.pyplot as plt


def Plotting(variable,file,tree,cuts,name):
    # variable= name of the variable to plot
    # file= name of the .root file with the bdt applied
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # name= name for the Jpsi_M plot

    data=uproot.open(file)[tree]
    features=data.keys()

    if cuts=="":
        fulldatapd=data.arrays(features,library="pd")
    else:
        fulldatapd=data.arrays(features,cuts,library="pd")

    data=fulldatapd.to_numpy()

    mas_pred = fulldatapd[[variable]]   #for the prediction of mass
    mas_pred_array = mas_pred.to_numpy()

    # Plotting the Jpsi_M
    print("Plotting %s_%s" % (variable,name))
    plt.hist(mas_pred_array,label="Prediction", bins=100)
    plt.title("%s %s Prediction" % (variable,name))
    plt.xlabel("%s" % variable)
    plt.legend(loc="upper left")
    plt.savefig("BDT/plots/%s_%s.png" % (variable,name))
    plt.close()



# FOR THE DATA
variable="Jpsi_M"
file="/eos/lhcb/user/p/pvidrier/roots/data_with_cuts.root"
tree="DecayTree;1"
cuts="Jpsi_M>100"
name="data"

Plotting(variable,file,tree,cuts,name)

# FOR THE MC
variable="Jpsi_M"
file="/eos/lhcb/user/p/pvidrier/roots/mc_with_cuts.root"
tree="DecayTree"
cuts="Bu_BKGCAT==0"
name="mc"

Plotting(variable,file,tree,cuts,name)

