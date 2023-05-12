import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from root_numpy import tree2array, array2root
import ROOT



def Bias(variable,file,tree,cuts,outputsfile,outputstree,cutsoutputs,xmin,xmax,bins,name):
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

    #outputs=uproot.open(outputsfile)[outputstree]
    #featuresoutputs=outputs.keys()

    #if cutsoutputs=="":
    #    outputspd=outputs.arrays(featuresoutputs,library="pd")
    #else:
    #    outputspd=outputs.arrays(featuresoutputs,cutsoutputs,library="pd")

    #outputs=outputspd.to_numpy()
    outputname="BDT_output"

    #outputss = outputspd[[outputname]]   #for the prediction of mass
    #outputs_array = outputss.to_numpy()

    rfile = ROOT.TFile(outputsfile)
    intree = rfile.Get(outputstree)
    if (cutsoutputs!=""): intree = intree.CopyTree(cutsoutputs)

    # and convert the TTree into an array
    outputs_array = tree2array(intree)

    # MAKING THE MEANS
    masses=np.linspace(xmin,xmax,bins)

    meanoutputs=[]
    errors=[]
    for i in range(bins):
        meanoutput=0
        error=0
        n=0
        forerror=[]
        for var in mas_pred_array:
            if i==(bins-1):
                if masses[i]<var:
                    j=list(mas_pred_array).index(var)
                    n+=1
                    meanoutput+=float(outputs_array[j][0])
                    forerror.append(outputs_array[j][0])
            else:
                if masses[i]<var<masses[i+1]:
                    j=list(mas_pred_array).index(var)
                    n+=1
                    meanoutput+=float(outputs_array[j][0])
                    forerror.append(outputs_array[j][0])
        if n!=0:
            meanoutput=meanoutput/n
            for val in forerror:
                error+=(val-meanoutput)**2.
            error=np.sqrt(error/n)
        errors.append(error)
        meanoutputs.append(meanoutput)  

    # Plotting the Jpsi_M Bias
    print("Plotting %s_Bias_%s" % (variable,name))
    plt.errorbar(masses,meanoutputs,yerr=errors,fmt="o",markersize=4,label="Nº of Entries = %i" % len(mas_pred_array))
    plt.title("%s %s Bias" % (variable,name))
    plt.ylim(0,1.1)
    plt.xlabel("%s" % variable)
    plt.ylabel("%s" % outputname)
    #plt.legend(loc="center right")
    plt.savefig("BDT/plots/BDT/%s_Bias_%s.png" % (variable,name))
    plt.close()



# FOR THE SIGNAL
variable="Jpsi_M"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total_epem.root"
tree="DecayTree"
cuts="(Bu_BKGCAT==0) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005)"
outputsfile="/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT/data/BDT_output_mc.root"
outputstree="DecayTree"
cutsoutputs=""
name="Signal"
xmin=2500
xmax=3500
bins=100

Bias(variable,file,tree,cuts,outputsfile,outputstree,cutsoutputs,xmin,xmax,bins,name)


# FOR THE BACKGROUND 
variable="Jpsi_M"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
cuts="(Jpsi_M>3200) & (acos((ep_PX*em_PX+ep_PY*em_PY+ep_PZ*em_PZ)/(ep_P*em_P))>0.0005) & ((Hlt1_Hlt1DisplacedDielectronDecision==1) | (Hlt1_Hlt1DisplacedLeptonsDecision==1) | (Hlt1_Hlt1LowMassNoipDielectronDecision==1) | (Hlt1_Hlt1SingleHighEtDecision==1) | (Hlt1_Hlt1SingleHighPtElectronDecision==1) | (Hlt1_Hlt1TrackElectronMVADecision==1))"
outputsfile="/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT/data/BDT_output_data_bkg.root"
outputstree="DecayTree"
cutsoutputs=""
name="Background"
xmin=3200
xmax=4000
bins=100

Bias(variable,file,tree,cuts,outputsfile,outputstree,cutsoutputs,xmin,xmax,bins,name)