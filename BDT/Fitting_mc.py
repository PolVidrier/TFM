import uproot
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import yaml
import ROOT
import time
from uncertainties import ufloat


RooFit         = ROOT.RooFit
RooRealVar     = ROOT.RooRealVar
RooArgList     = ROOT.RooArgList
RooArgSet      = ROOT.RooArgSet
RooDataSet     = ROOT.RooDataSet
RooGaussian    = ROOT.RooGaussian
RooExponential = ROOT.RooExponential
RooAddPdf      = ROOT.RooAddPdf
RooCrystalBall = ROOT.RooCrystalBall
RooAbsReal     = ROOT.RooAbsReal

# definition of functions for this script
def ballFit(bestcut,variables,file,tree, cuts,Has_ep_em, mean_val, xmin, xmax, name):

    # define variables and pdfs
    Jpsi_M = RooRealVar("Jpsi_M","Jpsi_M", xmin, xmax)
    
    meanball  = RooRealVar("meanball","meanball",mean_val,mean_val-300,mean_val+300)    
    sigmaball = RooRealVar("sigmaball", "sigmaball", 80, 10, 100)
    alphaL    = RooRealVar("alphaL", "alphaL", 0.5, 0.01, 1.)
    nL        = RooRealVar("nL", "nL", 10, 5, 25)
    alphaR    = RooRealVar("alphaR", "alphaR", 1, 0.1, 1.5)
    nR        = RooRealVar("nR", "nR", 5, 1, 10)
    ball=RooCrystalBall("ball","ball",Jpsi_M,meanball,sigmaball,alphaL,nL,alphaR,nR)

    
    # define coefficiencts
    nsig = RooRealVar("nsig", "nsig", 1000, 0, 20000)
    
    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(ball)
    #suma.add(exp)
    
    coeff.add(nsig)
    
    model = ROOT.RooAddPdf("model", "model", suma, coeff)
    
    # define dataset
    with open(bestcut,"r") as f:
        bestcut=float(f.read())
    f.close

    data=uproot.open(file)[tree]
    features=[]
    featuresmc=[]

    with open(variables,"r") as f:
        dictionary=yaml.safe_load(f)

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

    # Create the dataset
    ds = RooDataSet("data", "dataset with x", RooArgSet(Jpsi_M))

    for i in range(len(datapred_prob)):
        if datapred_prob[i] >= bestcut:    #Signal
            mas_pred.append(mas_test[i])
            Jpsi_M.setVal(mas_test[i])
            ds.add(RooArgSet(Jpsi_M))

    
    #create and open the canvas
    can = ROOT.TCanvas("hist","hist", 200,10, 1000, 550)
    pad1 = ROOT.TPad( "pad1", "Histogram",0.,0.15,1.0,1.0,0)
    pad2 = ROOT.TPad( "pad2", "Residual plot",0.,0.,1.0,0.15,0)
    can.cd()

    pad1.Draw()
    pad2.Draw()
    ################

    # plot dataset and fit
    massFrame = Jpsi_M.frame()
    
    ds.plotOn(massFrame)
 
    fitResults = model.fitTo(ds)
    model.plotOn(massFrame, RooFit.VisualizeError(fitResults, 1),
                 RooFit.Name("curve_model"))


    #Construct the histogram for the residual plot and plot it on pad2
    hresid = massFrame.residHist()
    chi2 = massFrame.chiSquare()
    
    pad2.cd()
    hresid.SetTitle("")
    hresid.Draw()

    ###################

    model.plotOn(massFrame, RooFit.Components("ball"), RooFit.LineColor(2),
                 RooFit.VisualizeError(fitResults, 1))

    #Draw the fitted histogram into pad1
    pad1.cd()
    t1 = ROOT.TPaveLabel(500.,200.,1500.,270., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
    massFrame.addObject(t1) 
    massFrame.SetTitle("Histogram and fit of %s" % name)
    

    # print results
    print("{} has been fit to {} with a chi2 = {}".format(model.GetName(), file, chi2))
 
    print("Total number of entries is: {}".format(ds.numEntries()))
    print("Number of sig entries is: {:.0f} +- {:.0f}".format(nsig.getValV(),
                                                              nsig.getError()))

    

    sigVal = ufloat(nsig.getValV(), nsig.getError())


    t3 = ROOT.TPaveLabel(500.,270.,1500.,340., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
    t5 = ROOT.TPaveLabel(500.,340.,1500.,410., 'Mean = {:.2f} +- {:.2f}'.format(meanball.getValV(), meanball.getError()))
    t6 = ROOT.TPaveLabel(500.,410.,1500.,480., 'Sigma = {:.2f} +- {:.2f}'.format(sigmaball.getValV(), sigmaball.getError()))
    t8 = ROOT.TPaveLabel(500.,480.,1500.,550., 'alphaL = {:.2f} +- {:.2f}'.format(alphaL.getValV(), alphaL.getError()))
    t9 = ROOT.TPaveLabel(500.,550.,1500.,620., 'nL = {:.2f} +- {:.2f}'.format(nL.getValV(), nL.getError()))
    t10 = ROOT.TPaveLabel(500.,620.,1500.,690., 'alphaR = {:.2f} +- {:.2f}'.format(alphaR.getValV(), alphaR.getError()))
    t11 = ROOT.TPaveLabel(500.,690.,1500.,760., 'nR = {:.2f} +- {:.2f}'.format(nR.getValV(), nR.getError()))
    
    
    massFrame.addObject(t3)
    massFrame.addObject(t5)
    massFrame.addObject(t6)
    massFrame.addObject(t8)
    massFrame.addObject(t9)
    massFrame.addObject(t10)
    massFrame.addObject(t11)

    massFrame.Draw()
    #Save the result
    can.SaveAs("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT/plots/Fit_%s.png" % name)
    #####################

    return 

# FOR THE MC
bestcut="BDT/BDT_Best_cut.txt"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/mc/mc_total.root"
tree="Tuple/DecayTree;1"
Has_ep_em=False
cuts="Bu_BKGCAT==0"
name="mc"
mean=3100.
xmin=0.
xmax=4000.

ballFit(bestcut,variables,file,tree,Has_ep_em,cuts,mean,xmin,xmax,name)

