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
def ballFit(bestcut,variables,file,tree, cuts,Has_ep_em, mean_val, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val):

    # define variables and pdfs
    Jpsi_M = RooRealVar("Jpsi_M","Jpsi_M", xmin, xmax)
    
    meanball  = RooRealVar("meanball","meanball",mean_val,mean_val-300,mean_val+300)
    sigmaball = RooRealVar("sigmaball", "sigmaball", 80, 10, 100)
    alphaL    = RooRealVar("alphaL", "alphaL", alphaL_val, alphaL_val-0.01, alphaL_val+0.01)
    nL        = RooRealVar("nL", "nL", nL_val, nL_val-0.01, nL_val+0.01)
    alphaR    = RooRealVar("alphaR", "alphaR", alphaR_val, alphaR_val-0.01, alphaR_val+0.01)
    nR        = RooRealVar("nR", "nR", nR_val, nR_val-0.01, nR_val+0.01)
    ball=RooCrystalBall("ball","ball",Jpsi_M,meanball,sigmaball,alphaL,nL,alphaR,nR)

    tau = RooRealVar("tau", "tau", -0.25512, -1, 0.)
    exp = RooExponential("exp", "exp", Jpsi_M, tau)
    
    alphaL.setConstant(True)
    alphaR.setConstant(True)
    nL.setConstant(True)
    nR.setConstant(True)

    # define coefficiencts
    nsig = RooRealVar("nsig", "nsig", 1000, 0, 20000)
    nbkg = RooRealVar("nbkg", "nbkg", 1000, 0, 20000)
    
    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(ball)
    suma.add(exp)
    
    coeff.add(nsig)
    coeff.add(nbkg)
    
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
            if mas_test[i]<xmin or mas_test[i]>xmax:
                continue 
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
    model.plotOn(massFrame, RooFit.Components("exp")  , RooFit.LineColor(3),
                 RooFit.VisualizeError(fitResults, 1))

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
    t1 = ROOT.TPaveLabel(500.,35.,1500.,40., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
    
    massFrame.SetTitle("Histogram and fit of %s" % name)
    

    # print results
    print("{} has been fit to {} with a chi2 = {}".format(model.GetName(), file, chi2))
 
    print("Total number of entries is: {}".format(ds.numEntries()))
    print("Number of sig entries is: {:.0f} +- {:.0f}".format(nsig.getValV(),
                                                              nsig.getError()))
    print("Number of bkg entries is: {:.0f} +- {:.0f}".format(nbkg.getValV(),
                                                              nbkg.getError()))
    
    # compute S/(S+B)**0.5, with error propagation from uncertainties module
    sigVal = ufloat(nsig.getValV(), nsig.getError())
    bkgVal = ufloat(nbkg.getValV(), nbkg.getError())
    signif = sigVal/(sigVal+bkgVal)**0.5

    print("S/sqrt(S+B) = {:.2f} +- {:.2f}".format(signif.nominal_value, signif.std_dev))

    t2 = ROOT.TPaveLabel(500.,40.,1500.,45., 'S/(S+B)^{1/2}' + '= {:.3f}'.format(signif.nominal_value))
    t3 = ROOT.TPaveLabel(500.,45.,1500.,50., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
    t4 = ROOT.TPaveLabel(500.,50.,1500.,55., 'NBkg = {:.0f} +- {:.0f}'.format(nbkg.getValV(), nbkg.getError()))
    t5 = ROOT.TPaveLabel(500.,55.,1500.,60., 'Mean = {:.2f} +- {:.2f}'.format(meanball.getValV(), meanball.getError()))
    t6 = ROOT.TPaveLabel(500.,60.,1500.,65., 'Sigma = {:.2f} +- {:.2f}'.format(sigmaball.getValV(), sigmaball.getError()))
    t8 = ROOT.TPaveLabel(500.,65.,1500.,70., 'alphaL = {:.2f} +- {:.2f}'.format(alphaL.getValV(), alphaL.getError()))
    t9 = ROOT.TPaveLabel(500.,70.,1500.,75., 'nL = {:.2f} +- {:.2f}'.format(nL.getValV(), nL.getError()))
    t10 = ROOT.TPaveLabel(500.,75.,1500.,80., 'alphaR = {:.2f} +- {:.2f}'.format(alphaR.getValV(), alphaR.getError()))
    t11 = ROOT.TPaveLabel(500.,80.,1500.,85., 'nR = {:.2f} +- {:.2f}'.format(nR.getValV(), nR.getError()))
    t7 = ROOT.TPaveLabel(500.,85.,1500.,90., 'Tau = {:.5f} +- {:.5f}'.format(tau.getValV(), tau.getError()))
    
    if xmin==2000 and xmax==3500:
        t1 = ROOT.TPaveLabel(2100.,15.,2400.,18., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
        t2 = ROOT.TPaveLabel(2100.,18.,2400.,21., 'S/(S+B)^{1/2}' + '= {:.3f}'.format(signif.nominal_value))
        t3 = ROOT.TPaveLabel(2100.,21.,2400.,24., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
        t4 = ROOT.TPaveLabel(2100.,24.,2400.,27., 'NBkg = {:.0f} +- {:.0f}'.format(nbkg.getValV(), nbkg.getError()))
        t5 = ROOT.TPaveLabel(2100.,27.,2400.,30., 'Mean = {:.2f} +- {:.2f}'.format(meanball.getValV(), meanball.getError()))
        t6 = ROOT.TPaveLabel(2100.,30.,2400.,33., 'Sigma = {:.2f} +- {:.2f}'.format(sigmaball.getValV(), sigmaball.getError()))
        t8 = ROOT.TPaveLabel(2100.,33.,2400.,36., 'alphaL = {:.2f} +- {:.2f}'.format(alphaL.getValV(), alphaL.getError()))
        t9 = ROOT.TPaveLabel(2100.,36.,2400.,39., 'nL = {:.2f} +- {:.2f}'.format(nL.getValV(), nL.getError()))
        t10 = ROOT.TPaveLabel(2100.,39.,2400.,42., 'alphaR = {:.2f} +- {:.2f}'.format(alphaR.getValV(), alphaR.getError()))
        t11 = ROOT.TPaveLabel(2100.,42.,2400.,45., 'nR = {:.2f} +- {:.2f}'.format(nR.getValV(), nR.getError()))
        t7 = ROOT.TPaveLabel(2100.,45.,2400.,48., 'Tau = {:.5f} +- {:.5f}'.format(tau.getValV(), tau.getError()))


    massFrame.addObject(t1) 
    massFrame.addObject(t2)
    massFrame.addObject(t3)
    massFrame.addObject(t4)
    massFrame.addObject(t5)
    massFrame.addObject(t6)
    massFrame.addObject(t7)
    massFrame.addObject(t8)
    massFrame.addObject(t9)
    massFrame.addObject(t10)
    massFrame.addObject(t11)

    massFrame.Draw()
    #Save the result
    can.SaveAs("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT/plots/Fit_%s_%ito%i.png" % (name,xmin,xmax))
    #####################

    return 

# FOR THE DATA
bestcut="BDT/BDT_Best_cut.txt"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
Has_ep_em=True
cuts="Jpsi_M>100"
name="data"
mean=3100.
xmin=0.
xmax=4000.
nL_val=9.51
alphaL_val=0.15
nR_val=2.49
alphaR_val=0.89

ballFit(bestcut,variables,file,tree, cuts,Has_ep_em, mean, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val)

# FOR THE DATA WITH RANGE FROM 2000 TO 3500
bestcut="BDT/BDT_Best_cut.txt"
variables="BDT/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
Has_ep_em=True
cuts="Jpsi_M>100"
name="data"
mean=3100.
xmin=2000.
xmax=3500.
nL_val=9.51
alphaL_val=0.15
nR_val=2.49
alphaR_val=0.89

ballFit(bestcut,variables,file,tree, cuts,Has_ep_em, mean, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val)

