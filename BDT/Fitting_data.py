import uproot
import pandas as pd
import yaml
import ROOT
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
def ballFit(bestcut,variables,file,tree, cuts, mean_val, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val):
    # bestcut= .txt file with the best cut of the BDT
    # variables= variables, in .yaml file
    # file= name of the .root file
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # mean_val= initial value of the mean
    # xmin= minimum of the range
    # xmax= maximum of the range
    # name= name for the fit plot
    # nL_val= fixed value for nL
    # alphaL_val= fixed values for alphaL
    # nR_val= fixed value for nR
    # alphaR_val= fixed value for alphaR

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
    
    # We read the best cut
    with open(bestcut,"r") as f:
        bestcut=float(f.read())
    f.close

    data=uproot.open(file)[tree]
    features=[]

    with open(variables,"r") as f:
        dictionary=yaml.safe_load(f)

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
    
    if xmin!=0: # so we can see the labels
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
    print("Fit_%s_%ito%i done" % (name,xmin,xmax))

# FOR THE DATA
bestcut="BDT/data/BDT_Best_cut.txt"
variables="BDT/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
cuts="Jpsi_M>100"
name="data"
mean=3100.
xmin=0.
xmax=4000.
nL_val=8.42
alphaL_val=0.15
nR_val=2.90
alphaR_val=0.72

ballFit(bestcut,variables,file,tree, cuts, mean, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val)

# FOR THE DATA WITH RANGE FROM 2000 TO 3500
bestcut="BDT/data/BDT_Best_cut.txt"
variables="BDT/data/variableselection.yaml"
file="/afs/cern.ch/work/p/pvidrier/private/roots/data/Jpsi2ee_MagAll.root"
tree="tuple_B2JpsiKRD/DecayTree;1"
cuts="Jpsi_M>100"
name="data"
mean=3100.
xmin=2000.
xmax=3500.
nL_val=8.42
alphaL_val=0.15
nR_val=2.90
alphaR_val=0.72

ballFit(bestcut,variables,file,tree, cuts, mean, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val)

