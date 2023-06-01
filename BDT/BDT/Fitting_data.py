import pandas as pd
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


def ballFit(file,tree, cuts, mean_val, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val):
    # file= name of the .root file with the bdt applied
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
    
    # define dataset
    file = ROOT.TFile(file)
    treee = file.Get(tree)
    if (cuts!=""): treee = treee.CopyTree(cuts)
    ds = RooDataSet("data", "dataset with x", treee, RooArgSet(Jpsi_M))

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
    t1 = ROOT.TPaveLabel(250.,40.,1250.,50., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
    
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

    t2 = ROOT.TPaveLabel(250.,50.,1250.,60., 'S/(S+B)^{1/2}' + '= {:.3f}'.format(signif.nominal_value))
    t3 = ROOT.TPaveLabel(250.,60.,1250.,70., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
    t4 = ROOT.TPaveLabel(250.,70.,1250.,80., 'NBkg = {:.0f} +- {:.0f}'.format(nbkg.getValV(), nbkg.getError()))
    t5 = ROOT.TPaveLabel(250.,80.,1250.,90., 'Mean = {:.2f} +- {:.2f}'.format(meanball.getValV(), meanball.getError()))
    t6 = ROOT.TPaveLabel(250.,90.,1250.,100., 'Sigma = {:.2f} +- {:.2f}'.format(sigmaball.getValV(), sigmaball.getError()))
    t8 = ROOT.TPaveLabel(250.,100.,1250.,110., 'alphaL = {:.2f} +- {:.2f}'.format(alphaL.getValV(), alphaL.getError()))
    t9 = ROOT.TPaveLabel(250.,110.,1250.,120., 'nL = {:.2f} +- {:.2f}'.format(nL.getValV(), nL.getError()))
    t10 = ROOT.TPaveLabel(250.,120.,1250.,130., 'alphaR = {:.2f} +- {:.2f}'.format(alphaR.getValV(), alphaR.getError()))
    t11 = ROOT.TPaveLabel(250.,130.,1250.,140., 'nR = {:.2f} +- {:.2f}'.format(nR.getValV(), nR.getError()))
    t7 = ROOT.TPaveLabel(250.,140.,1250.,150., 'Tau = {:.5f} +- {:.5f}'.format(tau.getValV(), tau.getError()))
    
    if xmin!=0: # so we can see the labels
        t1 = ROOT.TPaveLabel(2100.,20.,2400.,25., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
        t2 = ROOT.TPaveLabel(2100.,25.,2400.,30., 'S/(S+B)^{1/2}' + '= {:.3f}'.format(signif.nominal_value))
        t3 = ROOT.TPaveLabel(2100.,30.,2400.,35., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
        t4 = ROOT.TPaveLabel(2100.,35.,2400.,40., 'NBkg = {:.0f} +- {:.0f}'.format(nbkg.getValV(), nbkg.getError()))
        t5 = ROOT.TPaveLabel(2100.,40.,2400.,45., 'Mean = {:.2f} +- {:.2f}'.format(meanball.getValV(), meanball.getError()))
        t6 = ROOT.TPaveLabel(2100.,45.,2400.,50., 'Sigma = {:.2f} +- {:.2f}'.format(sigmaball.getValV(), sigmaball.getError()))
        t8 = ROOT.TPaveLabel(2100.,50.,2400.,55., 'alphaL = {:.2f} +- {:.2f}'.format(alphaL.getValV(), alphaL.getError()))
        t9 = ROOT.TPaveLabel(2100.,55.,2400.,60., 'nL = {:.2f} +- {:.2f}'.format(nL.getValV(), nL.getError()))
        t10 = ROOT.TPaveLabel(2100.,60.,2400.,65., 'alphaR = {:.2f} +- {:.2f}'.format(alphaR.getValV(), alphaR.getError()))
        t11 = ROOT.TPaveLabel(2100.,65.,2400.,70., 'nR = {:.2f} +- {:.2f}'.format(nR.getValV(), nR.getError()))
        t7 = ROOT.TPaveLabel(2100.,70.,2400.,75., 'Tau = {:.5f} +- {:.5f}'.format(tau.getValV(), tau.getError()))


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
file="/eos/lhcb/user/p/pvidrier/roots/data_with_cuts.root"
tree="DecayTree"
cuts="" # cuts already there
name="data"
mean=3100.
xmin=0.
xmax=4000.
nR_val=3.12
alphaR_val=0.77
nL_val=8.45
alphaL_val=0.19

ballFit(file,tree, cuts, mean, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val)

# FOR THE DATA WITH RANGE FROM 2000 TO 3500
file="/eos/lhcb/user/p/pvidrier/roots/data_with_cuts.root"
tree="DecayTree"
cuts="" # cuts already there
name="data"
mean=3100.
xmin=2000.
xmax=3500.
nR_val=3.12
alphaR_val=0.77
nL_val=8.45
alphaL_val=0.19

ballFit(file,tree, cuts, mean, xmin, xmax, name,nL_val,alphaL_val,nR_val,alphaR_val)

