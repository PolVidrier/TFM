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


def ballFit(file,tree, cuts, mean_val, xmin, xmax, name):
    # file= name of the .root file with the bdt applied
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # mean_val= initial value of the mean
    # xmin= minimum of the range
    # xmax= maximum of the range
    # name= name for the fit plot

    # define variables and pdfs
    Bu_DTFPV_JpsiConstr_MASS = RooRealVar("Bu_DTFPV_JpsiConstr_MASS","Bu_DTFPV_JpsiConstr_MASS", xmin, xmax)
    
    meanball  = RooRealVar("meanball","meanball",mean_val,mean_val-300,mean_val+300)    
    sigmaball = RooRealVar("sigmaball", "sigmaball", 10, 5, 30)
    alphaL    = RooRealVar("alphaL", "alphaL", 1, 0.5, 3)
    nL        = RooRealVar("nL", "nL", 7, 1, 10)
    alphaR    = RooRealVar("alphaR", "alphaR", 1, 0.1, 2)
    nR        = RooRealVar("nR", "nR", 7, 1, 10)
    ball=RooCrystalBall("ball","ball",Bu_DTFPV_JpsiConstr_MASS,meanball,sigmaball,alphaL,nL,alphaR,nR)

    
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
    file = ROOT.TFile(file)
    treee = file.Get(tree)
    if (cuts!=""): treee = treee.CopyTree(cuts)
    ds = RooDataSet("data", "dataset with x", treee, RooArgSet(Bu_DTFPV_JpsiConstr_MASS))

    #create and open the canvas
    can = ROOT.TCanvas("hist","hist", 200,10, 1000, 550)
    pad1 = ROOT.TPad( "pad1", "Histogram",0.,0.15,1.0,1.0,0)
    pad2 = ROOT.TPad( "pad2", "Residual plot",0.,0.,1.0,0.15,0)
    can.cd()

    pad1.Draw()
    pad2.Draw()
    ################

    # plot dataset and fit
    massFrame = Bu_DTFPV_JpsiConstr_MASS.frame()
    
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
    t1 = ROOT.TPaveLabel(5600.,200.,6000.,400., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
    massFrame.addObject(t1) 
    massFrame.SetTitle("Histogram and fit of %s" % name)
    

    # print results
    print("{} has been fit to {} with a chi2 = {}".format(model.GetName(), file, chi2))
 
    print("Total number of entries is: {}".format(ds.numEntries()))
    print("Number of sig entries is: {:.0f} +- {:.0f}".format(nsig.getValV(),
                                                              nsig.getError()))

    sigVal = ufloat(nsig.getValV(), nsig.getError())


    t3 = ROOT.TPaveLabel(5600.,400.,6000.,600., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
    t5 = ROOT.TPaveLabel(5600.,600.,6000.,800., 'Mean = {:.2f} +- {:.2f}'.format(meanball.getValV(), meanball.getError()))
    t6 = ROOT.TPaveLabel(5600.,800.,6000.,1000., 'Sigma = {:.2f} +- {:.2f}'.format(sigmaball.getValV(), sigmaball.getError()))
    t8 = ROOT.TPaveLabel(5600.,1000.,6000.,1200., 'alphaL = {:.2f} +- {:.2f}'.format(alphaL.getValV(), alphaL.getError()))
    t9 = ROOT.TPaveLabel(5600.,1200.,6000.,1400., 'nL = {:.2f} +- {:.2f}'.format(nL.getValV(), nL.getError()))
    t10 = ROOT.TPaveLabel(5600.,1400.,6000.,1600., 'alphaR = {:.2f} +- {:.2f}'.format(alphaR.getValV(), alphaR.getError()))
    t11 = ROOT.TPaveLabel(5600.,1600.,6000.,1800., 'nR = {:.2f} +- {:.2f}'.format(nR.getValV(), nR.getError()))
    
    
    massFrame.addObject(t3)
    massFrame.addObject(t5)
    massFrame.addObject(t6)
    massFrame.addObject(t8)
    massFrame.addObject(t9)
    massFrame.addObject(t10)
    massFrame.addObject(t11)

    massFrame.Draw()
    #Save the result
    can.SaveAs("/afs/cern.ch/work/p/pvidrier/private/GITHUB/BDT_Kone/plots/Fit_%s.png" % name)
    #####################
    print("Fit_%s done" % name)

# FOR THE MC
file="/eos/lhcb/user/p/pvidrier/roots/mc_Kone_with_cuts_Bu_JpsiConstr.root"
tree="DecayTree"
cuts="" # cuts already there
name="Bu_JpsiConstr_mc"
mean=5500.
xmin=4500.
xmax=6500.

ballFit(file,tree,cuts,mean,xmin,xmax,name)

