# SAME AS FIT BUT ADAPTED TO READ FROM A .TXT FILE



#!/usr/bin/env python
# =============================================================================
# @file   simpleFit.py
# @author C. Marin Benito (carla.marin.benito@cern.ch)
# @date   22.12.2014
# =============================================================================
"""This script fits a simple pdf to a given dataset using RooFit."""

# imports
import os
import ROOT
#import lib.RooFitDecorators as RooFitDecorators
## the "lib." above means that the corresponding module RooFitDecorators 
## is found at a subdirectory lib/ (from where you are executing python).
## The lib/ subdirectory must have a blank file named __init__.py
## See: http://docs.python.org/tutorial/modules.html
## the RooFitDecorators module is disabled since it makes use of a package
## called pyLHCb which is not part of the standard python distribution

#from lib.RooFitUtils import ResidualPlot

from uncertainties import ufloat
##
## Indeed uncertainties is separate Python package that you should install:
## https://pypi.python.org/pypi/uncertainties
## It is available at https://anaconda.org/conda-forge/uncertainties , so you should be able to ## get it like so:
## conda install -c conda-forge uncertainties
##
## If that doesn't work, you can always try:
## pip install uncertainties






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
def ballFitdata(tree, cuts, mean_val, xmin = 4000, xmax = 7000):   # NOW TREE IS A .TXT FILE
    """
    This function fits the "Jpsi_M" variable of a given TTree
    with a model formed by a Gaussian and an exponential pdf.
    All shape parameters are allowed to float in the fit. The 
    initial and range values are hardcoded in the code, except
    for the initial value of the Gaussian mean and the range
    of the Jpsi_M variable to be used.
    Returns the composed model (RooAbsPdf), the residual plot object
    and its chi2 value (float)
    Definition of the arguments:
    :tree: type TTree
    the root TTree that contains the variable to be fitted
    :cuts: type str
    optional cuts to apply to the TTree before fitting
    :mean_val: type float
    initial value for the Gaussian mean that will be floated
    during the fit
    :xmin: type float, optional
    minimum value of the Jpsi_M range to be fitted. Default: 4000
    :xmax: type float, optional
    maximum value of the Jpsi_M range to be fitted. Default: 7000
    """
    
    # define variables and pdfs
    Jpsi_M = RooRealVar("Jpsi_M","Jpsi_M", xmin, xmax)
    
    #mean  = RooRealVar("mean", "mean",  mean_val, mean_val-200, mean_val+200)
    #sigma = RooRealVar("sigma", "sigma", 80, 10, 150)
    #gauss = RooGaussian("gauss", "gauss", Jpsi_M, mean, sigma)
    
    

    mean_ball=mean_val

    meanball  = RooRealVar("meanball","meanball",mean_ball,mean_ball-300,mean_ball+300)
    sigmaball = RooRealVar("sigmaball", "sigmaball", 80, 10, 100)
    alphaL    = RooRealVar("alphaL", "alphaL", 0.5, 0.1, 10.)
    nL        = RooRealVar("nL", "nL", 20, 1, 40)
    alphaR    = RooRealVar("alphaR", "alphaR", 3, 0.1, 10.)
    nR        = RooRealVar("nR", "nR", 10, 0.1, 25)
    ball=RooCrystalBall("ball","ball",Jpsi_M,meanball,sigmaball,alphaL,nL,alphaR,nR)
    
    
    tau = RooRealVar("tau", "tau", -0.1, -1, 0.)
    exp = RooExponential("exp", "exp", Jpsi_M, tau)

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
    if (cuts!=""): tree = tree.CopyTree(cuts)
    
    input_file = open(tree, "r")
    
    # Create the dataset
    ds = RooDataSet("data", "dataset with x", RooArgSet(Jpsi_M))

    # Loop over the input file and add data to the dataset
    for line in input_file:
        Jpsi_M_val = float(line.strip())
        Jpsi_M.setVal(Jpsi_M_val)
        ds.add(RooArgSet(Jpsi_M))
    input_file.close()
    
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
    model.plotOn(massFrame, RooFit.Components("exp")  , RooFit.LineColor(3),
                 RooFit.VisualizeError(fitResults, 1))
    #model.paramOn(massFrame, Layout=(.55,.95,.93), Parameters=RooArgSet(nsig, nbkg, mean, sigma, tau))

    #Draw the fitted histogram into pad1
    pad1.cd()
    t1 = ROOT.TPaveLabel(500.,35.,1500.,40., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
    massFrame.addObject(t1) 
    massFrame.SetTitle("Histogram and fit")
    

    # print results
    print("{} has been fit to with a chi2 = {}".format(model.GetName(), chi2))
 
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
    #t12 = ROOT.TPaveLabel(500.,90.,1500.,95., 'Sigma SIGNAL = {:.5f} +- {:.5f}'.format(sigma.getValV(), sigma.getError()))
    
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
    #massFrame.addObject(t12)

    massFrame.Draw()
    #Save the result
    can.SaveAs("/afs/cern.ch/work/p/pvidrier/private/GITHUB/FITTING/myFITdata.png")
    #####################

    return 


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", action="store", type=str)
    parser.add_argument("-t", "--tree", default="DecayTree",
                        action="store", type=str)
    parser.add_argument("-m", "--mean", default=5280.,
                        action="store", type=float)
    parser.add_argument("-n", "--xmin", default=4000.,
                        action="store", type=float)
    parser.add_argument("-x", "--xmax", default=7000.,
                        action="store", type=float)
    parser.add_argument("-c", "--cuts", default="",
                        action="store", type=str)
    args = parser.parse_args()

    # sanity check
    if not os.path.exists(args.file):
        print("File doesn't exist! Exiting...")
        exit()

    # read data
    file = ROOT.TFile(args.file)
    tree = file.Get(args.tree)

    # fit it
    ballFitdata(tree, args.cuts, args.mean,args.xmin, args.xmax)


#EOF

