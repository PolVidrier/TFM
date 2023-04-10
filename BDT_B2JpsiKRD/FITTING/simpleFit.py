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

# definition of functions for this script
def simpleFit(tree, cuts, mean_val, xmin = 4000, xmax = 7000):
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
    
    mean  = RooRealVar("mean", "mean",  mean_val, mean_val-200, mean_val+200)
    sigma = RooRealVar("sigma", "sigma", 80, 10, 150)
    gauss = RooGaussian("gauss", "gauss", Jpsi_M, mean, sigma)
    
    tau = RooRealVar("tau", "tau", -0.005, -0.01, 0.)
    exp = RooExponential("exp", "exp", Jpsi_M, tau)
    
    # define coefficiencts
    nsig = RooRealVar("nsig", "nsig", 1000, 0, 20000)
    nbkg = RooRealVar("nbkg", "nbkg", 1000, 0, 20000)
    
    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(gauss)
    suma.add(exp)
    
    coeff.add(nsig)
    coeff.add(nbkg)
    
    model = ROOT.RooAddPdf("model", "model", suma, coeff)
    
    # define dataset
    if (cuts!=""): tree = tree.CopyTree(cuts)
    ds = RooDataSet("data", "dataset with x", tree, RooArgSet(Jpsi_M))
    
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

    model.plotOn(massFrame, RooFit.Components("gauss"), RooFit.LineColor(2),
                 RooFit.VisualizeError(fitResults, 1))
    model.plotOn(massFrame, RooFit.Components("exp")  , RooFit.LineColor(3),
                 RooFit.VisualizeError(fitResults, 1))
    #model.paramOn(massFrame, Layout=(.55,.95,.93), Parameters=RooArgSet(nsig, nbkg, mean, sigma, tau))

    #Draw the fitted histogram into pad1
    pad1.cd()
    t1 = ROOT.TPaveLabel(6000.,120.,6500.,140., '#chi^{2}' + ' / ndf = {:.3f}'.format(chi2))
    massFrame.addObject(t1) 
    massFrame.SetTitle("Histogram and fit")
    

    # print results
    print("{} has been fit to {} with a chi2 = {}".format(model.GetName(), tree.GetName(), chi2))
 
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

    
    t2 = ROOT.TPaveLabel(6000.,140.,6500.,160., 'S/(S+B)^{1/2}' + '= {:.3f}'.format(signif.nominal_value))
    t3 = ROOT.TPaveLabel(6000.,160.,6500.,180., 'NSig = {:.0f} +- {:.0f}'.format(nsig.getValV(), nsig.getError()))
    t4 = ROOT.TPaveLabel(6000.,180.,6500.,200., 'NBkg = {:.0f} +- {:.0f}'.format(nbkg.getValV(), nbkg.getError()))
    t5 = ROOT.TPaveLabel(6000.,200.,6500.,220., 'Mean = {:.2f} +- {:.2f}'.format(mean.getValV(), mean.getError()))
    t6 = ROOT.TPaveLabel(6000.,220.,6500.,240., 'Sigma = {:.2f} +- {:.2f}'.format(sigma.getValV(), sigma.getError()))
    t7 = ROOT.TPaveLabel(6000.,240.,6500.,260., 'tau = {:.5f} +- {:.5f}'.format(tau.getValV(), tau.getError()))
    
    massFrame.addObject(t2)
    massFrame.addObject(t3)
    massFrame.addObject(t4)
    massFrame.addObject(t5)
    massFrame.addObject(t6)
    massFrame.addObject(t7)

    massFrame.Draw()
    #Save the result
    can.SaveAs("/afs/cern.ch/work/p/pvidrier/private/GITHUB/FITTING/Fitted_histogram.png")
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
    simpleFit(tree, args.cuts, args.mean,args.xmin, args.xmax)


#EOF

