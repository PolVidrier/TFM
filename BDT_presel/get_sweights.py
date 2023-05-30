# =============================================================================
# @file   get_sweights.py
# @author C. Marin Benito, A. López Huertas
# @date   13.04.2017
# Last update nov-2022
# =============================================================================
"""Plot control variable using sweights from fit to a discriminant variable"""

import os
import argparse
import ROOT

RooRealVar     = ROOT.RooRealVar
RooArgSet      = ROOT.RooArgSet
RooArgList     = ROOT.RooArgList
RooFit         = ROOT.RooFit
RooCrystalBall = ROOT.RooCrystalBall
RooDataSet     = ROOT.RooDataSet

ROOT.gROOT.SetBatch(True)

def import_data(ws, tree, fit_var_name, fit_var_range, ds_name="ds",
                units_fit="MeV"):
    """Create RooDataset with fit and control variable and import it to workspace.

    Args:
        ws (RooWorkspace): workspace to store data, variables and pdfs
        tree (TTree): root tree with the original data
        fit_var_name (str): name of the discriminant variable to be fit. A branch with
            this name should be present in tree
        fir_var_range (list): list of min and max value of the fit variable 
        units_fit (str, optional): units of the fit variable
    Returns:
        True
    """
    min_fit, max_fit = fit_var_range
    fit_var = RooRealVar(fit_var_name, "mass", min_fit, max_fit,
                         unit=units_fit)#, PlotLabel="mass")
    print(tree.ClassName())
    ds = ROOT.RooDataSet(ds_name, ds_name, tree, RooArgSet(fit_var))
    ws.Import(ds)
    return ds

def fit_MC(tree, var, mean, xmin = 5000, xmax = 7000):
    """
    
    """
    
    # define variables and pdfs
    B_M = RooRealVar(var, var, xmin, xmax)

    meanMC = RooRealVar('meanMC','meanMC',mean,mean-200,mean+200)
    widthMC = RooRealVar('widthMC','widthMC',7.5,4.5,15)
    alphaL= RooRealVar("alphaL", 'alphaL', 1, 0.0000001, 45)
    nL = RooRealVar("nL", 'nL', 2, 0.00000001, 200)
    alphaH = RooRealVar("alphaH", 'alphaH', 1, 0.0000001, 45)
    nH = RooRealVar("nH", 'nH', 2, 0.00000001, 200)
    MCPdf = RooCrystalBall("MCPdf", '', B_M, meanMC, widthMC, alphaL, nL, alphaH, nH)


    # define coefficiencts
    totentries = tree.GetEntries()
    nsig = RooRealVar("nsig", "nsig", 1000, 0, 2*totentries)
    
    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(MCPdf)
    coeff.add(nsig)
    
    model = ROOT.RooAddPdf("model", "model", suma, coeff)
    
    ds = RooDataSet("data", "dataset with x", tree, RooArgSet(B_M))

 
    model.fitTo(ds,RooFit.NumCPU(10),RooFit.BatchMode(1),RooFit.Extended(True))

    return meanMC, widthMC, nL, nH, alphaL, alphaH

def def_model_free(ws, fit_var_name, init_sig_m=5620, min_sig_m=5420, max_sig_m=5820, units_sig_m="MeV",
              init_sig_r=7.5, min_sig_r=4.5, max_sig_r=15, units_sig_r="MeV",
              min_sig_n=0, max_sig_n=40000, init_bkg_t = -0.002, min_bkg_t=-0.01, max_bkg_t=0.,
              units_bkg_t="MeV^{-1}", min_bkg_n=0, max_bkg_n=40000,
              init_sig_aL=1., min_sig_aL=0.00000001, max_sig_aL=45., init_sig_nL=2., min_sig_nL=0.00000001, max_sig_nL=200.,
              init_sig_aR=1., min_sig_aR=0.00000001, max_sig_aR=45., init_sig_nR=2., min_sig_nR=0.00000001, max_sig_nR=200.,
              ):
    """Define model with signal and bkg pdfs and import to workspace.

    Args:
        ws (RooWorkspace): workspace to store data, variables and pdfs
        min_sig_m (float, optional): min value of the signal mean parameter
        max_sig_m (float, optional): max value of the signal mean parameter
        units_sig_m (str, optional): units of the signal mean parameter
        min_sig_r (float, optional): min value of the signal resolution parameter
        max_sig_r (float, optional): max value of the signal resolution parameter
        units_sig_r (str, optional): units of the signal resolution parameter
        min_sig_n (int, optional): min value of the signal yield parameter
        max_sig_n (int, optional): max value of the signal yield parameter
        min_bkg_t (float, optional): min value of the bkg exponent parameter
        max_bkg_t (float, optional): max value of the bkg exponent parameter
        units_bkg_t (str, optional): units of the bkg exponent parameter
        min_bkg_n (int, optional): min value of the bkg yield parameter
        max_bkg_n (int, optional): max value of the bkg yield parameter
    Returns:
        True
    """
    # get parameters from ws
    fit_var = ws.var(fit_var_name)
    # define signal pdf
    sig_m  = RooRealVar("sig_m", "sig_m", init_sig_m, min_sig_m, max_sig_m,
                        units_sig_m)#, PlotLabel="#mu")
    sig_r  = RooRealVar("sig_r", "sig_r", init_sig_r, min_sig_r, max_sig_r,
                        units_sig_r)#, PlotLabel="#sigma")
    sig_aL = RooRealVar("sig_aL", "sig_aL", init_sig_aL, min_sig_aL, max_sig_aL)
    sig_nL = RooRealVar("sig_nL", "sig_nL", init_sig_nL, min_sig_nL, max_sig_nL)
    sig_aR = RooRealVar("sig_aR", "sig_aR", init_sig_aR, min_sig_aR, max_sig_aR)
    sig_nR = RooRealVar("sig_nR", "sig_nR", init_sig_nR, min_sig_nR, max_sig_nR)
    sig_pdf = RooCrystalBall("sig_pdf", "sig_pdf", fit_var, sig_m, sig_r,
                                sig_aL, sig_nL, sig_aR, sig_nR)
    sig_n = RooRealVar("sig_n", "sig_n", 1000, min_sig_n, max_sig_n)
                       #PlotLabel="N_{sig}")
    # define bkg pdf
    bkg_t = RooRealVar("bkg_t", "bkg_t", init_bkg_t, min_bkg_t, max_bkg_t,
                       units_bkg_t)#, PlotLabel="#tau")
    bkg_pdf = ROOT.RooExponential("bkg_pdf", "bkg_pdf", fit_var, bkg_t)
    bkg_n = RooRealVar("bkg_n", "bkg_n", 1000, min_bkg_n, max_bkg_n)
                       #PlotLabel="N_{bkg}")
    # build model
    model = ROOT.RooAddPdf("model", "model", RooArgList(sig_pdf, bkg_pdf),
                           RooArgList(sig_n, bkg_n))
    ws.Import(model)
    return model

def def_model_MC(ws, fit_var_name, MCparams, min_sig_n = 0, max_sig_n = 40000, init_bkg_t = -0.002, min_bkg_t=-0.01, max_bkg_t=0.,
                units_bkg_t="MeV^{-1}", min_bkg_n=0, max_bkg_n=40000):
    """Define model with signal and bkg pdfs and import to workspace, with signal
    parameters constrained by MC."""

    fit_var = ws.var(fit_var_name)

    meanMC, widthMC, nL, nH, alphaL, alphaH = MCparams

    meanMC.setConstant(False)
    widthMC.setConstant(False)
    nL.setConstant(True)
    nH.setConstant(True)
    alphaL.setConstant(True)
    alphaH.setConstant(True)

    sig_n = RooRealVar("sig_n", "sig_n", 1000, min_sig_n, max_sig_n)
                       #PlotLabel="N_{sig}")

    sig_pdf = RooCrystalBall("sig_pdf", '', fit_var, meanMC, widthMC, alphaL, nL, alphaH, nH)

    bkg_t = RooRealVar("bkg_t", "bkg_t", init_bkg_t, min_bkg_t, max_bkg_t,
                       units_bkg_t)#, PlotLabel="#tau")
    bkg_pdf = ROOT.RooExponential("bkg_pdf", "bkg_pdf", fit_var, bkg_t)
    bkg_n = RooRealVar("bkg_n", "bkg_n", 1000, min_bkg_n, max_bkg_n)
                       #PlotLabel="N_{bkg}")

    model = ROOT.RooAddPdf("model", "model", RooArgList(sig_pdf, bkg_pdf),
                           RooArgList(sig_n, bkg_n))

    ws.Import(model)
    return model

def fit(ws, fit_var_name, ds, model, output=""):
    """Fit data and plot result.

    Args:
        ws (RooWorkspace): workspace to store data, variables and pdfs
        fit_var_name (str): name of the discriminant variable to be fit
        output (str, optional): if given, the fit plot is saved with this name
    Returns:
        TCanvas containing the fit plot
    """

    # Obtain data and model from ws
    fit_var = ws.var(fit_var_name)
    ds = ws.data(ds)
    model = ws.pdf(model)

    c = ROOT.TCanvas()
    pad1 = ROOT.TPad( "pad1", "Histogram",0.,0.25,1.0,1.0)
    pad2 = ROOT.TPad( "pad2", "Residual plot",0.,0.,1.0,0.25)
    c.cd()
    pad1.Draw()
    pad2.Draw()

    frame = fit_var.frame()

    ds.plotOn(frame, RooFit.Name("histo_data"))

    # Fit
    fit_res = model.fitTo(ds,RooFit.NumCPU(10),RooFit.BatchMode(1),RooFit.Extended(True))
    model.plotOn(frame, RooFit.VisualizeError(fit_res, 1), RooFit.Name("curve_model"))

    pullhist = frame.pullHist()
    chi2 = frame.chiSquare()
    #Plot.plot(bigLabels=True, residualBand=True)

    pad2.cd()
    pullhist.Draw()
    
    # Plot to check fit results

    pad1.cd()
    model.plotOn(frame, ROOT.RooFit.Components("sig_pdf"), ROOT.RooFit.LineStyle(ROOT.kDashed),
                 ROOT.RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame, ROOT.RooFit.Components("bkg_pdf"), ROOT.RooFit.LineStyle(ROOT.kDashed),
                 ROOT.RooFit.LineColor(ROOT.kGreen))
    params = RooArgSet(ws.var("sig_n"), ws.var("bkg_n"), ws.var("meanMC"),
                       ws.var("widthMC"), ws.var("bkg_t"))
    model.paramOn(frame, Parameters=params, Layout=(.55,.95,.93))
    pad1.cd()
    frame.Draw()

    if output!="":
        for suf in ['.root', '.png', '.pdf']:
            c.SaveAs(output+suf)

def get_sweights(ws, ds, model, fix_params, f_weights):
    """Get sweights.
    Args:
        ws (RooWorkspace): workspace to store data, variables and pdfs
        ds (str): name of the data set in ws to use
        model (str): name of the pdf in ws to use
        fix_params (list): list with variables in ws to be fixed for the splot,
                           all non-yield variables should be fixed
        f_weights (str): name of file to store sweights as TTree
    Returns:
        sData: ds with sweights
    """
    # Get yields, set other params constant
    sig_n = ws.var('sig_n')
    bkg_n = ws.var('bkg_n')
    ds = ws.data(ds)
    model = ws.pdf(model)
    for p in fix_params: ws.var(p).setConstant()
    sPlot = ROOT.RooStats.SPlot('sdata', 'sdata', ds, model,
                                RooArgList(sig_n, bkg_n))
    # Check weights make sense
    print ("Check SWeights:")
    print ("Sig yield is:", sig_n.getVal())
    print ("From sWeights:", sPlot.GetYieldFromSWeight("sig_n"))
    print ("Bkg yield is:", bkg_n.getVal())
    print ("From sWeights:", sPlot.GetYieldFromSWeight("bkg_n"))

    for i in range(10):
        print ("Sig weight:", sPlot.GetSWeight(i, "sig_n"))
        print ("Bkg weight:", sPlot.GetSWeight(i, "bkg_n"))
        print ("Total Weight:", sPlot.GetSumOfEventSWeight(i))

    # Import ds with weights and save to TTree
    f = ROOT.TFile(f_weights, 'recreate')
    ds.convertToTreeStore()
    f.cd()
    ds.Write()
    f.Close()
    #ws.Import(ds)
    return ds


def main(fdata, tree, fit_var_name, fit_var_range, fout, plot_dir, fMC):
    """Obtain sweights from fit to discriminant variable and
    plot sweighted control variable vs mc distribution.

    MC is fit first to obtain pdf parameters which are fixed
    on fit to data.

    Args:
        fdata (str): name of root file with data
        tree (str): name of TTree in root file
        fit_var_name (str): name of fit variable in TTree
        fout (str): name of root file to store sweights
    Returns:
        RooWorkspace with used variables, pdfs and data
    """
    # Read data
    f_data = ROOT.TFile(fdata)
    t_data = f_data.Get(tree)
    t_data=t_data.CopyTree("Bu_DTFPV_JpsiConstr_MASS>0")

    # Import to ws
    wspace = ROOT.RooWorkspace("BDT_presel")
    import_data(wspace, t_data, fit_var_name, fit_var_range, ds_name="ds_data")
    # Obtain pdf
    totentries = t_data.GetEntries()
    if fMC=="":
        def_model_free(wspace, fit_var_name, max_sig_n=1.5*totentries, max_bkg_n=1.5*totentries)
    else:
        f_MC = ROOT.TFile(fMC)
        t_MC = f_MC.Get(tree)
        MCparams = fit_MC(t_MC, fit_var_name, 5300, xmin = fit_var_range[0]-100, xmax = fit_var_range[1]+100)
        def_model_MC(wspace, fit_var_name, MCparams, max_sig_n=1.5*totentries, max_bkg_n=1.5*totentries)

    output = os.path.join(plot_dir, "fit_data")
    fit(wspace, fit_var_name, "ds_data", "model", output)
    fix_params = ['meanMC', 'widthMC', 'alphaL', 'nL', 'alphaH', 'nH',
                  'bkg_t']
    ds = get_sweights(wspace, "ds_data", "model", fix_params, fout)
    return wspace


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_f", action="store", type=str, help="data file")
    parser.add_argument("params", action="store", type=str, help="yaml file with the parameters of the fit")
    parser.add_argument("out_f" , action="store", type=str, help="output file")
    parser.add_argument("-t", "--tree", default="DecayTree", action="store",
                        type=str, help="tree name")
    parser.add_argument("-d", "--directory" , default="BDT_presel/weights/" , action='store',
                        help="directory to store splots")
    parser.add_argument("-m", "--MC_f", default = "", action="store", type=str, help="MC file in case that signal tails must be constrained.")
    args = parser.parse_args()
    # make dir
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
    # Read data#

    from yaml import safe_load
    with open(args.params, "r") as config:
        params = safe_load(config)
    fit_var = params["fit_var"]
    range_var = (params["xmin"],params["xmax"])
    ws = main(args.data_f, args.tree, fit_var, range_var, args.out_f, args.directory, args.MC_f)


data="/eos/lhcb/user/p/pvidrier/roots/data_presel_with_cuts.root"
paramete="BDT_presel/params.yaml"
outputname="output"
montecarlo="/eos/lhcb/user/p/pvidrier/roots/mc_presel_with_cuts_Bu_JpsiConstr.root"

# I WRITE
# python BDT_presel/get_sweights.py "/eos/lhcb/user/p/pvidrier/roots/data_presel_with_cuts.root" "BDT_presel/params.yaml" "BDT_presel/weights/weights.root" -m "/eos/lhcb/user/p/pvidrier/roots/mc_presel_with_cuts_Bu_JpsiConstr.root"



#if not os.path.exists("/plots"):
#    os.mkdir("/plots")
# Read data

#from yaml import safe_load
#with open(parameters, "r") as config:
#    params = safe_load(config)
#fit_var = params["fit_var"]
#range_var = (params["xmin"],params["xmax"])
#main(data, "DecayTree", fit_var, range_var, outputname, "/plots", montecarlo)


