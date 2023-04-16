import ROOT
import numpy as np
import pickle
from xgboost import XGBClassifier

def convert_df_to_float(df, variables = []):
   corr_vars = []
   for var in variables:
      if True: # df.GetColumnType(var) != float:
         df = df.Define(f"f_{var}",f"(float) {var}")
         corr_vars.append(f"f_{var}")
      else:
         corr_vars.append(var)
   print(corr_vars)
   return df, corr_vars

def load_data(fn_data, fn_mc, treename = "DecayTree", variables = [], first_half = True):

   # /*
   #    Train https://root.cern/doc/master/tmva101__Training_8py.html
   # */
   
   data_bkg = ROOT.RDataFrame(treename, fn_data)
   data_sig = ROOT.RDataFrame(treename, fn_mc)

   n_bkg = int(data_bkg.Count().GetValue()/2)
   n_sig = int(data_sig.Count().GetValue()/2)
   n_min = min(n_bkg, n_sig)
   if (first_half == True):
      data_bkg = data_bkg.Range(n_min)
      data_sig = data_sig.Range(n_min)
   else:
      data_bkg = data_bkg.Range(n_min, 2*n_min-1)
      data_sig = data_sig.Range(n_min, 2*n_min-1)

# // BDT related cuts              
   cut_sidebands = "(B_M < 5000) || (B_M >5600)"
   # cut_True = "" # "B_BKGCAT == 0"
   data_bkg = data_bkg.Filter(cut_sidebands) 
   # data_sig = data_sig.Filter(cut_True) 

   data_bkg, corr_vars = convert_df_to_float(data_bkg, variables)
   data_sig, corr_vars = convert_df_to_float(data_sig, variables)

   data_bkg = data_bkg.AsNumpy()
   data_sig = data_sig.AsNumpy()

   # Convert inputs to format readable by machine learning tools
   x_sig = np.vstack([data_sig[var] for var in corr_vars]).T
   x_bkg = np.vstack([data_bkg[var] for var in corr_vars]).T
   x = np.vstack([x_sig, x_bkg])
 
   # Create labels
   num_sig = x_sig.shape[0]
   num_bkg = x_bkg.shape[0]
   y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
      
   # Compute weights balancing both classes
   num_all = num_sig + num_bkg
   w = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])

   return x, y, w

def train_xgboost(fn_data, fn_mc, fn_BDT, treename = "DecayTree", variables = []):
      # Load data
   x, y, w = load_data(fn_data, fn_mc, variables=variables)
   x_eval, y_eval, w_eval = load_data(fn_data, fn_mc, variables=variables, first_half=False)

   # Fit xgboost model
   bdt = XGBClassifier(max_depth=3, n_estimators=500)
   bdt.fit(x, y, sample_weight=w, verbose=True, eval_set=[(x_eval,y_eval)])

   # Save model in TMVA format
   print("Training done on ",x.shape[0],"events. Saving model")
   ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", fn_BDT, num_inputs=x.shape[1])



if __name__ == "__main__":
   fn_data = "/eos/lhcb/user/a/alobosal/HeavyFiles/DATAfiles/KstIsoG/run2/LL/preselection_run2.root"
   fn_MC   = "/eos/lhcb/user/a/alobosal/HeavyFiles/MCfiles/KstIsoG/run2/LL/preselection_run2.root"
   fn_BDT  = "/afs/cern.ch/work/a/alobosal/analysis/reproduce/isospin_b2kstg/run2/KstIsoG/selection/BDT_XGBoost_run2.root"
   
   variables = [
      "gamma_eta",
      "B_PT",
      "B_DIRA_OWNPV",
      "B_FDCHI2_OWNPV",
      "B_IPCHI2_OWNPV",
      "gamma_PT",
      "Kst_892_plus_M",
      "Kst_892_plus_IPCHI2_OWNPV",
      "Kst_892_plus_ENDVERTEX_CHI2",
      "KS0_IPCHI2_OWNPV",
      "KS0_FD_OWNPV",
      "KS0_DIRA_ORIVX",
      "KS0_M",
      "KS0_ENDVERTEX_CHI2",
      "piminus_IPCHI2_OWNPV",
      "piplus_IPCHI2_OWNPV",
      "piplus0_IPCHI2_OWNPV",
      "piminus_PT",
      "piplus_PT"
   ]

   train_xgboost(fn_data, fn_MC, fn_BDT, variables=variables)