import ROOT 

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

def convert_df_to_float(df, variables = []):
   corr_vars = []
   for var in variables:
      if df.GetColumnType(var) != float:
         df = df.Define(f"f_{var}",f"(float) {var}")
         corr_vars.append(f"f_{var}")
      else:
         corr_vars.append(var)
   return df, corr_vars

def apply_BDT_add_branch(fn_bdt, filename, treename = "DecayTree"):

   # /*
   # Apply BDT to data and make a new branch with the BDT response
   # */

   df = ROOT.RDataFrame(treename, filename)
   bdt = ROOT.TMVA.Experimental.RBDT[""]("myBDT", fn_bdt)

   df, corr_vars = convert_df_to_float(df, variables)
   df2 = df.Define("bdt_output",ROOT.TMVA.Experimental.Compute[19,float](bdt),corr_vars)
   df2.Snapshot("DecayTree","b.root")
   
if __name__=="__main__":
   fn_data = "/eos/lhcb/user/a/alobosal/HeavyFiles/DATAfiles/KstIsoG/run2/LL/preselection_run2.root"
   fn_MC   = "/eos/lhcb/user/a/alobosal/HeavyFiles/MCfiles/KstIsoG/run2/LL/preselection_run2.root"
   # fn_BDT  = "/eos/lhcb/user/a/alobosal/HeavyFiles/BDT_run2.root"
   fn_BDT  = "/afs/cern.ch/work/a/alobosal/analysis/reproduce/isospin_b2kstg/run2/KstIsoG/selection/BDT_XGBoost_run2.root"
   apply_BDT_add_branch(fn_BDT,fn_data)