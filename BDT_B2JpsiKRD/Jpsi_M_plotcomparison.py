import matplotlib.pyplot as plt
import numpy as np

mas_pred = np.loadtxt("BDT_B2JpsiKRD/data_Jpsi_M.txt", dtype=float)
mas_pred0 = np.loadtxt("BDT_B2JpsiKRD/data_Jpsi_M_0BREM.txt", dtype=float)
mas_pred1 = np.loadtxt("BDT_B2JpsiKRD/data_Jpsi_M_1BREM.txt", dtype=float)
mas_pred2 = np.loadtxt("BDT_B2JpsiKRD/data_Jpsi_M_2BREM.txt", dtype=float)

mas_predbrem=np.concatenate((mas_pred0,mas_pred1,mas_pred2))

plt.hist(mas_pred,label="Total",bins=100,alpha=0.5)
plt.hist(mas_predbrem,label="Sum of BREMS", bins=100,alpha=0.5)
plt.title("Jpsi_M Prediction")
plt.xlabel("Jpsi_M")
plt.legend(loc="upper left")
plt.savefig("BDT_B2JpsiKRD/plots/Jpsi_M_BREM_comparison.png")
plt.close()