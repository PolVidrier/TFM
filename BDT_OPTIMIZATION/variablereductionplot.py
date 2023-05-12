import matplotlib.pyplot as plt

numbervariables=[27,25,21,23,21,19,16,15,14,13,12,11,10,9,8,7,6,5,4,3]
aucs=[0.9977653233,0.9944685382,0.9938564678,0.9947268456,0.9929709619, 0.9913170047,0.9915300659, 0.9906077210,0.9927032792,0.9926769881, 0.9912005838,0.9909560225, 0.9871667906,0.9871924652,0.9813609920,0.98417400303,0.9745831200,0.9741997587,0.9433283412,0.8774873581]
pro=0.9813609920
aucstrain=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9999999341,0.9999995427,0.9999866765,0.9998269485,0.9973279593,0.9755946278]

# WITH THE CHANGE IN ANGLE
numbervariables=[29,27,23,21,19,17,16,15,14,12,11,10,9,8,7,6,5,4,3]
aucs=[0.998303039,0.9942209575,0.9918266653,0.9925190670,0.9925373923,0.9928377757,0.9914757167,0.9923863991,0.9913164993,0.9883687429,0.9919987474,0.9896435770,0.9908887740,0.9879195990,0.9860706127,0.9806130326,0.9739293608,0.9692789822,0.8821717330]
aucstrain=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9999985015,0.9999902863,0.9997411154,0.998724543,0.9696870281]

plt.plot(numbervariables,aucs,"-o",label="Test")
plt.plot(numbervariables,aucstrain,"-yo",label="Train")
#plt.plot([8,8],[0.9813609920,0.9999999341],'ro',label="First AUC of train <1 in 10 digits")
#plt.plot([7,7],[0.98417400303,0.9999995427],'go',label="Chosen sweet spot")
plt.plot([9,9],[0.9908887740,1],'go',label="Chosen sweet spot")
plt.plot([7,7],[0.9860706127,0.9999985015],'ro',label="First AUC of train <1 in 10 digits")
plt.xlabel("Number of variables")
plt.ylabel("AUC")
plt.title("Variable Reduction")
plt.legend(loc="lower right")
plt.savefig("BDT_OPTIMIZATION/plots/VariableReduction.png")
plt.close()