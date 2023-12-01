import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt 


a=np.array([.376,.376,.377,.377,.377,.378,.377,.378,.377])
b=np.array([.376,.374,.374,.373,.372,.372,.371,.37,.37])
#plt.rcParams["font.family"] = "Times New Roman"

x = np.array([0,10,20,30,40,50,60,70,80])
plt.plot(x,a)
plt.plot(x,b)
#x=np.arange(6)
#markers_on = [1]
#plt.plot(x,a,'-ko',markevery=markers_on)
#markers_on = [2]
#plt.plot(x,b,'-kD',linestyle='dashed',markevery=markers_on)
#plt.ylim(.6,1)
plt.text(50,.390, "n=849", size=15,
         ha="right", va="top",
         )

plt.xlim(0,85)
plt.ylim(.369,.38)
plt.yticks([.3800,.3750,.3700], [.38,.375,.37])
plt.xlabel('Percentage of edges modified',fontsize=16,fontweight="bold")
plt.ylabel('Correlation of estimated protein expression',fontsize=16,fontweight="bold")
plt.legend(['Deletion','Addition'],loc ="center right",fontsize=14)
plt.show()
plt.savefig('849.pdf')
