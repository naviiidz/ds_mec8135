import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt



lgnd_list=[]
x = np.arange(0, 200, 0.001)

for k in [1,2,3,10,100]:
    #plot Chi-square distribution with k degrees of freedom
    plt.plot(x, chi2.pdf(x, df=k))
    
    #Plotting the Results
    lgnd_list.append("mu=%.2f & sd=%.2f" %(k,(2*k)))
    plt.xlabel('Data points')
    plt.ylabel('Probability Density')
    plt.xlim(0,120)
    plt.ylim(0,0.3)
    plt.legend(lgnd_list)

plt.show()
    
    
