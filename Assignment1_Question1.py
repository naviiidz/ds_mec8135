import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

#init without library
n=10**4
colors=['r','g','b','c','y']
k=[1,2,3,10,100]

#init with library
lgnd_list=[]
x = np.arange(0, 200, 0.001)


#method 1: plotting using hist
for j in range(len(k)):
  squared_samples=np.zeros(n)
  for i in range(k[j]):
    samples = np.random.standard_normal(n)
    squared_samples += samples**2
  fig=plt.figure()
  plt.title('Chi-square (k=%d)'%k[j])
  plt.xlabel('Value')
  plt.ylabel('Density')
  plt.hist(squared_samples, bins=200, density=True, color=colors[j])
  plt.show()

# method 2: plotting with Scipy lib
for k in [1,2,3,10,100]:
    plt.plot(x, chi2.pdf(x, df=k))
    lgnd_list.append("k=mu=%d & sd=%.2f" %(k,(2*k)))
    plt.xlabel('Data points')
    plt.ylabel('Probability Density')
    plt.title("Plot with Scipy library")
    plt.xlim(0,120)
    plt.ylim(0,0.3)
    plt.legend(lgnd_list)

plt.show()
    
    