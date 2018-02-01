# Calculate the bias over angular velocity, plot results
# input: string prefix(prefix of the set), int number of numbers from beginning to calculat bias over)

from utilities import *

import pandas as pd

csv = sys.argv[1] # csv to make histogram from

colors = ['b', 'g', 'r', 'c', 'm', 'k']

# data = pd.read_csv(csv + '.csv', sep=',', header=None)

csv = ['10u_leu_30s','10u_leu_60s','10u_leu_120s']

for k in range(0,len(csv)):
    data = np.loadtxt(csv[k] + '.csv', delimiter=',')
    plt.subplot(221)
    plt.hist(data[:,0], bins=20, range=(0,1), normed=1,facecolor=colors[k], alpha=0.25)
    plt.xlabel('Bias')
    plt.ylabel('Frequency')
    plt.subplot(222)
    plt.hist(data[:,1], bins=20, range=(0,30), normed=1,facecolor=colors[k], alpha=0.25)
    plt.xlabel('$\tau_{ccw}$')
    plt.ylabel('Frequency')
    plt.subplot(223)
    plt.hist(data[:,2], bins=20, range=(0,30), normed=1,facecolor=colors[k], alpha=0.25)
    plt.xlabel('$\tau_{cw}$')
    plt.ylabel('Frequency')
    plt.subplot(224)
    plt.hist(data[:,3], bins=20, range=(0,100), normed=1,facecolor=colors[k], alpha=0.25)
    plt.xlabel('$N_s$')
    plt.ylabel('Frequency')




plt.show()
