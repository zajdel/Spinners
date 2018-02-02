# Calculate the bias over angular velocity, plot results
# input: string prefix(prefix of the set), int number of numbers from beginning to calculat bias over)

from utilities import *
from scipy.stats import norm,gamma,expon,beta
import matplotlib.mlab as mlab
import pandas as pd
from matplotlib.patches import Rectangle

#csv = sys.argv[1] # csv to make histogram from

colors = ['b', 'g', 'r', 'c', 'm', 'k']
# data = pd.read_csv(csv + '.csv', sep=',', header=None)

csv = ['100n_asp_30s','100n_asp_60s','100n_asp_120s']

times = [30,60,120]

font = {'family' : 'Myriad Pro',
        'size'   : 16}

plt.rc('font', **font)

for k in range(0,len(csv)):
    data = np.loadtxt(csv[k] + '.csv', delimiter=',')
    
    plt.subplot(221)
    d = np.ones(len(data[:,0]))-data[:,0]
    d2=data[:,0]
    params = gamma.fit(d)
    n,bins,patches = plt.hist(d2, bins=40, range=(0,1),facecolor=colors[k], normed=True,alpha=0.2)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    plot_x = np.linspace(0,1.01,1000)
    pdf_fitted = gamma.pdf(plot_x,params[0],params[1],params[2])
    #l = plt.plot(plot_x,np.divide(pdf_fitted[::-1],sum(n)),colors[k]+'-',linewidth=2)
    plt.xlabel(r'$B$', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.ylim((0,0.15))
    plt.xlim((0,1))
    plt.errorbar(np.average(data[:,0]),0.15/2,xerr=np.std(data[:,0])/np.sqrt(len(data[:,0])),fmt='-',ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
    plt.axvline(x=np.average(data[:,0]),c=colors[k],linewidth=2)
    
    tau_range = 15
    plt.subplot(224)
    params = expon.fit(data[:,1])
    n,bins,patches = plt.hist(data[:,1], bins=40, range=(0,tau_range/3), normed=True,facecolor=colors[k], alpha=0.2)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    plot_x = np.linspace(0,30,1000)
    pdf_fitted = expon.pdf(plot_x,params[0],params[1])
    #l = plt.plot(plot_x,np.divide(pdf_fitted,sum(n)),colors[k]+'-',linewidth=2)
    plt.xlabel(r'$\tau_{cw}$ [seconds]', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.ylim((0,0.4))
    plt.xlim((0,tau_range/3))
    plt.errorbar(np.average(data[:,1]),0.4/2,xerr=np.std(data[:,1])/np.sqrt(len(data[:,1])),fmt='-',ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
    plt.axvline(x=np.average(data[:,1]),c=colors[k],linewidth=2)

    ax=plt.subplot(222)
    params = expon.fit(data[:,2])
    n,bins,patches = plt.hist(data[:,2], bins=40, range=(0,tau_range), normed=True,facecolor=colors[k], alpha=0.2)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    plot_x = np.linspace(0,30,1000)
    pdf_fitted = expon.pdf(plot_x,params[0],params[1])
    #l = plt.plot(plot_x,np.divide(pdf_fitted,sum(n)),colors[k]+'-',linewidth=2)
    plt.xlabel(r'$\tau_{ccw}$ [seconds]', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xlim((0,tau_range))
    plt.ylim((0,0.4))
    plt.errorbar(np.average(data[:,2]),0.4/2,xerr=np.std(data[:,2])/np.sqrt(len(data[:,2])),fmt='-',ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
    plt.axvline(x=np.average(data[:,2]),c=colors[k],linewidth=2)
	
    handles = [Rectangle((0,0),1,1,color=c,alpha=0.3,ec="k") for c in colors[0:3]]
    labels= ["30 sec","60 sec", "120 sec"]
    plt.legend(handles, labels, fontsize=16)

    plt.subplot(223)
    params = norm.fit(data[:,3]/times[k])
    n,bins,patches = plt.hist(data[:,3]/times[k], bins=40, range=(0,200), normed=True,facecolor=colors[k], alpha=0.2)
    for item in patches:
        item.set_height(item.get_height()/sum(n))
    plot_x = np.linspace(0,200,1000)
    pdf_fitted = norm.pdf(plot_x,params[0],params[1])
    #l = plt.plot(plot_x,np.divide(pdf_fitted,sum(n)),colors[k]+'-',linewidth=2)
    plt.xlabel(r'$N_s$', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xlim((0,200))
    plt.ylim((0,0.15))
    plt.errorbar(np.average(data[:,3]/times[k]),0.15/2,xerr=np.std(data[:,3]/times[k])/np.sqrt(len(data[:,3])/times[k]),fmt='-',ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
    plt.axvline(x=np.average(data[:,3]/times[k]),c=colors[k],linewidth=2)
	
plt.suptitle('100 nM Aspartate (N = 126)',fontsize=32)
plt.show()