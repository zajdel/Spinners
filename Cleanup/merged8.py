# Calculate the bias over angular velocity, plot results
# input: name of file, int number of numbers from beginning to calculate bias over)

import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import matplotlib.lines as mlines
from scipy.optimize import leastsq
import numpy.polynomial.polynomial as poly

path = sys.argv[1]
conc = sys.argv[2].split(',')
# concentration = sys.argv[1]
file_num = int(sys.argv[3])
#count = int(sys.argv[5])
count=2812
N = [468, 937, 1875, 2812, 3750]

font = {'family' : 'Myriad Pro',
        'size'   : 24}

plt.rc('font', **font)

interval_bias = lambda s: np.sum((-np.array(s) + 1) / 2) / len(s)  # CCW / (CCW + CW); s is interval over which to compute bias, s is signs of rotation direction. NOTE: correct if cw is positive, ccw is negative.

	 # from http://people.duke.edu/~ccc14/pcfb/analysis.html
def logistic4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D

def residuals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    A,B,C,D = p
    err = y-logistic4(x, A, B, C, D)
    return err

def peval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C,D = p
    return logistic4(x, A, B, C, D)
	
def leufit(x, A, B, C):
    """4PL lgoistic equation."""
    return A*np.power(x,2.0) + B*x + C

def leuresiduals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    A,B,C = p
    err = y-leufit(x, A, B, C)
    return err

def leupeval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C = p
    return leufit(x, A, B, C)

def hysteresis_threshold(trace,rel):
    max = np.percentile(trace[:1875],99.0)
    min = np.percentile(trace[:1875],1.0)
    tH = max - (np.absolute(max)+np.absolute(min))*rel
    tL = min + (np.absolute(max)+np.absolute(min))*rel
    dir = np.zeros(len(trace))
    
    high = True
    for k in range(0,len(trace)):
        if high:
            if trace[k]< tL:
                dir[k] = -1
            else:
                dir[k] = 1
                high = False
        elif ~high:
            if trace[k] > tH:
                dir[k] = 1
            else:
                dir[k] = -1
                high = True
            
    return dir

def moving_average(values, window=8):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def compute_features(trace, window, sensitivity):
    # Input: trace is single bacterium raw time series
    #       window is size of window over which we (locally) compute bias.
    # Output: time series of bias at all overlapping intervals of window-length.  (len: len(trace) - window)

    bias = []
	
    # 1. Derivative of 1al. (Angular velocity) Use to get signs, which tell us CCW or CW.
    unwrapped = np.unwrap(np.asarray(trace))
    ma_trace = moving_average(unwrapped, 8) # 8*1/32 fps ~ 250 ms median filter window
    velocity = np.convolve([-0.5,0.0,0.5], ma_trace, mode='valid')    
    d = hysteresis_threshold(velocity,sensitivity)
	
    # Optionally:
    #       median_filtered_conv = median_filter(conv, 7) # pick window size based on result. second arg is odd number.

    # 2. Get direction of rotation (CCW & CW)
    signs = np.sign(d)  # Positive values correspond to cw rotation. Negative = ccw rotation.

    # Optionally:
    # filtered_signs = median_filter(signs, 9) # pick window size based on result. second arg is odd number.
    # should probably use some hysteresis thresholding instead, try it!

    # 3. Compute bias over each window-length interval
    # no sliding window as here:
    # use first 'first' frames to compute bias
    interval = signs[:window]

    features = {}
    total = 0
    cw_intervals = []
    ccw_intervals = []
    run = 1
    n_switches = 0
    prev_direction = interval[0]
    
    for k in range(1,len(interval)):
        total += (-interval[k] + 1) / 2
		
        # check if the run is continuing
        if interval[k] == prev_direction:
            run += 1
            # if we made it to the end, count this as an interval cut off at end
            if k == len(interval)-1:
                if prev_direction == 1:
                    cw_intervals.append(run)
                elif prev_direction == -1:
                    ccw_intervals.append(run)			    
        # otherwise, we have ourselves a switch!
        else:
            n_switches += 1
            if prev_direction == 1:
                cw_intervals.append(run)
            elif prev_direction == -1:
                ccw_intervals.append(run)
            run = 1
            prev_direction = interval[k]

    features["bias"] = total / window
    features["cw"] = np.nanmean(cw_intervals)*.032
    features["ccw"] = np.nanmean(ccw_intervals)*.032
    if math.isnan(features["cw"]):
        features["cw"]=0
    if math.isnan(features["ccw"]):
        features["ccw"]=0	
    features["switch"] = n_switches
    return features


def compute_features_for_each_trace(status,traces,times):
    features = {"biases":[],"cw":[], "ccw":[], "switch":[]}
	# remove traces that have a status of 0 (rejected)
    for s,trace in zip(status,traces):
	    if (s>0):
             # unwrap and smooth the trace before computing the bias
             # trace =smooth(np.unwrap(trace*np.pi/180.0),11)*180/np.pi;
             results = compute_features(trace, times,s)  # Set first and frames so that it's about 10 s of data.
             features["biases"].append(results["bias"])
             features["cw"].append(results["cw"])
             features["ccw"].append(results["ccw"])
             features["switch"].append(results["switch"])
    return features

def graph(fts):
    biases = []
    cws = []
    ccws = []
    switches = []
    for pre in conc:
        # filename = pre + "_" + chem + ".csv"
        # data = np.loadtxt(filename, delimiter=",")
        # bias, cw, ccw, switch = np.hsplit(data, np.array([1,2,3]))
        features = fts[pre]
        bias, cw, ccw, switch = features["biases"], features["cw"], features["ccw"], features["switch"]
        biases.append((np.average(bias), np.std(bias)/(np.sqrt(len(bias)))))
        cws.append((np.average(cw), np.std(cw)/(np.sqrt(len(cw)))))
        ccws.append((np.average(ccw), np.std(ccw)/(np.sqrt(len(ccw)))))
        switches.append((np.average(switch), np.std(switch)/(np.sqrt(len(switch)))))

    plots = [biases, cws, ccws, switches]
    label = ["Average Biases", "Average Clockwise Time", "Average Counterclockwise Time", "Average Switches"]
    ylimit = [[0.5,1.0],[0,30],[0,30],[0,1]]
    ylabel = ["Bias","seconds","seconds","number"]
    c_vals = {"1m": -3,"100u": -4,"10u": -5,"1u": -6,"100n": -7, "control": -9}
    # for i in range(0,4):
        # plt.figure(i)
        # plt.xlabel('Concentrations')
        # plt.title(label[i])
        # plt.ylim(ylimit[i])
        # plt.ylabel(ylabel[i])
        # plt.xlim(-10,0)
        # c = 0
        # for avg, std in plots[i]:
            # plt.errorbar(c_vals[conc[c]], avg, std,fmt='--o',ecolor=colors[k],c=colors[k])
            # c += 1
    # plt.show()

def graph_time(features):
    i = 4
    k=0
    
    colors = ['c', 'm', 'k']
    shapes = ['o', '^', 'x']
    cons = ['1 '+u'\u03BC'+'M','100 nM','0 M']
    for con in conc:
        plt.figure(1)
        #plt.subplot(121)
        feats = features[con]
        b = feats["biases"]
        plt.xlabel('Time [seconds]')
        plt.ylabel(r'$B(T)$')
        xfit = []
        yfit = []
        for n in N:
            avg, std = b[n]
            xfit.append(n*.032)
            yfit.append(avg)
            plt.errorbar(n*.032, avg, yerr=std,fmt=shapes[k],markersize=10,ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
        m,b = np.polyfit(xfit, yfit, 1)
        handles = [mlines.Line2D((0,0.5),(1,0.5),lw=2,color=colors[kk],marker=shapes[kk]) for kk in range(0,len(colors))]
        #print ('B',con,m,b)
        x_plot = np.linspace(0,120,1000)
        plt.plot(x_plot,np.add(np.multiply(x_plot,m),b),c=colors[k],markersize=10,linewidth=2,label=con)
        plt.ylim((0.50,1.0))
        plt.xlim(-2,122)
        labels=cons
        lgnd = plt.legend(handles, labels, loc=1, fontsize=18,numpoints=1)
        lgnd.legendHandles[0]._legmarker.set_markersize(10)
        lgnd.legendHandles[1]._legmarker.set_markersize(10)
        lgnd.legendHandles[2]._legmarker.set_markersize(10)
        #plt.legend(handles, labels, loc=1, fontsize=16)
        i+=1
        # c1 = feats["cw"]
        # plt.figure(i)
        # plt.xlabel('Frames')
        # for n in N:
            # avg, std = c1[n]
            # plt.errorbar(n*.032, avg, yerr=std,fmt='o',ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
        # i+=1
        # c2 = feats["ccw"]
        # plt.figure(i)
        # plt.xlabel('Frames')
        # for n in N:
            # avg, std = c2[n]
            # plt.errorbar(n*.032, avg, yerr=std,fmt='o',ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
        # i+=1
        # plt.figure(1)
        # s = feats["switch"]
        # plt.subplot(122)
        # plt.xlabel('Time [seconds]')
        # plt.ylabel(r'$N_s$')
        # xfit = []
        # yfit = []
        # for n in N:
            # avg, std = s[n]
            # xfit.append(n*.032)
            # yfit.append(avg)
            # plt.errorbar(n*.032, avg, yerr=std,fmt=shapes[k],ecolor=colors[k],c=colors[k],capsize=5, elinewidth=2, capthick=2)
        # m,b = np.polyfit(xfit, yfit, 1)
        # x_plot = np.linspace(0,120,1000)
        # plt.plot(x_plot,np.add(np.multiply(x_plot,m),b),c=colors[k],linewidth=2)
        # handles = [mlines.Line2D((0,0.5),(1,0.5),lw=2,color=colors[kk],marker=shapes[kk]) for kk in range(0,len(colors))]
        # print ('N',con,m,b)
        # labels= cons
        # plt.legend(handles, labels, loc=2, fontsize=20,numpoints=1)
        # plt.ylim(0,60)
        # plt.xlim(-2,122)
        # if chem=='asp':
            # plt.suptitle('Asp',fontsize=32)
        # elif chem=='leu':
            # plt.suptitle('Leu',fontsize=32)
        i+=1
        i = 4
        k+=1
    plt.show()
	
def graph_conc(features,features2):
    k=0
    c_vals = {"1m":-3,"100u": -4,"10u": -5,"1u": -6,"100n":-7}
    c_vals_sanitized = {"1m": -3,"100u": -4,"10u": -5,"1u": -6,"100n": -7, "control": -9}

    colors = ['b', 'r']
    times = ['Asp','Leu']
    shapes = ['o', '^']

    #NN = (937, 1875, 3750)
    NN = [3750]
    for nn in NN:
        f = plt.figure(figsize=(8,5))
        ax=plt.subplot(1,2,(1,2))
        plt.xlabel('Concentration [M]')
        plt.xlim(0.5*math.pow(10,-7),1.5*math.pow(10,-3))
        plt.ylabel(r'$N_s$')
        ax.set_xscale("log", nonposx='clip')    
        xfit = []
        yfit = []		
        yfit2 = []
        for con in conc:
            feats = features[con]
            feats2=features2[con]
            s = feats["switch"]
            s2=feats2["switch"]
            avg, std = s[nn]
            avg2, std2 = s2[nn]
            k=0
            plt.errorbar(math.pow(10,c_vals[con]), avg, yerr=std,marker=shapes[k],ecolor=colors[k],c=colors[k],capsize=5, markersize=10,elinewidth=2, capthick=2)
            k = 1
            plt.errorbar(math.pow(10,c_vals[con]), avg2, yerr=std2,marker=shapes[k],ecolor=colors[k],c=colors[k],capsize=5, markersize=10,elinewidth=2, capthick=2)
            if con<>'control':
                xfit.append(math.pow(10,c_vals[con]))
                yfit.append(avg)
                yfit2.append(avg2)
        m,b = np.polyfit(np.log10(xfit), yfit, 1)
        m2,b2 = np.polyfit(np.log10(xfit), yfit2, 1)
        x_plot = np.logspace(-7,-3,1000)		
        k =0
        plt.plot(x_plot,np.add(np.multiply(np.log10(x_plot),m),b),c=colors[k],linewidth=2)
        k=1
        plt.plot(x_plot,np.add(np.multiply(np.log10(x_plot),m2),b2),c=colors[k],linewidth=2)
        plt.ylim(0,60)
        handles = [mlines.Line2D((0,0.5),(1,0.5),lw=2,color=colors[kk],marker=shapes[kk]) for kk in range(0,len(colors))]
        labels= times
        lgnd=plt.legend(handles, labels, loc=1, fontsize=18,numpoints=1)
        lgnd.legendHandles[0]._legmarker.set_markersize(10)
        lgnd.legendHandles[1]._legmarker.set_markersize(10)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()


def compute_features_over_time(status, traces):
    averaged = {}
    for c in N:
        #somehow count the number of entries here
        averaged[c] = {"biases":[], "cw": [], "ccw": [], "switch": []}
        temp = compute_features_for_each_trace(status,traces,c)
        averaged[c]["biases"].extend(temp["biases"])
        averaged[c]["cw"].extend(temp["cw"])
        averaged[c]["ccw"].extend(temp["ccw"])
        averaged[c]["switch"].extend(temp["switch"])
    return averaged

def combine_features_over_time(features, length):
    result = {}
    for con in conc:
        concen = features[con]
        #change to collect array of values then just get avg, std out
        bi_ar = []
        cw_ar = []
        ccw_ar = []
        sw_ar = []
        temp = {"biases":{}, "cw": {}, "ccw": {}, "switch": {}}
        for c in N:
            for times in concen:
                bi_ar.extend(times[c]["biases"])
                cw_ar.extend(times[c]["cw"])
                ccw_ar.extend(times[c]["ccw"])
                sw_ar.extend(times[c]["switch"])
            #set each key equal to (avg, std/sqrt(n)) pair
            temp["biases"][c] = (np.mean(bi_ar), np.std(bi_ar[:length])/np.sqrt(length))
            temp["cw"][c] = (np.mean(cw_ar), np.std(cw_ar[:length])/np.sqrt(length))
            temp["ccw"][c] = (np.mean(ccw_ar), np.std(ccw_ar[:length])/np.sqrt(length))
            temp["switch"][c] = (np.mean(sw_ar), np.std(sw_ar[:length])/np.sqrt(length))
        result[con] = temp
    return result


if __name__ == '__main__':
    fts = {}
    fts_time = {}
    fts2 = {}
    fts_time2 = {}
    for pre in conc:
        out_name = pre + "_" + "asp" + '.csv'
        out = open(out_name, 'a+')
        writer = csv.writer(out, delimiter=",")
        feats = {"biases":[], "cw": [], "ccw": [], "switch": []}
        fts_time[pre] = []
        feats2 = {"biases":[], "cw": [], "ccw": [], "switch": []}
        fts_time2[pre] = []
        for i in range(file_num):
            csv_name = path + pre + "_" + "asp" + str(i+1) + "_checked" + ".csv"
            csv_name2 =  path + pre + "_" + "leu" + str(i+1) + "_checked" + ".csv"
            if pre == "control":
                csv_name = path + pre + str(i+1) + "_checked"+ ".csv"
            data = np.loadtxt(csv_name, delimiter=",")
            data2 = np.loadtxt(csv_name2, delimiter=",")     	
            centers, status, traces = np.hsplit(data, np.array([2, 3]))
            centers2, status2, traces2 = np.hsplit(data2, np.array([2, 3]))
            features = compute_features_for_each_trace(status,traces, count)  # compute features using ALL traces from ONE concentration
            features2 = compute_features_for_each_trace(status2,traces2, count)  # compute features using ALL traces from ONE concentration
            # bias = [filename[:-1]] + result
            feats["biases"].extend(features["biases"])
            feats["cw"].extend(features["cw"])
            feats["ccw"].extend(features["ccw"])
            feats["switch"].extend(features["switch"])
            feats2["biases"].extend(features2["biases"])
            feats2["cw"].extend(features2["cw"])
            feats2["ccw"].extend(features2["ccw"])
            feats2["switch"].extend(features2["switch"])
            features_time = compute_features_over_time(status, traces)
            features_time2 = compute_features_over_time(status2, traces2)
            fts_time[pre].append(features_time)
            fts_time2[pre].append(features_time2)
            reformat = list(zip(features["biases"], features["cw"], features["ccw"], features["switch"]))
            reformat2 = list(zip(features2["biases"], features2["cw"], features2["ccw"], features2["switch"]))
            writer.writerows(reformat)
        fts[pre] = feats
        fts2[pre] = feats2
        out.close()
    combined_feats = combine_features_over_time(fts_time,30)
    combined_feats2 = combine_features_over_time(fts_time2,30)
    #graph_time(combined_feats)
    graph_conc(combined_feats,combined_feats2)

    #plt.show()
    # np.save("biases/" + concentration, result)

# To load:
# Note: Use [()], which allows us to load the dict() we saved as a .npy file.


# for t in trace:
# 	bias.append(compute_bias(t))
# out = np.asarray(bias)
# np.savetxt(out_name, bias, delimiter=",")
