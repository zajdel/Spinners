# Calculate the bias over angular velocity, plot results
# input: name of file, int number of numbers from beginning to calculate bias over)

import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

path = sys.argv[1]
conc = sys.argv[2].split(',')
chem = sys.argv[3]
# concentration = sys.argv[1]
file_num = int(sys.argv[4])
count = int(sys.argv[5])

interval_bias = lambda s: np.sum((-np.array(s) + 1) / 2) / len(s)  # CCW / (CCW + CW); s is interval over which to compute bias, s is signs of rotation direction. NOTE: correct if cw is positive, ccw is negative.

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


def compute_features_for_each_trace(status,traces):
    features = {"biases":[],"cw":[], "ccw":[], "switch":[]}
	# remove traces that have a status of 0 (rejected)
    for s,trace in zip(status,traces):
	    if (s>0):
             # unwrap and smooth the trace before computing the bias
             results = compute_features(trace, count, s)  # Set first and frames so that it's about 10 s of data.
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
    ylimit = [[0.5,1.0],[0,30],[0,30],[0,100]]
    ylabel = ["Bias","seconds","seconds","number"]
    c_vals = {"1m": -3,"100u": -4,"10u": -5,"1u": -6,"100n": -7, "control": -9}
    for i in range(0,4):
        plt.figure(i)
        plt.xlabel('Concentrations')
        plt.title(label[i])
        plt.ylim(ylimit[i])
        plt.ylabel(ylabel[i])
        plt.xlim(-10,0)
        c = 0
        for avg, std in plots[i]:
            plt.errorbar(c_vals[conc[c]], avg, std,fmt='--o',ecolor='b',c='b')
            c += 1
    plt.show()


if __name__ == '__main__':
    fts = {}
    for pre in conc:
        out_name = pre + "_" + chem + '.csv'
        out = open(out_name, 'a+')
        writer = csv.writer(out, delimiter=",")
        feats = {"biases":[], "cw": [], "ccw": [], "switch": []}
        for i in range(file_num):
            csv_name = path + pre + "_" + chem + str(i+1) + "_checked" + ".csv"
            if pre == "control":
                csv_name = path + pre + str(i+1) + "_checked.csv"
            data = np.loadtxt(csv_name, delimiter=",")
            centers, status, traces = np.hsplit(data, np.array([2, 3]))
            features = compute_features_for_each_trace(status,traces)  # compute features using ALL traces from ONE concentration
            # bias = [filename[:-1]] + result
            # print(features)
            feats["biases"].extend(features["biases"])
            feats["cw"].extend(features["cw"])
            feats["ccw"].extend(features["ccw"])
            feats["switch"].extend(features["switch"])
            reformat = list(zip(features["biases"], features["cw"], features["ccw"], features["switch"]))
            # print(reformat)
            writer.writerows(reformat)
        fts[pre] = feats
        out.close()
    graph(fts)
    # np.save("biases/" + concentration, result)

# To load:
# Note: Use [()], which allows us to load the dict() we saved as a .npy file.


# for t in trace:
# 	bias.append(compute_bias(t))
# out = np.asarray(bias)
# np.savetxt(out_name, bias, delimiter=",")
