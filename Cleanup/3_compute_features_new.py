# Calculate the bias over angular velocity, plot results
# input: name of file, int number of numbers from beginning to calculate bias over)

import sys
import numpy as np

filename = sys.argv[1]
# concentration = sys.argv[1]
count = int(sys.argv[2])
out = sys.argv[3]

csv_name = filename + '.csv'
out_name = out + '.csv'

interval_bias = lambda s: np.sum((-np.array(s) + 1) / 2) / len(s)  # CCW / (CCW + CW); s is interval over which to compute bias, s is signs of rotation direction. NOTE: correct if cw is positive, ccw is negative.


def compute_features(trace, window=900):
    # Input: trace is single bacterium raw time series
    #       window is size of window over which we (locally) compute bias.
    # Output: time series of bias at all overlapping intervals of window-length.  (len: len(trace) - window)

    bias = []

    # 1. Derivative of 1D signal. (Angular velocity) Use to get signs, which tell us CCW or CW.
    conv = np.convolve([-1., 1], trace, mode='full')
    # Optionally:
    #       median_filtered_conv = median_filter(conv, 7) # pick window size based on result. second arg is odd number.

    # 2. Get direction of rotation (CCW & CW)
    signs = np.sign(conv)  # Positive values correspond to cw rotation. Negative = ccw rotation.

    # Optionally:
    # filtered_signs = median_filter(signs, 9) # pick window size based on result. second arg is odd number.
    # should probably use some hysteresis thresholding instead, try it!

    # 3. Compute bias over each window-length interval
    # no sliding window as here:
    # use first 'first' frames to compute bias
    interval = signs[:window]

    features = {}
    total = 0
    avg_cw = (0,0)
    avg_ccw = (0,0)
    switch = 0
    count = 0
    prev_direction = interval[0]
    for i in interval:
        total += (-i + 1) / 2
        if i == prev_direction:
            count += 1
        else:
            switch += 1
            if prev_direction == -1.0:
                temp1 = avg_ccw[1] + 1
                temp0 = (avg_ccw[0]*avg_ccw[1] + count) / temp1
                avg_ccw = (temp0, temp1)
            elif prev_direction == 1.0:
                temp1 = avg_cw[1] + 1
                temp0 = (avg_cw[0]*avg_cw[1] + count) / temp1
                avg_cw = (temp0, temp1)
            count = 1
            prev_direction = i
    features["bias"] = total / 900
    features["cw"] = avg_cw[0]
    features["ccw"] = avg_ccw[0]
    features["switch"] = switch
    return features


def compute_features_for_each_trace(traces):
    features = {"biases":[],"cw":[], "ccw":[], "switch":[]}
    for trace in traces:
        # unwrap and smooth the trace before computing the bias
        # trace =smooth(np.unwrap(trace*np.pi/180.0),11)*180/np.pi;
        results = compute_features(trace, count)  # Set first and frames so that it's about 10 s of data.
        features["biases"].append(results["bias"])
        features["cw"].append(results["cw"])
        features["ccw"].append(results["ccw"])
        features["switch"].append(results["switch"])
    return features


if __name__ == '__main__':
    out = open(out_name, 'a+')
    writer = csv.writer(out, delimiter=",")
    # data = np.loadtxt("traces/" + csv_name, delimiter=",")
    data = np.loadtxt(csv_name, delimiter=",")
    centers, status, traces = np.hsplit(data, np.array([2, 3]))
    features = compute_features_for_each_trace(traces)  # compute features using ALL traces from ONE concentration
    # bias = [filename[:-1]] + result
    #print(features)
    reformat = list(zip(features["biases"], features["cw"], features["ccw"], features["switch"]))
    #print(reformat)
    writer.writerows(reformat)
    out.close()
    # np.save("biases/" + concentration, result)

# To load:
# Note: Use [()], which allows us to load the dict() we saved as a .npy file.


# for t in trace:
# 	bias.append(compute_bias(t))
# out = np.asarray(bias)
# np.savetxt(out_name, bias, delimiter=",")
