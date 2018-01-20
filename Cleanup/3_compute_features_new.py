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


def compute_bias(trace, window=30):
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
    bias = interval_bias(interval)
    return bias


def compute_features_for_each_trace(traces):
    results = []
    for trace in traces:
        # unwrap and smooth the trace before computing the bias
        # trace =smooth(np.unwrap(trace*np.pi/180.0),11)*180/np.pi;
        biases = compute_bias(trace, count)  # Set first and frames so that it's about 10 s of data.
        results.append(biases)
    return results


if __name__ == '__main__':
    out = open(out_name, 'a+')
    # data = np.loadtxt("traces/" + csv_name, delimiter=",")
    data = np.loadtxt(csv_name, delimiter=",")
    centers, status, traces = np.hsplit(data, np.array([2, 3]))
    result = compute_features_for_each_trace(traces)  # compute features using ALL traces from ONE concentration
    # bias = [filename[:-1]] + result
    print(result)
    np.savetxt(out, result, fmt='%5f', delimiter=",")
    out.close()
# np.save("biases/" + concentration, result)

# To load:
# Note: Use [()], which allows us to load the dict() we saved as a .npy file.


# for t in trace:
# 	bias.append(compute_bias(t))
# out = np.asarray(bias)
# np.savetxt(out_name, bias, delimiter=",")
