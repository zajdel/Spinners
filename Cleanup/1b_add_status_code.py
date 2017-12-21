from utilities import *
#  Ex. python 1b_add_status_code.py 100nM_leu100n_1 [Centers + Trace in CSV]
# Script is used to add status codes/flags (-1: unverified, 0: verified - bad, 1: verified - good) to older data

fname = sys.argv[1]
dataname = fname + '.csv'
data = np.loadtxt(dataname, delimiter=",")
centers, trace = np.hsplit(data, 2)

status = np.zeros((data.shape[0], 1), dtype=np.int)
new_data = np.hstack((centers, status, trace))

np.savetxt(fname + ".csv", new_data, fmt=','.join(["%.4f"] * centers.shape[1] + ["%i"] + ["%.4f"] * trace.shape[1]))
