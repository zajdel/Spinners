from utilities import *
#  Ex. python 1c_edit_status_code.py 100nM_leu100n_1 [Centers + Trace in CSV] 0 [New status code]
# Script is used to edit status codes/flags (-1: unverified, 0: verified - bad, 1: verified - good)

fname = sys.argv[1]
new_status = sys.argv[2]
dataname = fname + '.csv'
data = np.loadtxt(dataname, delimiter=",")
centers, status, trace = np.hsplit(data, np.array([2, 3]))

new_status = np.full((data.shape[0], 1), new_status, dtype=np.int)
new_data = np.hstack((centers, new_status, trace))

np.savetxt(fname + ".csv", new_data, fmt=','.join(["%.4f"] * centers.shape[1] + ["%i"] + ["%.4f"] * trace.shape[1]))
