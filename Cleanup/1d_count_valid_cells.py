from utilities import *
#  Ex. python 1c_edit_status_code.py 100nM_leu100n_1 [Centers + Trace in CSV] 0 [New status code]
# Script is used to edit status codes/flags (-1: unverified, 0: verified - bad, 1: verified - good)

fname = sys.argv[1]
dataname = fname + '_checked.csv'
data = np.loadtxt(dataname, delimiter=",")
centers, status, trace = np.hsplit(data, np.array([2, 4]))

count = 0
for s in status:
    if not np.array_equal(s, [0, 0]):
        count += 1

print(count)
