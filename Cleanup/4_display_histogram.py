# Calculate the bias over angular velocity, plot results
# input: string prefix(prefix of the set), int number of numbers from beginning to calculat bias over)

from utilities import *

import pandas as pd

csv = sys.argv[1] # csv to make histogram from
concentration = sys.argv[2]

# data = pd.read_csv(csv + '.csv', sep=',', header=None)
data = np.loadtxt(csv + '.csv', delimiter=',')

plt.hist(data, bins=20, range=(0, 1))
plt.ylabel('Frequency')
plt.xlabel('Bias')
plt.title('Bias of ' + concentration)

plt.show()
