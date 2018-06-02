# Import packages
from __future__ import division, unicode_literals  # , print_function
import argparse
import numpy as np
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import medfilt

mpl.rc('figure', figsize=(16, 10))
mpl.rc('image', cmap='gray')
