# Import packages
from __future__ import division, unicode_literals  # , print_function
import numpy as np
from math import degrees, radians, sin, cos, floor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import trackpy as tp
import scipy
import pims
from scipy import interpolate, signal
# import cv2
import time
import xml.etree.ElementTree
# import tifffile
from datetime import datetime
import metamorph_timestamps
from scipy.ndimage.filters import median_filter
from scipy.signal import medfilt
from scipy.stats import linregress
# from skimage.morphology import disk
# from skimage.filters import threshold_otsu, rank
import sys
import os
from PIL import Image

mpl.rc('figure', figsize=(16, 10))
mpl.rc('image', cmap='gray')

## Paths to our files. Update as data is recorded.

# HOW TO ADD A PREFIX TO ALL FILES IN A DIRECTORY:
# for f in * ; do mv "$f" "PREFIX_$f" ; done

## TODO: notice how tif files now have prefix!!!
### We will have to change how code SAVES and OPENS tifs/params/kymos/etc so that pipeline runs smoothly.

concentrations = ['100nM', '1uM', '10uM', '100uM', '1mM', 'MotMed']

# for Leucine
paths = {
    '1mM': [
        '1mM/1mM_leu1m_1.tif',
        '1mM/1mM_leu1m_2.tif',
        '1mM/1mM_leu1m_3.tif',
        '1mM/1mM_leu1m_4.tif',
        '1mM/1mM_leu1m_5.tif'
    ],
    '1uM': [
        '1uM/1uM_leu1u_1.tif',
        '1uM/1uM_leu1u_2.tif',
        '1uM/1uM_leu1u_3.tif',
        '1uM/1uM_leu1u_4.tif',
        '1uM/1uM_leu1u_5.tif'
    ],
    '10uM': [
        '10uM/10uM_leu10u_1.tif',
        '10uM/10uM_leu10u_2.tif',
        '10uM/10uM_leu10u_3.tif',
        '10uM/10uM_leu10u_4.tif',
        '10uM/10uM_leu10u_5.tif'
    ],
    '100nM': [
        '100nM/100nM_leu100n_1.tif',
        '100nM/100nM_leu100n_2.tif',
        '100nM/100nM_leu100n_3.tif',
        '100nM/100nM_leu100n_4.tif',
        '100nM/100nM_leu100n_5.tif'
    ],
    '100uM': [
        '100uM/100uM_leu100u_1.tif',
        '100uM/100uM_leu100u_2.tif',
        '100uM/100uM_leu100u_3.tif',
        '100uM/100uM_leu100u_4.tif',
        '100uM/100uM_leu100u_5.tif'
    ],
    'MotMed': [
        'MotMed/MotMed_mm_1.tif',
        'MotMed/MotMed_mm_2.tif',
        'MotMed/MotMed_mm_3.tif',
        'MotMed/MotMed_mm_4.tif',
        'MotMed/MotMed_mm_5.tif'
    ]
}

angs = []
# Set this!
for i in np.linspace(0, 360, 72):  # select num. intervals per circle.
    angs.append(i)


def create_directories(list_of_directories):
    map(os.mkdir, filter(lambda dir: not os.path.isdir(dir), list_of_directories))


def get_video_path(args):
    # Arg1 = folder w/ concentration. Arg2 = stream number in that folder (see paths dict)
    path = paths[args[1]][int(args[2])]
    videos_dir = unicode.join('/', unicode.split(path, '/')[:-1])
    video_name = unicode.split(unicode.split(path, '/')[-1], '.')[0]
    return '/' + str(video_name), str(videos_dir)


# Path to tif stack

def convert_to_8bit(image):
    # from int32
    im = image.astype(np.float64)
    im2 = (im - im.min())
    im2 = im2 * 255 / im2.max()
    im2 = np.uint8(im2)
    return im2


def press(event):
    if event.key == 'n':
        plt.close()
    if event.key == 'escape':
        print('Exiting!')
        exit(0)
    else:
        plt.close()


def show(image):
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    ax.set_title('Is this a good choice of parameters? If yes, press \'y\', else press ESC.')
    plt.imshow(image)
    plt.show()


def create_rot_mask(orig_mask, deg):
    rotmymask = []
    ang = radians(deg)
    for i in orig_mask:
        rotMatrix = np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
        rotmymask.append(list(rotMatrix.dot(i)))
    return np.array(rotmymask)


def adj_ctr_mask(mask, deg, cell_num, centers):
    adj_mask = []
    rotated_mask_about_00 = create_rot_mask(mask, deg)
    for mask_boundary in rotated_mask_about_00:
        # Move the origin of the mask to the center of the bacterium.
        # (Mask is originally at origin (0,0).)
        adj_mask.append(list(mask_boundary + centers[cell_num]))
    return np.array(adj_mask)


def angle_kym(ang, cell_num, frames, mymask, centers, Show=False):
    ang_ar = []
    t0 = time.time()
    for i in range(frames.shape[0]):
        frame = frames[i].astype(np.uint8)
        box = np.int64(adj_ctr_mask(mymask, ang, cell_num, centers))  # this is the box rotated at ang deg.
        cv2.drawContours(frame, [box], 0, (0, 0, 0), 1)
        if Show:
            if i == 0 and ang == 0:  # shows the windows on top of 75th frame
                show(frame)  # only showing filter do a 360 on first frame.
        mask = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(mask, [box], 0, 1, -1)  # cv2.drawContours(mask,[box],0,255,-1)
        ang_ar.append(cv2.mean(frame, mask=mask)[0])
    return ang_ar  # for each frame, computes the pixel average for a window rot'd at a given theta


def invert_colors(kymograph):
    kymograph -= kymograph.min()
    kymograph /= kymograph.max()
    return (1 - kymograph) * 255


def build_kymograph(cell_num, frames, mask, selected_points, Show=False):
    kymograph = []
    for ang in angs:
        kymograph.append(angle_kym(ang, cell_num, frames, mask, selected_points, Show=Show))
    return np.array(kymograph)


def process_kymograph(kymograph):
    # Appends kymograph to kymograph_images and returns
    # computed trace from that saved kymograph image.
    trace = []
    # Remove background by subtracting median of each vertical column from itself.
    no_background = []
    for i in range(kymograph.shape[1]):
        # black cells on white background (switch signs if reversed)
        no_background.append(kymograph[:, i] - np.median(kymograph, 1))
    no_background = np.array(no_background).T

    # Change negative values to 0.
    clipped_background = no_background.clip(min=0)
    return clipped_background  # the processed kymograph


def process_kymograph2(kymograph):
    filtered = []
    rmeans = np.mean(kymograph, axis=1)
    for i in range(kymograph.shape[0]):
        # first, subtract each horizontal row's mean from itself to filter the kymograph
        krow = kymograph[i, :]
        krow[:] = [k - rmeans[i] for k in kymograph[i, :]]
        filtered.append(krow)

    filtered = np.array(filtered)

    return filtered


def compute_trace(processed_kymograph):
    # Extract 1D signal using LA trick.
    eps = 1e-12

    def exp_func(x):
        return np.dot(np.arange(len(x)), np.power(x, 10)) / (eps + np.sum(np.power(x, 10)))

    weighted_sum = np.apply_along_axis(exp_func, 0, processed_kymograph)

    # Derivative of 1D signal. Continuous parts show angular velocity of cell (not 100% sure on this.)
    # conv = np.convolve([-1.,1],weighted_sum, mode='full')[:-1]
    # median_filtered_conv = median_filter(conv, 7) #pick window size based on result. second arg and odd number.
    trace = weighted_sum
    return trace


def compute_trace2(processed_kymograph):
    # extract 1D signal by going through the kymograph column by column and detecting minima
    trace = []

    img = np.array(processed_kymograph, dtype=np.uint8)
    edges = cv2.Canny(img, 100, 255, apertureSize = 3)

    edges += img

    kernel = np.zeros((1, 1), np.uint8)
    edges = cv2.erode(img, kernel, iterations=1)

    removePepperNoise(edges)

    plt.imshow(edges)

    potential_points = {}
    for i in range(edges.shape[1]):
        potential_points[i] = [j for j in range(edges.shape[0]) if edges[j, i] > 200]
        if potential_points[i]:
            median_index = np.argsort(potential_points[i])[len(potential_points[i])//2]
            if len(trace) > 0:
                # TODO: Is this necessary?
                # If previous point was in middle 2/3 of the graph, prevent jumping more than 5/8 of kymograph height (should reduce random jumps)
                while trace[-1][1] > 1/6 * edges.shape[1] and potential_points[i][median_index] - trace[-1][1] > 5/8 * edges.shape[1] and median_index < len(potential_points[i]) - 1:
                    median_index += 1
                while trace[-1][1] < 5/6 * edges.shape[1] and potential_points[i][median_index] - trace[-1][1] < 5/8 * edges.shape[1] and median_index > 0:
                    median_index -= 1
            trace.append((i, potential_points[i][median_index]))
    for i in range(2, len(trace) - 2):
        y = trace[i][1]
        # If maxima/minima, extend up/down as far as possible
        if trace[i - 1][1] <= y >= trace[i + 1][1] and trace[i - 2][1] <= y >= trace[i + 2][1]:
            while y < edges.shape[0] - 1 and y in potential_points[trace[i][0]]:
                y += 1
            trace[i] = (trace[i][0], y)
        elif trace[i - 1][1] >= y <= trace[i + 1][1] and trace[i - 2][1] >= y <= trace[i + 2][1]:
            while y > 0 and y in potential_points[trace[i][0]]:
                y -= 1
            trace[i] = (trace[i][0], y)
    trace = [point[1] for point in trace]

    return np.asarray(trace)


def removePepperNoise(img):
    for y in range(1, len(img) - 1):
        for x in range(1, len(img[0]) - 1):
            is_black = [[False for j in range(3)] for i in range(3)]
            for xdiff in range(-1, 2):
                for ydiff in range(-1, 2):
                    if img[y + ydiff, x + xdiff] < 50:
                        is_black[ydiff + 1][xdiff + 1] = True
            # If the 3x3 square surrounding (x, y) has more than 7 black points, set (x, y) to black
            if sum([sum(row) for row in is_black]) >= 7:
                img[y][x] = 0

def smooth(a, WSZ):
    # smoothing function that emulates MATLAB smooth weighted average over windowsize = WSZ, odd integer
    # shared by Divakar on stack overflow https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    # special cases for the edge cases are handled the same way as in MATLAB
    out0 = np.convolve(a, np.ones(WSZ), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def plot_kymograph(kymograph):
    plt.title('Kymograph', fontsize=20)
    plt.ylabel('Angles', fontsize=20)
    plt.xlabel('Frame', fontsize=20)
    plt.imshow(kymograph)


# Returns the indices (frame locations) of when the sign switches.
def sign_switch(oneDarray):
    inds = []
    for ind in range(len(oneDarray) - 1):
        if (oneDarray[ind] < 0 and oneDarray[ind + 1] > 0) or (oneDarray[ind] > 0 and oneDarray[ind + 1] < 0):
            inds.append(ind)
    return np.array(inds)


def get_date(path_to_tif):
    # Parse timestamp strings from the XML 
    # of the Metamorph metadata.
    # Returns an equivalent numpy array (in miliseconds).
    scope_times = []
    scope_tif_file = tifffile.TiffFile(path_to_tif)
    for t in range(len(scope_tif_file)):
        metadata = scope_tif_file[t].image_description
        root = xml.etree.ElementTree.fromstring(metadata).find('PlaneInfo')
        for neighbor in root:
            if neighbor.attrib['type'] == 'time':
                if neighbor.attrib['id'] == 'acquisition-time-local':
                    first = neighbor.attrib['value']
                    return first


def print_to_csv(data, fname, meta, tifname):
    acquisition_time = get_date(tifname)
    with open(fname + ".csv", "wb") as f:
        f.write("Rotation Data (1D Kymograph Signal [radians vs time])," + '\n')
        f.write(tifname + ',\n')
        f.write(acquisition_time + ',\n,\n')
        header = "time (ms),"
        for i in range(len(data)):
            header += "cell" + str(i) + ","
        header += "\n"
        f.write(header)
        T = len(data[0])
        print "len of run " + str(T)
        Cells = len(data)
        print "len data " + str(Cells)
        for t in range(T):
            new_row_in_file = str(meta[t]) + ','
            for cell_data in range(Cells - 1):
                new_row_in_file += (str(data[cell_data][t]) + ',')
            new_row_in_file += (str(data[Cells - 1][t]))
            f.write(new_row_in_file + '\n')
    f.close()


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))

# Goal:
# * Filter particles better
# * Organize code
# * Read data from .csv
