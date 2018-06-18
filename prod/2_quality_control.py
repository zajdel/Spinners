from utilities import *
#  Ex. python 2_quality_control.py 1mM_asp1
from math import sin, cos
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

parser = argparse.ArgumentParser(description="Perform quality control on generated traces and/or determine thresholds for switches.")
parser.add_argument("source", help="source file [CSV]")
parser.add_argument("frames", nargs="?", help="frames [TIF]")
parser.add_argument("-d", "--dest", help="destination file [CSV]")
parser.add_argument("-t", "--type", type=int, choices=[0, 1, 2, 3], default=2,
                    help="""type of quality control:    0 - show trace,
                                                        1 - show reconstructed cells overlaid on actual video,
                                                        2 - show velocity graph processed from trace with manual thresholding,
                                                        3 - show velocity graph processed from trace with automated threhsolding"""
                    )
args = parser.parse_args()

if args.frames is None and args.type == 2:
    parser.error("frames required when type is 1")

if args.frames:
    tif_name = args.frames + '.tif'
    raw_frames = pims.TiffStack(tif_name, as_grey=False)
    frames = np.array(raw_frames, dtype=np.uint8)

data_name = args.source + '.csv'
data = np.loadtxt(data_name, delimiter=",", ndmin=2)
num_cells = data.shape[0]
# Status code:  (-1, -1): unverified,
#               (0, 0): verified - bad,
#               (1, 1): verified - good,
#               (x, y): verified - good with lower threshold x and upper threshold y
centers, status, trace = np.hsplit(data, np.array([2, 4]))
# for backward compatibility when status did not include threshold
if status.shape[1] == 1:
    status = np.hstack((status, status))

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^    Helper Functions    *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^


def moving_average(values, window=8):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def hysteresis_threshold(tr, thresh_high, thresh_low):
    direction = np.zeros(len(tr))
    prev_direction = 1

    for k in range(0, len(tr)):
        if tr[k] < thresh_low:
            direction[k] = -1
            prev_direction = -1
        elif tr[k] > thresh_high:
            direction[k] = 1
            prev_direction = 1
        else:
            direction[k] = prev_direction

    return direction


def mad(arr):
    """Computes the median absolute deviation (MAD) of an input array."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def threshold(y, lag, thresh_high, thresh_low, influence):
    """Dynamically thresholds input data array.

    Uses median, median absolute deviation, and asymmetric treatment of up and down signals to threshold input.
    Essentially a parametrized peak-trough detection algorithm.
    Args:
        y: input data
        lag: lag of moving window used to smooth data
        thresh_high: number of median absolute deviations data point differs above median to threshold up
        thresh_low: number of median absolute deviations data point differs below median to threshold down
        influence: influence (between 0 and 1) of new signals on median and median absolute deviation

    Returns:
        signals, median filter, median absolute deviation filter

    adapted from https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887
    """
    signals = np.zeros(len(y))
    filtered_y = np.array(y)
    med_filter = [0] * len(y)
    mad_filter = [0] * len(y)
    med_filter[lag - 1] = np.median(y[0:lag])
    mad_filter[lag - 1] = mad(y[0:lag])
    # for first lag signals, do not have prior data, so evaluate based on sign
    for i in range(0, lag):
        signals[i] = 1 if y[i] > 0 else -1
    for i in range(lag, len(y)):
        # UP: (y[i] - med_filter[i - 1] > threshold AND difference > 0.1) OR y[i] > 0.5 + min of 4 previous y[i]
        if (y[i] - med_filter[i - 1] > thresh_high * mad_filter[i - 1] and y[i] - med_filter[i - 1] > 0.3) or y[i] - min(y[i - 4:i]) > 0.4:
            signals[i] = 1

            filtered_y[i] = influence * y[i] + (1 - influence) * filtered_y[i - 1]
            med_filter[i] = np.median(filtered_y[(i-lag):i])
            mad_filter[i] = mad(filtered_y[(i-lag):i])
        # DOWN: (y[i] - med_filter[i - 1] < threshold AND difference < -0.1) OR y[i] < -0.5 + max of 4 previous y[i]
        elif (y[i] - med_filter[i - 1] < -thresh_low * mad_filter[i - 1] and y[i] - med_filter[i - 1] < -0.3) or y[i] - max(y[i - 4:i]) < -0.4:
            signals[i] = -1

            filtered_y[i] = influence * y[i] + (1 - influence) * filtered_y[i - 1]
            med_filter[i] = np.median(filtered_y[(i - lag):i])
            mad_filter[i] = mad(filtered_y[(i - lag):i])
        # NO CHANGE: signals[i] = signals[i - 1]
        else:
            signals[i] = signals[i - 1]
            filtered_y[i] = y[i]
            med_filter[i] = np.median(filtered_y[(i - lag):i])
            mad_filter[i] = mad(filtered_y[(i - lag):i])

    return np.asarray(signals), np.asarray(med_filter), np.asarray(mad_filter)

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^    QC Type 1   *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^


def animate_frames_overlay(counter):
    num_subplots = 9
    num_frames = data.shape[1]
    radius = 8

    fig, ax = plt.subplots(3, 3)
    animations = []
    cells = []
    time_text = fig.text(0.147, 0.92, '', horizontalalignment='left', verticalalignment='top')

    def init():
        for i in range(num_subplots):
            center_x, center_y = centers[counter].astype(np.int)
            ax[i % 3, i // 3].set_xlim(center_x - 8, center_x + 8)
            ax[i % 3, i // 3].set_ylim(center_y - 8, center_y + 8)
            animations.append(ax[i % 3, i // 3].imshow(frames[num_frames / num_subplots * i, center_y - 8:center_y + 8, center_x - 8:center_x + 8], aspect='equal', extent=[center_x - 8, center_x + 8, center_y - 8, center_y + 8]))
            x = [center_x, center_x + radius * cos(trace[counter, num_frames / num_subplots * i])]
            y = [center_y, center_y + radius * sin(trace[counter, num_frames / num_subplots * i])]
            cells.append(ax[i % 3, i // 3].plot(x, y)[0])
        time_text.set_text('Frame 0 of %d' % (num_frames / num_subplots))

    def animate(frame):
        for i in range(num_subplots):
            center_x, center_y = centers[counter].astype(np.int)
            animations[i] = ax[i % 3, i // 3].imshow(frames[(num_frames / num_subplots * i) + frame % (num_frames / num_subplots), center_y - 8:center_y + 8, center_x - 8:center_x + 8], aspect='equal', extent=[center_x - 8, center_x + 8, center_y - 8, center_y + 8])
            # angle is calculated with respect to numpy array, i.e. arctan(x/y), so we correct with
            # x = center_x + sin(theta) and y = center_y + cos(theta)
            x = [center_x, center_x + radius * cos(trace[counter, (num_frames / num_subplots * i) + frame % (num_frames / num_subplots)])]
            y = [center_y, center_y + radius * sin(trace[counter, (num_frames / num_subplots * i) + frame % (num_frames / num_subplots)])]
            cells[i].set_data(x, y)
        time_text.set_text('Frame %d of %d' % (frame % (num_frames / num_subplots), num_frames / num_subplots - 1))

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=80)
    plt.show()

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^    QC Types 2 and 3    *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^


def show_trace(counter):
    fig, ax = plt.subplots()

    def record_yes(event):
        status[counter] = thresh
        plt.close()

    def record_no(event):
        status[counter] = [0, 0]
        plt.close()

    unwrapped = np.unwrap(np.asarray(trace[counter]))
    ma_trace = moving_average(unwrapped, 8)  # 8*1/32 fps ~ 250 ms moving average filter window
    velocity = np.convolve([-0.5, 0.0, 0.5], ma_trace, mode='valid')

    plt.xlabel('Frame', fontsize=20)
    plt.ylabel('Angle', fontsize=20)
    plt.title('Trace ({0}, {1}): {2} of {3}'.format(centers[counter][0], centers[counter][1], counter + 1, num_cells), fontsize=20)

    if args.type == 0:
        plt.plot(unwrapped, 'r-', lw=1)
    elif args.type == 2:
        thresh = [-1, -1]

        vel_range = np.abs(np.nanmax(velocity) - np.nanmin(velocity))
        # if thresh has previously been set
        if not np.array_equal(status[counter], [0, 0]) or not np.array_equal(status[counter], [-1, -1]):
            thresh_high, thresh_low = status[counter]
        else:
            thresh_high = np.nanmax(velocity) - vel_range * 0.50
            thresh_low = np.nanmin(velocity) + vel_range * 0.25
        thresh = [thresh_high, thresh_low]
        d = hysteresis_threshold(velocity, *thresh)

        def update_sensitivity(val):
            thresh_high = s_high_thresh.val
            thresh_low = s_low_thresh.val
            thresh[0], thresh[1] = thresh_high, thresh_low
            dd = hysteresis_threshold(velocity, *thresh)
            f1.set_ydata(dd)
            f2.set_ydata((thresh_high, thresh_high))
            f3.set_ydata((thresh_low, thresh_low))
            fig.canvas.draw_idle()

        s_high_thresh = Slider(fig.add_axes([0.20, 0.15, 0.65, 0.03]), 'Upper Threshold', -2.0, 2.0, valinit=thresh_high)
        s_low_thresh = Slider(fig.add_axes([0.20, 0.1, 0.65, 0.03]), 'Lower Threshold', -2.0, 2.0, valinit=thresh_low)
        s_high_thresh.on_changed(update_sensitivity)
        s_low_thresh.on_changed(update_sensitivity)
        f1, = plt.plot(range(0, len(velocity)), d, 'b-')
        f2, = plt.plot((0, len(velocity)), (thresh_high, thresh_high), 'g', lw=3)
        f3, = plt.plot((0, len(velocity)), (thresh_low, thresh_low), 'k', lw=3)
    elif args.type == 3:
        thresh = [1, 1]  # no threshold in type 3, so set default to [1, 1]

        signals, med_filter, mad_filter = threshold(velocity, 4, 10, 3.5, 0.2)
        plt.plot(range(0, len(velocity)), velocity, 'r-')
        # FOR DEBUGGING USE, DISPLAY TRIGGERS FOR SWITCH
        # plt.plot(range(0, len(velocity)), med_filter, 'k', lw=0.5)
        # plt.plot(range(0, len(velocity)), med_filter + 10 * mad_filter, 'g', lw=0.5)
        # plt.plot(range(0, len(velocity)), med_filter - 3.5 * mad_filter, 'g', lw=0.5)
        plt.plot(range(0, len(velocity)), signals, 'b-')
        plt.ylim((-2, 2))
        plt.xlim((0, 1875))

    plt.grid(True, which='both')

    b_yes = Button(fig.add_axes([0.65, 0.9, 0.1, 0.03]), 'Yes')
    b_no = Button(fig.add_axes([0.80, 0.9, 0.1, 0.03]), 'No')
    b_yes.on_clicked(record_yes)
    b_no.on_clicked(record_no)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()


for i in range(num_cells):
    if args.type == 1:
        animate_frames_overlay(i)
    else:
        show_trace(i)

# output: center_x, center_y, status/upper threshold, status/lower threshold, trace
np.savetxt(args.source + "_checked.csv", np.hstack((centers, status, trace)), fmt=','.join(["%.4f"] * 2 + ["%.4f"] * 2 + ["%.4f"] * trace.shape[1]))
