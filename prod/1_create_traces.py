from __future__ import division
from utilities import *
#  Ex. python 1_create_traces.py 100u_leu1 [TIF]
import pims
from PIL import Image
from Queue import Queue

parser = argparse.ArgumentParser(description="Create traces of cells identified in a TIF and output to CSV.")
parser.add_argument("source", help="source file [TIF]")
parser.add_argument("dest", nargs="?", help="destination file [CSV]")
parser.add_argument("-v", "--verbose", help="verbose output: display trace and speed plots for every cell", action="store_true")
args = parser.parse_args()

tif_name = args.source + '.tif'
raw_frames = pims.TiffStack(tif_name, as_grey=False)
frames = np.array(raw_frames, dtype=np.uint8)

# use and/or overwrite existing file
centers = []
if args.dest:
    csv_name = args.dest + '.csv'
    data = np.loadtxt(csv_name, delimiter=",")
    centers, _, _ = np.hsplit(data, np.array([2, 3]))

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^       Getting Mean Image       *^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

mean = np.mean(frames[0:500], axis=0)

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^     Overlay Mean on Frames     *^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

overlay = Image.fromarray(mean).convert('RGB')
new_frames = []
for frame in frames:
    frame = Image.fromarray(frame).convert('RGB')
    new_frames.append(np.asarray(Image.blend(frame, overlay, 0.8)))
frameview = new_frames

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^    Show Frames for Center Selection    *^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

# need these for cycling through cells
N = 50
show_frames = frameview[0:N]

fig, ax = plt.subplots()
im = ax.imshow(show_frames[0], aspect='equal')

# if using existing CSV, show previously selected centers
for center in centers:
    rect = patches.Rectangle((center[0] - 0.5, center[1] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# convert centers array to list of tuples and create a set from it
selected_points = set(tuple(map(tuple, centers)))


# Error handling (find minimum intensity in mean in 5x5 area around selected center)
def on_press(event):
    if event.xdata and event.ydata:
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print('You pressed {0} at ({1}, {2}) with mean value of {3}.'.format(event.button, x, y, mean[y, x]))

        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        roi = [(i, j) for i in range(x - 2, x + 3) for j in range(y - 2, y + 3) if 0 < i < mean.shape[1] and 0 < j < mean.shape[0]]
        min_intensity = min([mean[p[::-1]] for p in roi])  # p[::-1] reverses tuple
        correct_x, correct_y = np.mean([p for p in roi if mean[p[::-1]] == min_intensity], axis=0)
        selected_points.add(
            (correct_x, correct_y)  # use set to avoid duplicates being stored. (use tuple because can hash.)
        )
        if correct_x != x or correct_y != y:
            print('Corrected green box at ({0}, {1})'.format(correct_x, correct_y))
            area = patches.Rectangle((x - 2.5, y - 2.5), 5, 5, linewidth=0.5, edgecolor='r', facecolor='none')
            correction = patches.Rectangle((correct_x - 0.5, correct_y - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
            ax.add_patch(area)
            ax.add_patch(correction)


fig.canvas.mpl_connect('button_press_event', on_press)


def init():
    im.set_data(show_frames[0])


def animate(i):
    im.set_data(show_frames[i % N])
    return im


anim = animation.FuncAnimation(fig, animate, init_func=init, interval=100)
plt.show()

num_selected_points = len(selected_points)

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^  Post-Processing: Calculate Angle and Generate Traces  *^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

num_frames = len(frames)


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))


# use furthest point from center to calculate angle
def find_furthest_points(center, frame):
    # get all points connected to center ("cell"), find furthest points
    nearest_pixel_to_center = (int(round(center[0])), int(round(center[1])))
    fringe = Queue()
    fringe.put(nearest_pixel_to_center)
    cell = set()
    cell.add(nearest_pixel_to_center)
    marked = set()
    marked.add(nearest_pixel_to_center)

    # modified breadth-first search
    while not fringe.empty():
        p = fringe.get()
        p1 = (p[0] - 1, p[1])
        p2 = (p[0] + 1, p[1])
        p3 = (p[0], p[1] - 1)
        p4 = (p[0], p[1] + 1)
        p5 = (p[0] - 1, p[1] - 1)
        p6 = (p[0] - 1, p[1] + 1)
        p7 = (p[0] + 1, p[1] - 1)
        p8 = (p[0] + 1, p[1] + 1)
        for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
            if (p not in marked
                    and 0 <= p[0] < len(frames[frame][1])
                    and 0 <= p[1] < len(frames[frame][0])
                    and euclidean_distance(center, p) <= 8
                    and frames[frame][p[1], p[0]] == 0):
                marked.add(p)
                cell.add(p)
                fringe.put(p)

    cell = list(cell)
    max_dist = max([euclidean_distance(p, center) for p in cell])
    return [p for p in cell if euclidean_distance(p, center) == max_dist]


wrapped_traces = []
for center in selected_points:
    ellipses = []
    trace = []
    for i in range(num_frames):
        furthest_points = find_furthest_points((center[0], center[1]), i)
        if len(furthest_points) > 1:
            if len(trace):
                # take point whose angle is closest to previous angle
                furthest_point = min(furthest_points, key=lambda x: abs(trace[i - 1] % (2 * np.pi) - np.arctan2(x[0] - center[0], center[1] - x[1]) % (2 * np.pi)))
            else:
                # TODO: what to do if first frame is ambiguous
                furthest_point = furthest_points[0]
        else:
            furthest_point = furthest_points[0]
        # define angle to increase positively clockwise
        ang = np.arctan2(center[1] - furthest_point[1], furthest_point[0] - center[0])
        trace.append(ang)

    # add wrapped trace to CSV output
    # prepend center_x, center_y, -1 (status unverified)
    wrapped_traces.append(np.append([center[0], center[1]], np.append([-1, -1], trace)))

    if args.verbose:
        # unwrap trace and apply 1D median filter (default kernel size 3)
        unwrapped = medfilt(np.unwrap(np.asarray(trace[2:])))

        plt.xlabel('Frame', fontsize=20)
        plt.ylabel('Angle', fontsize=20)
        plt.title('Trace ({0}, {1})'.format(center[0], center[1]), fontsize=20)
        plt.plot(unwrapped, 'r-', lw=1)
        plt.grid(True, which='both')
        plt.show()

        # Calculate speed from trace
        speed = []
        for i in range(unwrapped.shape[0]):
            indices = []
            angs = []
            for j in range(i - 1, i + 2):
                if 0 <= j < unwrapped.shape[0]:
                    indices.append(j)
                    angs.append(unwrapped[j])
            slope = linregress(indices, angs)[0]
            speed.append(slope)

        plt.xlabel('Frame', fontsize=20)
        plt.ylabel('Speed', fontsize=20)
        plt.title('Speed ({0}, {1})'.format(center[0], center[1]), fontsize=20)
        plt.plot(medfilt(speed), 'r-', lw=1)
        plt.grid(True, which='both')
        plt.show()

np.savetxt(args.dest or args.source + ".csv", wrapped_traces, fmt=','.join(["%.4f"] * 2 + ["%i"] * 2 + ["%.4f"] * num_frames))
