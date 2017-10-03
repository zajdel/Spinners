from __future__ import division
from utilities import *
#  Ex. python 1_create_traces.py 1mM [Concentration] 2 [File No. in paths dictionary]
# (See utilities.py)
from Queue import Queue
from matplotlib import animation

video_name, videos_dir = get_video_path(sys.argv)
fname = videos_dir + video_name
# fname = argv[1]
tifname = fname + '.tif'
meta = metamorph_timestamps.get(tifname)
raw_frames = pims.TiffStack(tifname, as_grey=False)
frames = [np.fromstring(f.data, dtype=np.int16) for f in raw_frames]  # int16 may have to change depending on dtype
frames = [np.reshape(f, (-1, raw_frames[0].shape[0], raw_frames[0].shape[1]))[0] for f in frames]

bit_frames = []
for i in range(len(frames)):
    bit_frames.append(convert_to_8bit(frames[i]))
frames = np.array(bit_frames)

# Use following code to check fps of image stack

# a = raw_frames._tiff[0].tags.datetime.value
# b = raw_frames._tiff[1].tags.datetime.value
# d1 = datetime.strptime(a, '%Y%m%d %H:%M:%S.%f')
# d2 = datetime.strptime(b, '%Y%m%d %H:%M:%S.%f')
# delta = (d2 - d1).microseconds / 1000000 # time difference in milliseconds
# freq = 1 / delta # 62.5 fps

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^       Inverting Image          *^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

# If we have BLACK cells on WHITE background, UNCOMMENT me!
# frames = [np.invert(frame, dtype=np.int8) for frame in frames]

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^     Getting Donut Image        *^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

sdv = np.std(frames, axis=0)
# rescale sdv and then multiply by (2^N - 1), where N is the depth of each pixel
sdv = np.divide(np.subtract(sdv, np.amin(sdv)), np.amax(sdv) - np.amin(sdv)) * (2**8 - 1)

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^     Overlay Donut on BG        *^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

overlay = Image.fromarray(sdv).convert('RGB')
new_frames = []
for frame in frames:
    frame = Image.fromarray(frame).convert('RGB')
    new_frames.append(np.asarray(Image.blend(frame, overlay, 0.8)))
frameview = new_frames

# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
# *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^

# need these for cycling through cells
N = 20
show_frames = frameview[0:N]

if len(sys.argv) >= 4 and sys.argv[3] == '--s':
    Show = True
    print 'Will be showing frames...'
else:
    Show = False

#################################################################
#################################################################
#################################################################

fig, ax = plt.subplots()
im = ax.imshow(show_frames[0], aspect='equal')

# Points automatically chosen
# selected_points = set()
# def on_press(event):
#     if event.xdata and event.ydata:
#         x, y = int(round(event.xdata)), int(round(event.ydata))
#         print('You pressed {0} at ({1}, {2}) with sdv value of {3}.'.format(event.button, x, y, sdv[y, x]))
#
#         rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         correct_x, correct_y = min([(i, j) for i in range(x - 1, x + 2) for j in range(y - 1, y + 2)
#                                     if 0 < i < sdv.shape[1] and 0 < j < sdv.shape[0]], key=lambda p: sdv[p[::-1]])
#         selected_points.add(
#             (correct_x, correct_y)  # use set to avoid duplicates being stored. (use tuple because can hash.)
#         )
#         if correct_x != x or correct_y != y:
#             print('Corrected green box at ({0}, {1})'.format(correct_x, correct_y))
#             area = patches.Rectangle((x - 2.5, y - 2.5), 5, 5, linewidth=0.5, edgecolor='r', facecolor='none')
#             correction = patches.Rectangle((correct_x - 0.5, correct_y - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
#             ax.add_patch(area)
#             ax.add_patch(correction)
#     else:
#         pass

# Points not automatically chosen, click to remove
selected_points = {}
def on_press(event):
    if event.xdata and event.ydata:
        # Note that numpy arrays are accessed by [row (y), col(x)], but images are indexed by [x, y]
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print('You pressed {0} at ({1}, {2}) with sdv value of {3}.'.format(event.button, x, y, sdv[y, x]))

        if (x, y) not in selected_points:
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=0.5, edgecolor='r', facecolor='none')
            selected_points[(x, y)] = rect

            # Add the patch to the Axes
            ax.add_patch(rect)

            correct_x, correct_y = min([(i, j) for i in range(x - 1, x + 2) for j in range(y - 1, y + 2)
                                        if 0 < i < sdv.shape[1] and 0 < j < sdv.shape[0]], key=lambda p: sdv[p[::-1]])
            if correct_x != x or correct_y != y:
                print('Corrected green box at ({0}, {1})'.format(correct_x, correct_y))
                area = patches.Rectangle((x - 2.5, y - 2.5), 5, 5, linewidth=0.5, edgecolor='r', facecolor='none')
                correction = patches.Rectangle((correct_x - 0.5, correct_y - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
                ax.add_patch(area)
                ax.add_patch(correction)
        else:
            selected_points.pop((x, y)).remove()
    else:
        pass


fig.canvas.mpl_connect('button_press_event', on_press)


def init():
    im.set_data(show_frames[0])


def animate(i):
    im.set_data(show_frames[i % N])
    return im


anim = animation.FuncAnimation(fig, animate, init_func=init, interval=100)
plt.show()

# after closing plot:
selected_points = selected_points.keys()
selected_points = list(selected_points)
selected_points = [list(c) for c in selected_points]

num_selected_points = len(selected_points)

#################################################################
#################################################################
#################################################################

# v3
# 1. Threshold the sdv image (close if necessary) and use Canny edge detection on frames.
# 2. Take bitwise AND of the threshold and the image stack, result should be "active" edges.
# 3. For each selected point:
#     3a. Find the points on the edge.  Define a region of interest (5x5 or 6x6 box) around the selected point and find
#         all white points in the "active" image.
#     3b. Take weighted regression of the points on the edge, the center point must lie on the line.  Using the center
#         point as the origin, the line lies in two of the four quadrants.  Compare each point on the edge to the center,
#         final angle can be calculated from slope of line and quadrant with most points.

# ret, thresh = cv2.threshold(sdv, 50, 255, cv2.THRESH_BINARY)
# plt.imshow(sdv.astype(np.uint8))
ret, thresh = cv2.threshold(sdv.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt.imshow(thresh)

num_frames = len(frames)
frames_and_thresh = [cv2.bitwise_and(frames[i], thresh) for i in range(num_frames)]
thresh_edges = [cv2.Canny(frame, 100, 250, apertureSize=3) for frame in frames_and_thresh]
edges = [cv2.Canny(frames[i], 100, 250, apertureSize=3) for i in range(num_frames)]
tifffile.imsave('edges2000.tif', np.asarray(edges))


def plot_ellipses(ellipses):
    # necessary to redefine fig to show animation?
    fig, ax = plt.subplots()

    im = ax.imshow(edges[0], aspect='equal')

    ellipse_patch = None

    p1, p2 = None, None

    def init():
        # preferably nonlocal, but not supported in Python 2
        global ellipse_patch
        global p1, p2
        im.set_data(edges[0])
        # find first non-None ellipse in ellipses
        first_ellipse = next(ellipse for ellipse in ellipses if ellipse is not None)
        ellipse_patch = patches.Ellipse(*first_ellipse, color='r', fill=False)
        ax.add_patch(ellipse_patch)

        center, width, height, theta = first_ellipse
        major, minor = width / 2, height / 2
        p1 = patches.Rectangle(
            (center[0] - cos(radians(theta)) * major - 0.5, center[1] - sin(radians(theta)) * major - 0.5), 1, 1,
            color='b', fill=False)
        p2 = patches.Rectangle(
            (center[0] + cos(radians(theta)) * major - 0.5, center[1] + sin(radians(theta)) * major - 0.5), 1, 1,
            color='g', fill=False)
        ax.add_patch(p1)
        ax.add_patch(p2)

    def animate(i):
        global ellipse_patch
        global p1, p2
        im.set_data(edges[i % len(edges)])
        if ellipse_patch:
            ellipse_patch.remove()
            ellipse_patch = None

            p1.remove()
            p2.remove()
        if ellipses[i % len(edges)]:
            ellipse_patch = patches.Ellipse(*ellipses[i % len(edges)], color='r', fill=False)
            ax.add_patch(ellipse_patch)

            center, width, height, theta = ellipses[i % len(edges)]
            major, minor = width / 2, height / 2
            p1 = patches.Rectangle((center[0] - cos(radians(theta)) * major - 0.5, center[1] - sin(radians(theta)) * major -0.5), 1, 1, color='b', fill=False)
            p2 = patches.Rectangle((center[0] + cos(radians(theta)) * major - 0.5, center[1] + sin(radians(theta)) * major - 0.5), 1, 1, color='g', fill=False)
            ax.add_patch(p1)
            ax.add_patch(p2)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=100)

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=10)
    # anim.save('ellipses.mp4', writer=writer)

    plt.show()


def find_ellipse(point, frame):
    points = Queue()
    points.put(point)
    fringe = set()
    marked = set()
    marked.add(point)

    start = point
    # find one pixel on border with floodfill, then perform border traversal
    if edges[frame][point[1], point[0]] == 0:
        while not len(fringe):
            p = points.get()
            if edges[frame][p[1], p[0]] == 255:
                fringe.add(p)
                start = p
                break
            p1 = (p[0] - 1, p[1])
            p2 = (p[0] + 1, p[1])
            p3 = (p[0], p[1] - 1)
            p4 = (p[0], p[1] + 1)
            for p in [p1, p2, p3, p4]:
                if p not in marked and 0 <= p[0] < len(edges[frame][1]) and 0 <= p[1] < len(edges[frame][0]):
                    marked.add(p)
                    points.put(p)

    points = Queue()
    points.put(start)
    fringe = set()
    fringe.add(start)

    while not points.empty():
        p = points.get()
        p1 = (p[0] - 1, p[1])
        p2 = (p[0] + 1, p[1])
        p3 = (p[0], p[1] - 1)
        p4 = (p[0], p[1] + 1)
        p5 = (p[0] - 1, p[1] - 1)
        p6 = (p[0] - 1, p[1] + 1)
        p7 = (p[0] + 1, p[1] - 1)
        p8 = (p[0] + 1, p[1] + 1)
        for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
            if (p not in fringe
                    and 0 <= p[0] < len(edges[frame][1])
                    and 0 <= p[1] < len(edges[frame][0])
                    and edges[frame][p[1], p[0]] == 255
                    and frames_and_thresh[i][p[1], p[0]] != 0):
                fringe.add(p)
                points.put(p)

    # returns ((x-coordinate of center, y-coordinate of center), (height, width), rotation), width >= height
    # angle orientation: 0 vertical up, positive clockwise
    if len(fringe) >= 5:
        ellipse = cv2.fitEllipse(np.array(list(fringe)))
    else:
        # fitEllipse does not work if there are fewer than 5 points on the fringe
        return None
    center = ellipse[0]
    # width >= height
    minor = ellipse[1][0] / 2 # minor = height / 2
    major = ellipse[1][1] / 2 # major = width / 2
    theta = ellipse[2] + 90 # define angle to be zero horizontally right

    return list(fringe), center, major, minor, theta


def find_correct_orientation(theta, center, points):
    score = {theta: 0, (theta + 180) % 360: 0}
    for p in points:
        a = np.arctan2(p[1] - center[1], p[0] - center[0])
        b = np.degrees(a)
        c = b - theta
        d = c % 180
        if (np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) - theta) % 180 <= 90:
            score[theta] += 1
        else:
            score[(theta + 180) % 360] += 1
    return max(score, key=score.get)


unwrapped = None
for point in selected_points:
    ellipses = []
    trace = []
    for i in range(num_frames):
        ellipse = find_ellipse(tuple(point), i)
        if ellipse:
            border, center, major, minor, theta = ellipse

            correction = find_correct_orientation(theta, point, border)

            ellipses.append([center, major * 2, minor * 2, theta])
            # trace.append(original_theta)
            # trace.append(radians(original_theta))
            trace.append(radians(correction % 360))
        else:
            ellipses.append(None)
            trace.append(None)

    if trace[0] is None:
        # TODO: what to do if first ellipse is undefined
        trace[0] = next(ang for ang in trace if ang is not None)
    for i in range(1, len(trace) - 2):
        if trace[i] is None:
            # if ellipse does not exist, take previous and next non-None values and linearly approximate
            try:
                # TODO: check
                next_valid_index = next(j for j in range(i, len(trace)) if trace[j] is not None)
                trace[i] = trace[i - 1] + (trace[next_valid_index] - trace[i - 1]) / (next_valid_index - (i - 1))
            except StopIteration:
                prev, next_prev = i, i - 1
                for j in range(i, -1, -1):
                    if edges[j] is not None:
                        prev = j
                        break
                for k in range(prev, -1, -1):
                    if edges[k] is not None:
                        next_prev = k
                        break
                trace[i] = trace[next_prev] + (trace[prev] - trace[next_prev]) / (prev - next_prev) * (i + 1 - prev)
    plot_ellipses(ellipses)

    unwrapped = np.unwrap(np.asarray(trace))

    plt.xlabel('Frame', fontsize=20)
    plt.ylabel('Angle', fontsize=20)
    plt.title('Trace', fontsize=20)
    plt.plot(unwrapped, 'r-', lw=1)
    # annotation for 100nM_leu100n_1.tif
    plt.axvspan(203, 207, color='green', alpha=0.5)
    plt.axvspan(1656, 1658, color='green', alpha=0.5)
    plt.axvspan(1662, 1666, color='green', alpha=0.5)
    plt.axvspan(1824, 1825, color='green', alpha=0.5)
    # plt.plot(trace[:300], 'r-', lw=1)
    # plt.plot(trace, 'bo', markersize=1)
    plt.grid(True, which='both')
    plt.show()

    speed = []
    for i in range(unwrapped.shape[0]):
        indices = []
        angs = []
        for j in range(i - 2, i + 3):
            if 0 <= j < unwrapped.shape[0]:
                indices.append(j)
                angs.append(unwrapped[j])
        try:
            slope = linregress(indices, angs)[0]
        except RuntimeWarning:
            # TODO: confirm this is zero division error and find fix (infinite slope)
            print "Encountered zero division error"
            print angs
        speed.append(slope)

    plt.xlabel('Frame', fontsize=20)
    plt.ylabel('Speed', fontsize=20)
    plt.title('Speed', fontsize=20)
    plt.plot(speed, 'r-', lw=1)
    plt.grid(True, which='both')
    plt.show()
