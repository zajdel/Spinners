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

num_selected_points = len(selected_points)

#################################################################
#################################################################
#################################################################

# ret, thresh = cv2.threshold(sdv, 50, 255, cv2.THRESH_BINARY)
# plt.imshow(sdv.astype(np.uint8))

ret, thresh = cv2.threshold(sdv.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt.imshow(thresh)

num_frames = len(frames)

# Median filter with 3x3 kernel
frames = [cv2.medianBlur(f, 3) for f in frames]

# Local Otsu threshold with radius 15
radius = 15
selem = disk(radius)
# frames = [(f >= rank.otsu(f, selem)) * 255 for f in frames]
for i in range(5):
    start = time.time()
    local_thresholded_frame = (frames[i] >= rank.otsu(frames[i], selem)) * 255
    print("Frames %i: %f seconds elapsed." % (i, time.time() - start))

# Standard deviation of local thresholded Otsu image
sdv = np.std(frames, axis=0)
# rescale sdv and then multiply by (2^N - 1), where N is the depth of each pixel
sdv = np.divide(np.subtract(sdv, np.amin(sdv)), np.amax(sdv) - np.amin(sdv)) * (2**8 - 1)
sdv = (sdv >= threshold_otsu(sdv)) * 255

# Take bitwise AND of SDV mask and thresholded Otsu frames
frames_and_sdv = [cv2.bitwise_and(frames[i].astype(np.uint8), sdv.astype(np.uint8)) for i in range(num_frames)]

tifffile.imsave('thresh_frames.tif', np.asarray(frames_and_sdv))

# plt.imshow(frames_and_thresh[0])

# thresh_edges = [cv2.Canny(frame, 100, 250, apertureSize=3) for frame in frames_and_thresh]
# edges = [cv2.Canny(frames[i], 100, 250, apertureSize=3) for i in range(num_frames)]
# tifffile.imsave('edges2000.tif', np.asarray(edges))


def find_furthest_points(center, frame):
    # get all points connected to center ("cell"), find furthest points
    fringe = Queue()
    fringe.put(center)
    cell = set()
    cell.add(center)
    marked = set()
    marked.add(center)

    # Modified breadth-first search
    while not fringe.empty() and len(cell) <= 9:
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
                    and 0 <= p[0] < len(frames_and_sdv[frame][1])
                    and 0 <= p[1] < len(frames_and_sdv[frame][0])
                    and frames_and_sdv[frame][p[1], p[0]] == 0):
                marked.add(p)
                cell.add(p)
                fringe.put(p)

    cell = list(cell)
    # pairs = [(cell[i], cell[j]) for i in range(len(cell)) for j in range(i + 1, len(cell))]
    # return max(pairs, key=lambda x: euclidean_distance(*x))
    max_dist = max([euclidean_distance(p, center) for p in cell])
    return [p for p in cell if euclidean_distance(p, center) == max_dist]


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
        ang = np.arctan2(furthest_point[0] - center[0], center[1] - furthest_point[1])
        trace.append(ang)

    # unwrap trace and apply 1D median filter (default kernel size 3)
    unwrapped = medfilt(np.unwrap(np.asarray(trace)))

    plt.xlabel('Frame', fontsize=20)
    plt.ylabel('Angle', fontsize=20)
    plt.title('Trace', fontsize=20)
    plt.plot(unwrapped, 'r-', lw=1)

    # # annotation for 100nM_leu100n_1.tif
    # plt.axvspan(203, 207, color='green', alpha=0.5)
    # plt.axvspan(1656, 1658, color='green', alpha=0.5)
    # plt.axvspan(1662, 1666, color='green', alpha=0.5)
    # plt.axvspan(1824, 1825, color='green', alpha=0.5)

    # annotation for leu_100um_2.tif (100uM_leu100u_6.tif) (2017-09-22) Point 150, 18
    plt.axvspan(0, 26, color='green', alpha=0.5)
    plt.axvspan(146, 168, color='green', alpha=0.5)
    plt.axvspan(173, 199, color='green', alpha=0.5)
    plt.axvspan(209, 217, color='green', alpha=0.5)
    plt.axvspan(240, 273, color='green', alpha=0.5)
    plt.axvspan(279, 281, color='green', alpha=0.5)
    plt.axvspan(287, 308, color='green', alpha=0.5)
    plt.axvspan(331, 383, color='green', alpha=0.5)
    plt.axvspan(386, 400, color='green', alpha=0.5)
    plt.axvspan(406, 440, color='green', alpha=0.5)
    plt.axvspan(441, 482, color='green', alpha=0.5)
    plt.axvspan(483, 504, color='green', alpha=0.5)
    plt.axvspan(507, 513, color='green', alpha=0.5)
    plt.axvspan(516, 535, color='green', alpha=0.5)
    plt.axvspan(538, 563, color='green', alpha=0.5)
    plt.axvspan(624, 878, color='green', alpha=0.5)

    # plt.plot(trace[:300], 'r-', lw=1)
    # plt.plot(trace, 'bo', markersize=1)
    plt.grid(True, which='both')
    # plt.savefig("leu_100u_6_trace.png")
    plt.show()

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
    plt.title('Speed', fontsize=20)
    plt.plot(medfilt(speed), 'r-', lw=1)

    # annotation for leu_100um_2.tif (100uM_leu100u_6.tif) (2017-09-22) Point 150, 18
    plt.axvspan(0, 26, color='green', alpha=0.5)
    plt.axvspan(146, 168, color='green', alpha=0.5)
    plt.axvspan(173, 199, color='green', alpha=0.5)
    plt.axvspan(209, 217, color='green', alpha=0.5)
    plt.axvspan(240, 273, color='green', alpha=0.5)
    plt.axvspan(279, 281, color='green', alpha=0.5)
    plt.axvspan(287, 308, color='green', alpha=0.5)
    plt.axvspan(331, 383, color='green', alpha=0.5)
    plt.axvspan(386, 400, color='green', alpha=0.5)
    plt.axvspan(406, 440, color='green', alpha=0.5)
    plt.axvspan(441, 482, color='green', alpha=0.5)
    plt.axvspan(483, 504, color='green', alpha=0.5)
    plt.axvspan(507, 513, color='green', alpha=0.5)
    plt.axvspan(516, 535, color='green', alpha=0.5)
    plt.axvspan(538, 563, color='green', alpha=0.5)
    plt.axvspan(624, 878, color='green', alpha=0.5)

    plt.grid(True, which='both')
    # plt.savefig("leu_100u_6_speed.png")
    plt.show()
