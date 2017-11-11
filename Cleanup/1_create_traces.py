from __future__ import division
from utilities import *
#  Ex. python 1_create_traces.py 1mM [Concentration] 2 [File No. in paths dictionary]
# (See utilities.py)
from Queue import Queue
from matplotlib import animation

# video_name, videos_dir = get_video_path(sys.argv)
# fname = videos_dir + video_name
# # fname = argv[1]
fname = sys.argv[1]
tifname = fname + '.tif'
# meta = metamorph_timestamps.get(tifname)
raw_frames = pims.TiffStack(tifname, as_grey=False)
# frames = [np.fromstring(f.data, dtype=np.int8) for f in raw_frames]  # int16 may have to change depending on dtype
# frames = [np.reshape(f, (-1, raw_frames[0].shape[0], raw_frames[0].shape[1]))[0] for f in frames]
# bit_frames = []
# for i in range(len(frames)):
#     bit_frames.append(convert_to_8bit(frames[i]))
# frames = np.array(bit_frames)
frames = np.array(raw_frames[0], dtype=np.uint8)

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

sdv = np.std(frames[0:15], axis=0)
# rescale sdv and then multiply by (2^N - 1), where N is the depth of each pixel
sdv = np.divide(np.subtract(sdv, np.amin(sdv)), np.amax(sdv) - np.amin(sdv)) * (2**8 - 1)

plt.imshow(sdv)
plt.show()

def draw_circles(img, circles):
    # img = cv2.imread(img,0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in circles[0, :]:
    # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.putText(cimg, str(i[0]) + str(',') + str(i[1]), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    return cimg

def detect_circles():
    gray = sdv.astype(np.uint8)
    # ret, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)

    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    plt.imshow(gray)
    plt.show()
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    plt.imshow(gray)
    plt.show()

    # gray_blur = cv2.medianBlur(gray, 3)  # Remove noise before laplacian
    # gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=3)
    # dilate_lap = cv2.dilate(gray_lap, (3, 3))  # Fill in gaps from blurring. This helps to detect circles with broken edges.
    # # Furture remove noise introduced by laplacian. This removes false pos in space between the two groups of circles.
    # lap_blur = cv2.bilateralFilter(dilate_lap, 5, 9, 9)
    # Fix the resolution to 16. This helps it find more circles. Also, set distance between circles to 55 by measuring dist in image.
    # Minimum radius and max radius are also set by examining the image.
    # circles = cv2.HoughCircles(gray_lap, cv2.HOUGH_GRADIENT, 16, 5, param2=15, minRadius=0, maxRadius=9)
    # for i in range(1, 72, 5):
    #     for j in range(1, 14, 2):
    #         for k in range(10, 211, 20):
    #             circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, i, j, param2=k, maxRadius = 12)
    #             num_circles = circles[0].shape[0] if circles is not None else 0
    #             print("dp: {}, minDist: {}, param2: {} gives {} circles".format(i, j, k, num_circles))

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 6, 3, param2=10, minRadius=5, maxRadius=12)
    # 16 2 200 -- 67
    # 16 4 200 -- 67
    # 16 8 200 -- 67
    # 1 4 200 -- 0
    # 32 4 200 -- 32
    # 8 4 200 -- 138
    # 4 4 200 -- 195
    # 64 4 200 -- 2
    # 8 4 100 -- 256
    print("{} circles detected.".format(circles[0].shape[0]))
    cimg = draw_circles(gray, circles)
    return cimg

cimg = detect_circles()
plt.imshow(cimg)
plt.show()

# featuresize = 11
# f1=tp.locate(sdv, featuresize)
#
# plt.figure()
# tp.annotate(f1, sdv)


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

            # correct_x, correct_y = min([(i, j) for i in range(x - 1, x + 2) for j in range(y - 1, y + 2)
            #                             if 0 < i < sdv.shape[1] and 0 < j < sdv.shape[0]], key=lambda p: sdv[p[::-1]])
            # if correct_x != x or correct_y != y:
            #     print('Corrected green box at ({0}, {1})'.format(correct_x, correct_y))
            #     area = patches.Rectangle((x - 2.5, y - 2.5), 5, 5, linewidth=0.5, edgecolor='r', facecolor='none')
            #     correction = patches.Rectangle((correct_x - 0.5, correct_y - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
            #     ax.add_patch(area)
            #     ax.add_patch(correction)
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

num_frames = len(frames)

# # Median filter with 3x3 kernel
# frames = [cv2.medianBlur(f, 3) for f in frames]
#
# # Local Otsu threshold with radius 15
# radius = 15
# selem = disk(radius)
# # frames = [(f >= rank.otsu(f, selem)) * 255 for f in frames]
# for i in range(num_frames):
#     start = time.time()
#     frames[i] = (frames[i] >= rank.otsu(frames[i], selem)) * 255
#     print("Frames %i: %f seconds elapsed." % (i, time.time() - start))
#
# tifffile.imsave('local_otsu.tif', np.asarray(frames))
#
# # Standard deviation of local thresholded Otsu image
# sdv = np.std(frames, axis=0)
# # rescale sdv and then multiply by (2^N - 1), where N is the depth of each pixel
# sdv = np.divide(np.subtract(sdv, np.amin(sdv)), np.amax(sdv) - np.amin(sdv)) * (2**8 - 1)
# sdv = (sdv >= threshold_otsu(sdv)) * 255
#
# # Take bitwise AND of SDV mask and thresholded Otsu frames
# frames_and_sdv = [cv2.bitwise_and(frames[i].astype(np.uint8), sdv.astype(np.uint8)) for i in range(num_frames)]
#
# tifffile.imsave('thresh_frames.tif', np.asarray(frames_and_sdv))


def find_furthest_points(center, frame):
    # get all points connected to center ("cell"), find furthest points
    fringe = Queue()
    fringe.put(center)
    cell = set()
    cell.add(center)
    marked = set()
    marked.add(center)

    # Modified breadth-first search
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
                    and euclidean_distance(center, p) <= 10
                    and frames[frame][p[1], p[0]] == 0):
                marked.add(p)
                cell.add(p)
                fringe.put(p)

    cell = list(cell)
    # pairs = [(cell[i], cell[j]) for i in range(len(cell)) for j in range(i + 1, len(cell))]
    # return max(pairs, key=lambda x: euclidean_distance(*x))
    max_dist = max([euclidean_distance(p, center) for p in cell])
    return [p for p in cell if euclidean_distance(p, center) == max_dist]


traces = []
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
    traces.append(unwrapped)

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
    # plt.axvspan(0, 26, color='green', alpha=0.5)
    # plt.axvspan(146, 168, color='green', alpha=0.5)
    # plt.axvspan(173, 199, color='green', alpha=0.5)
    # plt.axvspan(209, 217, color='green', alpha=0.5)
    # plt.axvspan(240, 273, color='green', alpha=0.5)
    # plt.axvspan(279, 281, color='green', alpha=0.5)
    # plt.axvspan(287, 308, color='green', alpha=0.5)
    # plt.axvspan(331, 383, color='green', alpha=0.5)
    # plt.axvspan(386, 400, color='green', alpha=0.5)
    # plt.axvspan(406, 440, color='green', alpha=0.5)
    # plt.axvspan(441, 482, color='green', alpha=0.5)
    # plt.axvspan(483, 504, color='green', alpha=0.5)
    # plt.axvspan(507, 513, color='green', alpha=0.5)
    # plt.axvspan(516, 535, color='green', alpha=0.5)
    # plt.axvspan(538, 563, color='green', alpha=0.5)
    # plt.axvspan(624, 878, color='green', alpha=0.5)

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
    # plt.axvspan(0, 26, color='green', alpha=0.5)
    # plt.axvspan(146, 168, color='green', alpha=0.5)
    # plt.axvspan(173, 199, color='green', alpha=0.5)
    # plt.axvspan(209, 217, color='green', alpha=0.5)
    # plt.axvspan(240, 273, color='green', alpha=0.5)
    # plt.axvspan(279, 281, color='green', alpha=0.5)
    # plt.axvspan(287, 308, color='green', alpha=0.5)
    # plt.axvspan(331, 383, color='green', alpha=0.5)
    # plt.axvspan(386, 400, color='green', alpha=0.5)
    # plt.axvspan(406, 440, color='green', alpha=0.5)
    # plt.axvspan(441, 482, color='green', alpha=0.5)
    # plt.axvspan(483, 504, color='green', alpha=0.5)
    # plt.axvspan(507, 513, color='green', alpha=0.5)
    # plt.axvspan(516, 535, color='green', alpha=0.5)
    # plt.axvspan(538, 563, color='green', alpha=0.5)
    # plt.axvspan(624, 878, color='green', alpha=0.5)

    plt.grid(True, which='both')
    # plt.savefig("leu_100u_6_speed.png")
    plt.show()

np.savetxt("traces.csv", np.asarray(traces), delimiter=",")
