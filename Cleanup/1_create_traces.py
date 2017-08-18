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
#                                     if 0 < i < sdv.shape[0] and 0 < j < sdv.shape[1]], key=lambda p: sdv[p[::-1]])
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
    num_frames = 300
    edges = [cv2.Canny(frames[i], 100, 250, apertureSize=3) for i in range(num_frames)]
    plt.imshow(edges[0])
    a1 = patches.Rectangle((71 - 0.5, 70 - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
    a2 = patches.Rectangle((78 - 0.5, 77 - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
    ax.add_patch(a1)
    ax.add_patch(a2)
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
                                        if 0 < i < sdv.shape[0] and 0 < j < sdv.shape[1]], key=lambda p: sdv[p[::-1]])
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

# 1. Canny around center
# 2. Find points farthest apart on ellipse --> constrain to within distance of line (line passes within proximity of center)
# 3. Calculate orientation
#    a) separated points
#    b) distance to center (which quadrant?)
# Reach: plot all traces for 1 condition

num_frames = 300
edges = [cv2.Canny(frames[i], 100, 250, apertureSize=3) for i in range(num_frames)]
# kernel = np.ones((3, 3),np.uint8)
# edges = [cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel) for edge in edges]
# edges = [edges[i] + frames[i] for i in range(len(edges))]
# plt.imshow(edges[8])

def find_furthest_points(point, frame):
    best_solution = {'p1': (0, 0), 'p2': (0, 0), 'distance': -sys.maxint - 1}

    points = Queue()
    points.put(point)
    fringe = set()
    marked = set()
    marked.add(point)

    while not points.empty():
        p = points.get()
        p1 = (p[0] - 1, p[1])
        p2 = (p[0] + 1, p[1])
        p3 = (p[0], p[1] - 1)
        p4 = (p[0], p[1] + 1)
        potential = [p1, p2, p3, p4]
        if edges[frame][p[1], p[0]] == 255:
            fringe.add(p)
            p5 = (p[0] - 1, p[1] - 1)
            p6 = (p[0] - 1, p[1] + 1)
            p7 = (p[0] + 1, p[1] - 1)
            p8 = (p[0] + 1, p[1] + 1)
            potential.extend([p5, p6, p7, p8])
            for p in potential:
                if p not in marked and edges[frame][p[1], p[0]] == 255:
                    marked.add(p)
                    points.put(p)
        else:
            for p in potential:
                if (p not in marked
                    and 0 <= p[0] < len(edges[frame][1])
                    and 0 <= p[1] < len(edges[frame][0])
                    and point[0] - 6 <= p[0] <= point[0] + 6
                    and point[1] - 6 <= p[1] <= point[1] + 6):
                    marked.add(p)
                    points.put(p)

    fringe = list(fringe)
    for i in range(len(fringe)):
        for j in range(i + 1, len(fringe)):
            d = np.linalg.norm(np.asarray(fringe[i]) - np.asarray(fringe[j]))
            if d > best_solution['distance']:
                best_solution['p1'] = fringe[i]
                best_solution['p2'] = fringe[j]
                best_solution['distance'] = d
    return best_solution

for point in selected_points:
    for i in range(num_frames):
        plt.imshow(edges[0])
        best_solution = find_furthest_points(tuple(point), i)
        print(best_solution)

# np.save('kymographs/' + videos_dir + video_name + '_kymographs', kymograph_images)
# print('Sucessfully saved!')
#
# # save last kymograph as a tif for kicks on imagej
# kym = Image.fromarray(processed_kymograph)
# kym.save('kymographs/processedkym.tif')
#
# kym = Image.fromarray(unprocessed_kymograph)
# kym.save('kymographs/unprocessedkym.tif')
