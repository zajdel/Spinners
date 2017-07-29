from utilities import *
#  Ex. python 1_create_kymographs.py 1mM [Concentration] 2 [File No. in paths dictionary]
# (See utilities.py)
from matplotlib.patches import Circle
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
# frames = [np.invert(frame, dtype=np.int16) for frame in frames]

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

params = np.load('params/' + videos_dir + video_name + '_params.npy')[()]  # dict created in 0_save_params.py
diameter = params['diameter']
ecc = params['ecc']
minmass = params['minmass']

# avg = np.mean(frames, axis=0)
# sdv = np.std(frames, axis=0)

# display sdv

# ret,th1=cv2.threshold(convert_to_8bit(sdv),50,255,cv2.THRESH_BINARY)

# use a 3x3 kernel to close the binary image
# kernel = np.ones((3,3), np.uint8)
# th1_closed=cv2.morphologyEx(th1,cv2.MORPH_CLOSE,kernel)
# avg_masked = cv2.bitwise_and(avg,avg,mask=th1_closed)

# without bandpass preprocessing we achieve tighter detection of maxima
# f = tp.locate(avg_masked,diameter=diameter,invert=False,preprocess=False)


# cycle through cells and filter the ones we want to keep. We want to do this so that the centers array is accurate and can be processed directly.

centers = []
# num_elems = len(f.x) # number of particles detected
# xs = list(f.x)
# ys = list(f.y)
# for i in range(num_elems):
#   x = xs[i]
#  y = ys[i]
# center = [x, y]
# centers.append(center)

selected_points = {}

#################################################################
#################################################################
#################################################################

fig, ax = plt.subplots()
im = ax.imshow(show_frames[0], aspect='equal')
F = ax.scatter(x=[c[0] for c in centers], y=[c[1] for c in centers], s=240, facecolors=len(centers) * ["none"],
               color=len(centers) * ["blue"], picker=5)  # 5 points tolerance


def on_press(event):
    if event.xdata and event.ydata:
        # ind = event.ind[0]

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
                area = patches.Rectangle((x - 2.5, y - 2.5), 5, 5, linewidth=0.5, edgecolor='r', facecolor='none')
                correction = patches.Rectangle((correct_x - 0.5, correct_y - 0.5), 1, 1, linewidth=0.5, edgecolor='g', facecolor='none')
                ax.add_patch(area)
                ax.add_patch(correction)
        else:
            selected_points.pop((x, y)).remove()
        # F._facecolors[ind,:] = (1, 0, 0, 0)
        # F._edgecolors[ind,:] = (1, 0, 0, 1)
        fig.canvas.draw()
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

### CHANGE RADIUS and W; view with "--s" to see if window is big enough.
radius = 3  # pixel radius of cell == length of filter
w, l = 1.5, radius  # choose dimensions of rotating window
mymask = np.array([[w, 0], [-w, 0], [-w, l], [w, l]])

kymograph_images = []
unprocessed_kymograph = np.array([])
processed_kymograph = np.array([])
for cell_num in range(num_selected_points):  #### NOTE: For now only 10 cells until we get things working! ####
    t0 = time.time()
    print 'Percent complete:', cell_num * 100. / num_selected_points, '%'
    unprocessed_kymograph = build_kymograph(cell_num, frames, mymask, selected_points, Show=Show)
    print "step1", time.time() - t0
    # kymograph = invert_colors(unprocessed_kymograph) -- this line for black cells on white background
    processed_kymograph = process_kymograph2(unprocessed_kymograph)

    kymograph_images.append(processed_kymograph)
    print "step2", time.time() - t0

np.save('kymographs/' + videos_dir + video_name + '_kymographs', kymograph_images)
print('Sucessfully saved!')

# save last kymograph as a tif for kicks on imagej
kym = Image.fromarray(processed_kymograph)
kym.save('kymographs/processedkym.tif')

kym = Image.fromarray(unprocessed_kymograph)
kym.save('kymographs/unprocessedkym.tif')
