from utilities import *
#  Ex. python 2_quality_control.py 100nM_leu100n_1 [Centers + Trace in CSV] 100nM_leu100n_1.tif [Original TIF]
from matplotlib import animation

fname1 = sys.argv[1]
fname2 = sys.argv[2]

dataname = fname1 + '.csv'
data = np.loadtxt(dataname, delimiter=",")
num_centers = data.shape[0]
# Status code: -1: unverified, 0: verified - bad, 1: verified - good
centers, status, trace = np.hsplit(data, np.array([2, 3]))

tifname = fname2 + '.tif'
raw_frames = pims.TiffStack(tifname, as_grey=False)
frames = np.array(raw_frames[0], dtype=np.uint8)


num_subplots = 9
num_frames = data.shape[1]
radius = 6

def animate(counter):
    fig, ax = plt.subplots(3, 3)
    animations = []
    cells = []
    time_text = fig.text(0.15, 0.95, '', horizontalalignment='left', verticalalignment='top')

    def init():
        for i in range(num_subplots):
            center_x, center_y = centers[counter].astype(np.int)
            ax[i % 3, i // 3].set_xlim(center_x - 10, center_x + 10)
            ax[i % 3, i // 3].set_ylim(center_y - 10, center_y + 10)
            animations.append(ax[i % 3, i // 3].imshow(frames[num_frames / num_subplots * i, center_x - 10 : center_x + 10, center_y - 10 : center_y + 10], aspect='equal', extent=[center_x - 10, center_x + 10, center_y - 10, center_y + 10]))
            x = [center_x, center_x + radius * cos(trace[counter, num_frames / num_subplots * i])]
            y = [center_x, center_y + radius * sin(trace[counter, num_frames / num_subplots * i])]
            cells.append(ax[i % 3, i // 3].plot(x, y)[0])
        time_text.set_text('Frame 0 of %d' % (num_frames / num_subplots))


    def animate(frame):
        for i in range(num_subplots):
            center_x, center_y = centers[counter].astype(np.int)
            animations[i] = ax[i % 3, i // 3].imshow(frames[(num_frames / num_subplots * i) + frame % (num_frames / num_subplots), center_x - 10: center_x + 10, center_y - 10: center_y + 10], aspect='equal', extent=[center_x - 10, center_x + 10, center_y - 10, center_y + 10])
            x = [center_x, center_x + radius * cos(trace[counter, (num_frames / num_subplots * i) + frame % (num_frames / num_subplots)])]
            y = [center_y, center_y - radius * sin(trace[counter, (num_frames / num_subplots * i) + frame % (num_frames / num_subplots)])]
            cells[i].set_data(x, y)
        time_text.set_text('Frame %d of %d' % (frame % 400, num_frames / num_subplots))

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=50)
    plt.show()

for i in range(num_centers):
    animate(i)


# np.savetxt(tifname + "-corrected.csv", np.asarray(np.hstack((centers, trace))), delimiter=",", fmt="%.4f")
