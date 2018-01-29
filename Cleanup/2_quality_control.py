from utilities import *
#  Ex. python 2_quality_control.py 100nM_leu100n_1 [Centers + Code + Trace in CSV] 100nM_leu100n_1 [Original TIF] 0 [Quality Control Type]
from matplotlib import animation
from matplotlib.widgets import Button

fname1 = sys.argv[1]
fname2 = sys.argv[2]
type = sys.argv[3] # if type == 0, show trace graphs, if type == 1, show reconstructed cells overlaid on actual video
				   # if type == 2, show velocity graph processed from trace graph
dataname = fname1 + '.csv'
data = np.loadtxt(dataname, delimiter=",")
num_cells = data.shape[0]
# Status code: -1: unverified, 0: verified - bad, 1: verified - good
centers, status, trace = np.hsplit(data, np.array([2, 3]))

# tifname = fname2 + '.tif'
# raw_frames = pims.TiffStack(tifname, as_grey=False)
# frames = np.array(raw_frames[0], dtype=np.uint8)


num_subplots = 9
num_frames = data.shape[1]
radius = 6

def moving_average(values, window=8):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def show_trace(counter):
    fig, ax = plt.subplots()

    def record_yes(event):
        status[counter] = 1
        plt.close()

    def record_no(event):
		status[counter] = 0
		plt.close()
	
    unwrapped = np.unwrap(np.asarray(trace[i]))
    ma_trace = moving_average(unwrapped, 8) # 8*1/32 fps ~ 250 ms moving average filter window
    velocity = np.convolve([-1., 1], ma_trace, mode='full')    
    plt.xlabel('Frame', fontsize=20)
    plt.ylabel('Angle', fontsize=20)
    plt.title('Trace ({0}, {1})'.format(centers[i][0], centers[i][1]), fontsize=20)
	
    if type=="0":
        plt.plot(unwrapped, 'r-', lw=1)
    elif type=="2":
        plt.plot(velocity, 'r-', lw=1)	
        plt.ylim((-3,3))
	
	plt.grid(True, which='both')

    b_yes = Button(fig.add_axes([0.65, 0.9, 0.1, 0.03]), 'Yes')
    b_no = Button(fig.add_axes([0.80, 0.9, 0.1, 0.03]), 'No')
    b_yes.on_clicked(record_yes)
    b_no.on_clicked(record_no)

    plt.show()

def animate_frames_overlay(counter):
    fig, ax = plt.subplots(3, 3)
    animations = []
    cells = []
    time_text = fig.text(0.147, 0.92, '', horizontalalignment='left', verticalalignment='top')

    def record_yes(event):
        status[counter] = 1
        plt.close()

    def record_no(event):
        status[counter] = 0
        plt.close()

    b_yes = Button(fig.add_axes([0.605, 0.9, 0.1, 0.03]), 'Yes')
    b_no = Button(fig.add_axes([0.755, 0.9, 0.1, 0.03]), 'No')
    b_yes.on_clicked(record_yes)
    b_no.on_clicked(record_no)

    def init():
        for i in range(num_subplots):
            center_x, center_y = centers[counter].astype(np.int)
            ax[i % 3, i // 3].set_xlim(center_x - 10, center_x + 10)
            ax[i % 3, i // 3].set_ylim(center_y - 10, center_y + 10)
            animations.append(ax[i % 3, i // 3].imshow(frames[num_frames / num_subplots * i, center_y - 10 : center_y + 10, center_x - 10 : center_x + 10], aspect='equal', extent=[center_x - 10, center_x + 10, center_y - 10, center_y + 10]))
            x = [center_x + 0.5, center_x + radius * cos(trace[counter, num_frames / num_subplots * i]) + 0.5]
            y = [center_y - 0.5, center_y + radius * sin(trace[counter, num_frames / num_subplots * i]) - 0.5]
            cells.append(ax[i % 3, i // 3].plot(x, y)[0])
        time_text.set_text('Frame 0 of %d' % (num_frames / num_subplots))

    def animate(frame):
        for i in range(num_subplots):
            center_x, center_y = centers[counter].astype(np.int)
            animations[i] = ax[i % 3, i // 3].imshow(frames[(num_frames / num_subplots * i) + frame % (num_frames / num_subplots), center_y - 10: center_y + 10, center_x - 10: center_x + 10], aspect='equal', extent=[center_x - 10, center_x + 10, center_y - 10, center_y + 10])
            # TODO: is this actually correct?
            # angle is calculated with respect to numpy array, i.e. arctan(x/y), so we correct with x = center_x + sin(theta) and y = center_y + cos(theta)
            x = [center_x + 0.5, center_x + radius * cos(trace[counter, (num_frames / num_subplots * i) + frame % (num_frames / num_subplots)]) + 0.5]
            y = [center_y - 0.5, center_y + radius * sin(trace[counter, (num_frames / num_subplots * i) + frame % (num_frames / num_subplots)]) - 0.5]
            cells[i].set_data(x, y)
        time_text.set_text('Frame %d of %d' % (frame % 400, num_frames / num_subplots))

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=50)
    plt.show()


for i in range(num_cells):
    if type == "0" or type == "2":
        show_trace(i)
    elif type == "1":
        animate_frames_overlay(i)

np.savetxt(fname1 + "_checked.csv", np.asarray(np.hstack((centers, status, trace))), fmt=','.join(["%.4f"] * centers.shape[1] + ["%i"] + ["%.4f"] * trace.shape[1]))