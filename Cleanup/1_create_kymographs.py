from utilities import *
# # 
#  Ex. python traces.py 1mM 2
# [Concentration] [File No. in paths dictionary]
# (See utilities.py)
video_name, videos_dir = get_video_path(sys.argv)
fname = videos_dir + video_name
# fname = argv[1]
tifname = fname + '.tif'
meta = metamorph_timestamps.get(tifname)
raw_frames = pims.TiffStack(tifname, as_grey=False)
frames = [np.fromstring(f.data, dtype=np.int16) for f in raw_frames] # int16 may have to change depending on dtype
frames = [np.reshape(f, (-1, raw_frames[0].shape[0], raw_frames[0].shape[1]) )[0] for f in frames]

if len(sys.argv) >= 4 and sys.argv[3] == '--s':
    Show=True
    print 'Will be showing frames...'
else:
    Show = False


bit_frames = []
for i in range(len(frames)):
    bit_frames.append(convert_to_8bit(frames[i]))
frames = np.array(bit_frames)

params = np.load('params' + video_name + '_params.npy')[()]
diameter = params['diameter']
ecc = params['ecc']
minmass = params['minmass']

avg = np.mean(frames, axis = 0)

#possibly filter particles using ecc vals stationary cells will not look circular
f = tp.locate(avg, diameter=diameter, invert=False, minmass=minmass) #change 15 later, need to tune
f = f[(f['ecc'] < ecc)]
# f.head() # shows the first few rows of data

centers = []
num_elems = len(f.x) # number of particles detected
xs = list(f.x)
ys = list(f.y)
for i in range(num_elems):
    x = xs[i]
    y = ys[i]
    center = [x, y]
    centers.append(center)

radius = 8 # pixel radius of cell == length of filter
w, l = 2.45, radius # choose dimensions of rotating window
mymask = np.array([[w,0],[-w,0],[-w,l],[w,l]])

bacterial_traces = []
kymograph_images = []
for cell_num in range(num_elems): #### NOTE: For now only 10 cells until we get things working! ####
    t0 = time.time()
    print 'Percent complete:', cell_num*100./num_elems, '%'
    unprocessed_kymograph = build_kymograph(cell_num, frames, mymask, centers, Show=Show)
    print "step1", time.time() - t0
    # kymograph = invert_colors(unprocessed_kymograph) -- this line for black cells on white background
    processed_kymograph = process_kymograph(unprocessed_kymograph)
    kymograph_images.append(processed_kymograph)
    print "step2", time.time() - t0

np.save('kymographs' + video_name + '_kymographs', kymograph_images)
kymographs = np.load('kymographs' + video_name + '_kymographs.npy')
print('Sucessfully saved!')
# print_to_csv(bacterial_traces, 'test_csv', meta, tifname)
# Later, go through kymograph images and delete bad ones.
# Use those to compute trace; then save trace to csv and with np.save.