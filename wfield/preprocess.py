from wfield import *
import matplotlib.pyplot as plt
import pandas as pd
from labcams import parse_cam_log
import numpy as np
import imageio
from scipy.ndimage import gaussian_filter1d, gaussian_filter


# This should be an SSD or a fast drive
localdisk = r'D:\imaging_data\1photon\test_motion_corr_12-10-2020'
######### Run Motion Correction####################################################################
dat_path = glob(pjoin(localdisk,'*.dat'))[0]
# open file with read/write 
dat = mmap_dat(dat_path, mode='r+')
(yshifts,xshifts),rot = motion_correct(dat,chunksize=512,
                                     apply_shifts=True)
del dat # close and finish writing
# save the shifts
shifts = np.rec.array([yshifts,xshifts],dtype=[('y','float32'),('x','float32')])
np.save(pjoin(localdisk,'motion_correction_shifts.npy'),shifts)

plt.matplotlib.style.use('ggplot')
shifts = np.load(pjoin(localdisk,'motion_correction_shifts.npy'))
plot_summary_motion_correction(shifts,localdisk);

###### Get onset times and eventually all other trial events ######################################

logdata,led,sync,ncomm = parse_cam_log(pjoin(localdisk,'20201210_run000_00000000.camlog'),
                                       readTeensy=True)

pulsetimes = []
counts = np.unique(sync['count'])
for c in counts:
    pulse = sync[np.array(sync['count']) == c].timestamp
    if len(pulse) == 2:
        pulsetimes.append(pulse)
    else:
        print('There was a pulse with no rise: '.format(pulse))
pulsetimes = np.stack(pulsetimes)
print('N pulses: {0}'.format(len(pulsetimes)))

all_onsets = sync.iloc[::2] #take only the upfronts
onsets = np.diff(pulsetimes) < 10 #shortest pulses are the trial onset times
onsets = [on[0] for on in onsets]
onset_times = all_onsets[onsets]
# onset_times.to_pickle(pjoin(localdisk,'trial_onsets.npy'))
np.save(pjoin(localdisk,'trial_onsets.npy'),onset_times)

###### Calculate the baseline for calculating DF/F ################################################

# ## Uncomment this to do trial-by-trial baseline
# dat_path = glob(pjoin(localdisk,'*.dat'))[0]
# dat = mmap_dat(dat_path)
# # load trial onsets
# trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'),allow_pickle=True)

# nbaseline_frames = 15

# frames_average_trials = frames_average_for_trials(dat,
#                                            trial_onsets['frame'],
#                                            nbaseline_frames)
# # Compute the average of all trials and save it
# np.save(pjoin(localdisk,'frames_average.npy'),frames_average_trials.mean(axis=0))

## Uncomment this to do session-long baseline
dat_path = glob(pjoin(localdisk,'*.dat'))[0]
dat = mmap_dat(dat_path)
