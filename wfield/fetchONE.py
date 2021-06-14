#!/usr/bin/python
### function to grab IBL data from flatiron to align to the task, must be run from iblenv
### Chris Krasniak 2021-01-08

from oneibl.one import ONE
from pathlib import Path
import numpy as np
import sys, getopt
import pandas as pd
## receptive field mapping in widefield imaging

from oneibl.one import ONE
from pathlib import Path
import numpy as np
import sys, getopt
import brainbox.io.one as bbone
import os
import brainbox.task.passive as passive
import numpy as np
import pickle
one = ONE()


one = ONE()
def fetchONE(argv):
    _,args = getopt.getopt(argv,'abc:d:')
    
    localdisk = args[0]
    print(localdisk)
    subject = localdisk.split('\\')[-2]
    date = localdisk.split('\\')[-1]
    print(subject,date)
    eid = one.search(subject=subject, date=date)
    if type(eid) == list:
        eid = eid[0]
    dtypes = ['_spikeglx_sync.channels', '_spikeglx_sync.polarities',
            '_spikeglx_sync.times']
    sync_data = one.load(eid, dataset_types=dtypes)
    data_path = one.path_from_eid(eid)
    if type(data_path) == list:
        data_path = data_path[0]
    try:
        sync_path = Path(data_path, 'raw_ephys_data')

        ch = np.load(Path(sync_path, '_spikeglx_sync.channels.npy'))
        pol = np.load(Path(sync_path, '_spikeglx_sync.polarities.npy'))
        times = np.load(Path(sync_path, '_spikeglx_sync.times.npy'))
        bpod_ch = 16
        use_ch = ch == bpod_ch
        bpod_times = times[use_ch]
        bpod_gaps = np.diff(bpod_times)
        np.save(Path(localdisk, 'bpod_times.npy'), bpod_times)
        
        rf_map = bbone.load_passive_rfmap(eid, one=one)
        rf_stim_times, rf_stim_pos, rf_stim_frames = passive.get_on_off_times_and_positions(rf_map)

        # rf_stim_times - time of frame change
        # rf_stim_pos - position of each voxel on (15 x 15) screen
        # rf_stim_frames - frames at which stimulus was turned 'on' (grey to white) or 'off' (grey to
        #                   black) at each position on screen

        np.save(Path(localdisk, 'rf_stim_times.npy'),rf_stim_times)
        np.save(Path(localdisk, 'rf_stim_pos.npy'),rf_stim_pos)

        f = open(Path(localdisk,"rf_stim_frames.pkl"),"wb")
        pickle.dump(rf_stim_frames,f)
        f.close()
    except:
        print('No ephys data, proceeding to download behavior data')
    d_sets = ['trials.choice',
            'trials.contrastLeft',
            'trials.contrastRight',
            'trials.feedbackType',
            'trials.feedback_times',
            'trials.firstMovement_times',
            'trials.goCueTrigger_times',
            'trials.goCue_times',
            'trials.intervals',
            'trials.probabilityLeft',
            'trials.response_times',
            'trials.rewardVolume',
            'trials.stimOff_times',
            'trials.stimOn_times']
    _ = one.load(eid, dataset_types=d_sets, download_only=True)
if __name__ == "__main__":
   fetchONE(sys.argv[1:])