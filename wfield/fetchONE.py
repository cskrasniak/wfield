#!/usr/bin/python
### function to grab IBL data from flatiron to align to the task, must be run from iblenv
### Chris Krasniak 2021-01-08

from oneibl.one import ONE
from pathlib import Path
import numpy as np
import sys, getopt
one = ONE()
def fetchONE(argv):
    _,args = getopt.getopt(argv,'abc:d:')
    
    localdisk = args[0]
    print(localdisk)
    subject = localdisk.split('\\')[-3]
    date = localdisk.split('\\')[-1]
    print(subject,date)
    eid = one.search(subject=subject, date=date)
    dtypes = ['_spikeglx_sync.channels', '_spikeglx_sync.polarities',
            '_spikeglx_sync.times']
    sync_data = one.load(eid, dataset_types=dtypes)
    data_path = one.path_from_eid(eid)
    sync_path = Path(data_path[0], 'raw_ephys_data')

    ch = np.load(Path(sync_path, '_spikeglx_sync.channels.npy'))
    pol = np.load(Path(sync_path, '_spikeglx_sync.polarities.npy'))
    times = np.load(Path(sync_path, '_spikeglx_sync.times.npy'))
    bpod_ch = 16
    use_ch = ch == bpod_ch
    bpod_times = times[use_ch]
    bpod_gaps = np.diff(bpod_times)
    np.save(Path(localdisk, 'bpod_times.npy'), bpod_times)

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