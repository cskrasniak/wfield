from labcams import parse_cam_log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  wfield import *
import imageio
import math
from matplotlib import cm
import matplotlib.colors as mplc
from matplotlib.lines import Line2D
import seaborn as sns


def extract_onset_times(localdisk):
        
    logdata,led,sync,ncomm = parse_cam_log(glob(pjoin(localdisk,'*.camlog'))[0],readTeensy=True)

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
    onset_times['frame']=(onset_times['frame']/2).astype(int)
    # onset_times.to_pickle(pjoin(localdisk,'trial_onsets.npy'))
    np.save(pjoin(localdisk,'trial_onsets.npy'),onset_times[['timestamp','frame']])
    return onset_times

def fetch_task_data(subject,date,exp='001',FlatIron='G:\\FlatIron\\zadorlab\\Subjects\\'):
    """
    fetches the task data for a given session and puts it into a dataframe
    Input:
        subject: subject_nickname
        date: date of the experiment
        exp: experiment number
        FlatIron: folder for the dowloaded FlatIron data

    Output:
        DataFrame of length ntrials containing all alf information
    """
    
    import os
    alf_path = pjoin(FlatIron,subject,date,exp,'alf')
    files = os.listdir(alf_path)
    df = pd.DataFrame(columns=['choice','contrastLeft','contrastRight','feedback_times',
                               'feedbackType','firstMovement_times','goCue_times',
                               'goCueTrigger_times','intervals','intervals_bpod','probabilityLeft',
                               'response_times','rewardVolume','stimOff_times','stimOn_times'])
    for file in files:
        for column in df.columns:
            if column in str(file):
                df[column] = np.load(pjoin(alf_path,file))
    df['contrastRight'][np.isnan(df['contrastRight'])] = 0
    df['contrastLeft'][np.isnan(df['contrastLeft'])] = 0
    df['signedContrast'] = df['contrastRight'] - df['contrastLeft']
    return df


def sync_to_task(localdisk):
    """
    synchronize the task data to the imaging data.
    Input:
        localdisk: directory where the imaging data is, and where bpod_times file has been saved
                   this requires you to have already run the function fetchONE script for this 
                   session to make sure that alf data is on local computer
    Output:
        a dataframe that has the sync times and frames from both imaging and bpod
    """

    logdata,led,sync,ncomm = parse_cam_log(glob(pjoin(localdisk,'*.camlog'))[0],readTeensy=True)
    bpod_times = np.load(pjoin(localdisk,'bpod_times.npy'))
    bpod_gaps = np.diff(bpod_times)
    sync.timestamp = sync.timestamp/1000 #convert from milliseconds to seconds
    sync_gaps = np.diff(sync.timestamp)
    sync['task_time'] = np.nan
    assert len(bpod_times) == len(sync)

    for i in range(len(bpod_gaps)): 
        # make sure the pulses are the same
        if math.isclose(bpod_gaps[i],sync_gaps[i],abs_tol = .0015):
            sync['task_time'].iloc[i] = bpod_times[i]
        else:
            print('WARNING: syncs do not line up for index {}!!!'.format[i])
    sync['frame'] = (sync['frame']/2).astype(int)
    return sync.dropna(axis=0)

def time_to_frames(sync,localdisk,event_times, dropna=True):
    """
    This function attributes the closest frame of the imaging data to an array of events, such as 
    stimulus onsets.
    sync: the synchronization pandas array including the frame and timestamp from imaging, and the
          timestamp for the bpod events from the fpga
    event_times: the time in seconds of the bpod event associated with 

    returns an array of len(event_times) with a frame attributed to each event
    """
    if dropna:
        event_times = event_times[~np.isnan(event_times)]
    logdata,led,_,ncomm = parse_cam_log(glob(pjoin(localdisk,'*.camlog'))[0],readTeensy=True)
    sync['conversion'] = sync['timestamp']-sync['task_time']

    led['timestamp'] = led['timestamp']/1000  # this is the time of each frame
    def find_nearest(array, value):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()
    event_frames = np.empty(event_times.shape)
    for i in range(len(event_times)):
        offset = sync.iloc[find_nearest(sync['task_time'],event_times[i])]['conversion']
        event_frames[i] = led.iloc[find_nearest(led['timestamp'],event_times[i]+offset)]['frame']

    return (event_frames/2).astype(int)

def make_allen_outline(atlas, allen_transform=None):
    '''
       takes in the projection of the dorsal allen atlas (an image with a number for each pixel 
       coresponding to each brain area), and the transform matrix to apply if you want to map 
       it onto an image.
       outputs an image of the outline of the dorsal cortex
    '''
    try: 
        outline = im_apply_transform(np.diff(atlas)!=0,allen_transform)
    except:
        outline = np.diff(atlas)!=0

    # diff removes the last row, so add back a row of zeros
    pad = np.zeros([outline.shape[0],1])
    # pad[:] = np.nan
    outline = np.concatenate([outline,pad],axis=1)
    outline[outline < .1] = 0
    outline[outline > .1] = 1
    outline_im = outline
    outline_trace = np.where(outline_im)
    outline_trace = np.array(outline_trace)
    # change the image to a L x W x 4 matrix to make all the 0 values clear instead of white
    trues = outline_im == 1
    falses = outline_im == 0
    clear = [0,0.,0.,0]
    black = [1,1.,1.,255]
    new_im = np.empty([outline_im.shape[0],outline_im.shape[1],4])
    new_im[trues] = black
    new_im[falses] = clear

    return new_im, outline_trace

def peth_by_allen_area(event_frames, U,SVT, allen_idx, allen_atlas, window=[-10,60], smoothing=1,channel=0):
    """
     This function creates a PETH for a given allen region
     Inputs:
         event_frames: (numpy array) of the frames at which your event occurred
         frames: (numpy array) of the video you want to create peths for, nframes
                 x nchannels x Wpixels x Hpixels or nframes x wPixels x Hpixels.
                 MUST BE TRANSFORMED TO ALLEN SPACE!!!
         allen_idx: (int) the integer key for an area from ccf_regions obtained with the
                     allen_load_reference() function corresponding to one brain region, or a list of two regions
         allen_atlas: (numpy array) the transformed atlas returned by 
                      atlas_from_landmarks_file(<landmarks_file>,do_transform=True)
         window: the window in frames around which to make the peth
         smoothing: the size of the gaussian kernal used for temporal smoothing
    Outputs:
        channel 0 peth for this region as a 1D numpy array the length of the window size
        channel 1 peth for this region as a 1D numpy array the length of the window size
        allen acronym
    """
    assert atlas.shape == U.shape[:2], 'atlas is not transformed to match the images, use: atlas_from_landmarks_file(<landmarks_file>,do_transform=True)'
    from scipy.ndimage import gaussian_filter1d

    if type(allen_idx)==int or type(allen_idx) == np.int64:
        mask = allen_atlas == allen_idx
    elif len(allen_idx) ==2:
        mask = np.logical_or(allen_atlas == allen_idx[0],allen_atlas == allen_idx[1])
    x,peth, std = whole_brain_peth(event_frames,U,SVT,window=window,smoothing=None)
    mean_peth = peth[:,mask].mean(axis=1) #take a spatial average across all pixels in the area
    mean_std = std[:,mask].mean(axis=1)
    if smoothing:
        mean_peth = gaussian_filter1d(mean_peth,smoothing)
    return x, mean_peth, mean_std

def whole_brain_peth(event_frames, U,SVT, window=[-10,30],smoothing=[1,1,1]):
    """
    Creates a video PETH across the whole dorsal cortex for a given event, aligned to the allen atlas
    
    Inputs:
        event_frames: a numpy array of the frames on which events occurred
        U: a numpy array of Wpixels x Hpixels x nComponents from SVD
        SVT: a numpy array of nComponents x nFrames from SVD
        window: the window in number of frames around an event for which to make the peth 
        smoothing: list length 3, of sigmas for the kernal in number of frames, Wpixels, Hpixels 
                   to apply 3d gaussian smoothing, if None, then no smoothing occurs
        
    """
    from scipy.ndimage import gaussian_filter

    if max(event_frames)/SVT.shape[1]>1.9:
        print('looks like this is data from one channel, downsampling the frames to match')
        event_frames = np.asarray(event_frames/2).astype(np.int64)
    assert len(U.shape)==3, 'U does not have the right dimensions, must be nframes x Wpixels x Hpixels'
    
    cnt=0
    runsum = np.zeros((200,window[1]-window[0]))
    runsq = 0
    for event in event_frames:
        event = int(event)
        start = event + window[0]
        end = event + window[1]
        if end > SVT.shape[1]:
            print('data ended too soon, padding with nans for the right shape')#pad it with nans if there isnt enough data
            pad = np.empty((SVT.shape[0],end-SVT.shape[1]))
            pad[:] = np.nan
            SVT = np.concatenate((SVT,pad),axis=1)
        section = SVT[:,start:end]
        if cnt == 0: #on first loop the first mean is just the first sample
            m = section
        else: #update according to  Mk = M(k-1) + (Xk-M(k-1))/k
            m = m + (section-m)/cnt
        runsum+= section
        runsq += np.square(section)
        cnt+=1
    m=runsum/cnt
    reconstruction = reconstruct(U,m)
    # reconstruction = reconstruction/cnt
    ste = reconstruct(U,np.sqrt(runsq/cnt - np.square(runsum/cnt))/np.sqrt(len(event_frames)))
    if smoothing:
        reconstruction = gaussian_filter(reconstruction,smoothing)
    x = np.arange(window[0],window[1])
    return x, reconstruction, ste

def psth_from_df(U,SVT,behavior,sync_behavior,atlas, area, ccf_regions,localdisk,ax=plt.gca(), events='stimOn_times', split_on='contrastRight',window=[-20,60],smoothing=[1,1,1]):
    """
    Make and plot psth by calling a column from behavior to plot different lines for each condition
    in that column. best used with the contrasts, probabilities, and choices, though you can also 
    create new columns such as reaction_time >1s.
    inputs: 
        U,SVT: spatial and temporal components
        behavior: df with len(num_trials) and different columns witih trial info
        sync_behavior: the sync dataframe to go between task time and frames
        atlas: 540 x 640 allen_atlas to index into with the area parameter
        area: the integer label for the area you want to select in the atlas
        ccf_regions: the df with regions and labels to draw the area name from.
        events: the events to align the PSTH to
        split_on: a string of the column name for which to plot a different line for each unique value
        window: the extent of the PSTH around the event
        smoothing: the 3d parameters for the gaussian kernal used to smooth the PSTH
    Outputs:
        x: the range of times to use for plotting
        psth: an array of len(np.unique(split_on)) of the resulting psths
        ste: an array of len(psth) with the standard error for each psth
    """

    uniques = np.unique(behavior[split_on])
    frameDF = time_to_frameDF(behavior,sync_behavior,localdisk)
    frameDF[frameDF==0]=np.nan
    psths=[]
    stes=[]
    #loop over unique values
    for u in tqdm(uniques):
        event_frames = np.array(frameDF[behavior[split_on] == u][events])
        x,psth,ste = whole_brain_peth(event_frames,U,SVT,window=window,smoothing=smoothing)
        psths.append(psth)
        stes.append(ste)
    #loop to plot each psth
    # fig,ax = plt.subplots(1,1,constrained_layout=True)
    norm = mplc.Normalize(vmin=0,vmax=len(psths))
    clist = []
    labels = []
    for cnt in range(len(psths)):
        color = cm.tab20(norm(cnt))
        temp_psth = psths[cnt][:,atlas==area].mean(axis=1)
        temp_ste = stes[cnt][:,atlas==area].mean(axis=1)
        ax.plot(x,temp_psth,color=color)
        ax.fill_between(x,temp_psth+temp_ste,temp_psth-temp_ste,color=color,alpha=.5)
        clist.append(color)
        labels.append(split_on + ' = ' + str(uniques[cnt]))

    make_legend(ax,clist,labels,location='upper right',bbox_to_anchor=(.99,.99))
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_xticks([0,15,30,45,60])
    ax.set_xticklabels([0,.5,1,1.5,2])
    ax.axvline(0,0,1,color='k')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('df/f')
    ax.set_ylim(-.01,.03)
    if area > 0:
        ax.set_title(ccf_regions[ccf_regions.label==abs(area)]['acronym'].iloc[0] + ' left')
    elif area < 0:
        ax.set_title(ccf_regions[ccf_regions.label==abs(area)]['acronym'].iloc[0] + ' right')
    return x, psths, stes

def plot_bi_psth(U,SVT,events,allen_area,allen_atlas,ax=plt.gca()):
    x,peth_l,std_l = peth_by_allen_area(events,U,SVT,allen_area.label, allen_atlas,window=[-10,60])
    x,peth_r,std_r = peth_by_allen_area(events,U,SVT,allen_area.label*-1, allen_atlas,window=[-10,60])
    steR = std_r
    steL = std_l

    ax.plot(x,peth_l,'b')
    ax.fill_between(x,steL+peth_l, peth_l-steL, color='b',alpha=.5)
    ax.plot(x,peth_r,'r')
    ax.fill_between(x,steR+peth_r, peth_r-steR, color='r',alpha=.5)

    ax.set_xlim(np.min(x),np.max(x))
    ax.set_xticks([0,15,30,45,60])
    ax.set_xticklabels([0,.5,1,1.5,2])
    ax.axvline(0,0,1,color='k')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('df/f')

    plt.show(block=False)

def plot_multi_psth(ax,x, psth, ste, atlas, area_list,ccf_regions,format_axis=True):
    '''
    Takes the output of whole_brain_peth and plots psth traces for a subset of areas.
    x,psth,std: the output of whole_brain_peth
    atlas: (numpy array) the transformed atlas returned by 
                      atlas_from_landmarks_file(<landmarks_file>,do_transform=True)
    area_list: the list off integer area codes to index into the atlas
    '''
    
    # fig,ax = plt.subplots(1,1)
    norm = mplc.Normalize(vmin=0,vmax=len(area_list))
    cnt=0
    clist = []
    labels = []
    for area in area_list:
        color = cm.tab20(norm(cnt))
        temp_df = psth[:,atlas==area].mean(axis=1)
        temp_ste = ste[:,atlas==area].mean(axis=1)
        ax.plot(x,temp_df,color=color)
        ax.fill_between(x,temp_df+temp_ste,temp_df-temp_ste,color=color,alpha=.5)
        clist.append(color)
        if area > 0:
            labels.append(ccf_regions[ccf_regions.label==abs(area)]['acronym'].iloc[0] + ' left')
        elif area < 0:
            labels.append(ccf_regions[ccf_regions.label==abs(area)]['acronym'].iloc[0] + ' right')
        cnt+=1
    if format_axis:
        make_legend(ax,clist,labels, location='upper right',bbox_to_anchor=(.99,.99))
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_xticks([0,15,30,45,60])
        ax.set_xticklabels([0,.5,1,1.5,2])
        ax.axvline(0,0,1,color='k')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('df/f')
        # plt.show(block=False)


def plot_wbpeth(x, psth, transform, outline, show_frames = [-1,1,3,5,7,9]):
    """
    plot time slices of the whole brain brain PSTH with the allen outline
    """
    trans_psth = psth[:]
    for frame in range(psth.shape[0]):
        trans_psth[frame] = im_apply_transform(trans_psth[frame],transform)
    # brain_mask = im_apply_transform(brain_mask,transform).astype(np.float64)

    fig,axs = plt.subplots(nrows=1, ncols=len(show_frames), figsize=[30,60])
    for idx in range(len(show_frames)):
        ax=axs[idx]
        psth_norm = trans_psth/np.max(trans_psth[show_frames[idx]])
        cmap = cm.get_cmap(name='binary')
        cmap.set_bad(color='white')
        
        ax.imshow(psth_norm[show_frames[idx],:,:])
        ax.imshow(outline,cmap = 'Greys')
        ax.axis("off")
        ax.set_title('t={}'.format(round(show_frames[idx]/30,2)))
    plt.show(block=False)

def functional_connectivity(U, SVT, atlas, spont_frames, area_names):
    """
    Calculate the functional connectivity matrix between all brain areas and plot it as a 
    correlation matrix.
    Inputs: 
        U: the spatial components
        SVT: temporal components, should use the hemodynamically corrected version
        atlas: the overlay containing indexing for each brain area, gotten with the 
               wfield.atlas_from_landmarks_file(...,...,do_transform=True) function
               ***CRITICAL*** must be transformed to the image space with do_transform=True
        spont_frames: a ndarray of the frames from which to draw spontaneous activity
        area_names: the list containing name-integer pairs labeling each atlas area with its
                    corresponding allen name
    Outputs:
        a brain areas x brain areas matrix of the spontaneous correlation between areas
    """
    # need to break up spont_frames because this will be too big to reconstruct video for
    # all these spontaneous frames
    slices = np.array_split(spont_frames,100)
    u_areas=np.unique(atlas)
    frame_cnt0 = 0
    frame_cnt1 = 0
    normal = len(slices[0])
    # initialize dictionary
    area_activities = np.empty([len(spont_frames),len(u_areas)])
    # for area in np.unique(atlas):
    #     area_activities[area] = np.array([])
    print('Reconstructing the spontaneous activity from the SVD...')
    for frames in tqdm(slices):
        spont_stack = reconstruct(U,SVT[:,frames.astype(np.int64)])
        # if spont_stack.shape[0] < normal:
        #     pad = np.empty([normal-spont_stack.shape[0],540,640])
        #     pad[:] = np.nan
        #     spont_stack = np.concatenate([spont_stack,pad])
        frame_cnt1+=spont_stack.shape[0]
        for area in range(len(u_areas)):
            area_activities[frame_cnt0:frame_cnt1,area] = np.nanmean(spont_stack[:,atlas==u_areas[area]],axis=1)
        frame_cnt0+=spont_stack.shape[0]

    corr_mat=np.corrcoef(area_activities.T)
    disp_order = [1,-1,-2,-3,-4,4,3,2,5,6,7,-7,-6,-5,-8,-9,-10,-11,11,10,9,8,13,14,15,16,17,18,-18,-17,-16,-15,-14,-13,-19,19,20,21,22,23,-23,-22,-21,-20, 24,25,26,27,28,29,30,31,32,33,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24]
    for i in disp_order:
        if i <0:
            i=i*-1+16
        test.append(i)
         

def get_spont_frames(SVT, behavior,sync_behavior, use_end=False):
    """
    Fetch the frames in which 'spontaneous' activity is going on, to be used as an input to the 
    functional_connectivity function. By default this will take the frames that occur during the 
    inter-trial intervals, however if use_end=True it will take the time after the last behavior
    timestamp as spontaneous activity.
    Intputs: 
        SVT: the temporal components so we know how many frames there are overall
        behavior: a pandas dataframe containing all relavent behavior information from the 
        fetch_task_data function.
        sync_behavior: the pandas dataframe containing frames, timestamps, sync upfronts, and 
        aligned bpod timestamps retrieved with the sync_to_task function
        use_end: a boolean to determine if spontaneous activity should be drawn from the end of the
        session rather than the ITIs
    Outputs:
        a ndarray of frames to use as spontaneous activity

    """
    all_frames = np.arange(SVT.shape[1])
    if use_end:  # return all frames after the last frame there was a behavior TTL
        return all_frames[sync_behavior['frame'][np.argmax(sync_behavior.task_time)]:]
    ITI_starts = behavior.stimOff_times[:-1]
    ITI_starts = time_to_frames(sync_behavior, localdisk, np.array(ITI_starts))
    ITI_ends = behavior.stimOn_times[1:]
    ITI_ends = time_to_frames(sync_behavior, localdisk, np.array(ITI_ends))
    # The task seems pretty buggy in that there are several very tiny ITIs (>.1s), throw those out
    use_ITIs = ITI_ends-ITI_starts > 10
    ITI_starts = ITI_starts[use_ITIs].astype(np.int64)
    ITI_ends = ITI_ends[use_ITIs].astype(np.int64)
    use_frames = np.array([])
    for i in range(len(ITI_ends)):
        use_frames = np.append(use_frames,all_frames[ITI_starts[i]:ITI_ends[i]])
    return use_frames.astype(np.int64)

def make_legend(ax,colors,labels, line_width=4,location='upper left', bbox_to_anchor=(1.01, 1)):
    """
    makes a custom legend from a given list of colors (or colormap) and labels
    Inputs: 
        ax: axis to plot lengend onto
        colors: either an interible of colors, or a colormap
        labels: a list of strings containing the labels for the legend
    """
    import matplotlib.colors as c
    from matplotlib.lines import Line2D
    assert len(colors) == len(labels) or type(colors) == c.ListedColormap, 'the length of labels and colors ust be the same'
    if type(colors) == c.ListedColormap:
        print('TO DO: implement colormap mapping')
        return None
    lines = []
    for color in colors:
        lines.append(Line2D([0], [0], color=color, lw=line_width))
    ax.legend(lines,labels, bbox_to_anchor=bbox_to_anchor, loc=location)

def time_to_frameDF(behavior,sync_behavior,localdisk):
    """
    Makes a dataframe that has all the timing events converted to frames
    Inputs: 
        behavior: df with len(num_trials) and different columns witih trial info
        sync_behavior: the sync dataframe to go between task time and frames
        localdisk: the directory where the above two are saved for this session
    outputs:
        frameDF: a dataframe with all the behavioral events that end with _times, that are instead
        aligned to the camera and displayed as frames
    """
    mask = ['times' in key for key in behavior.keys()]
    time_df = behavior[behavior.keys()[mask]]
    frame_df = pd.DataFrame(columns=time_df.keys())
    for (columnName, columnData) in time_df.iteritems():
        frame_df[columnName] = time_to_frames(sync_behavior,localdisk,np.array(columnData),dropna=False)
    frameDF = frame_df.astype(np.int64)
    frameDF[frameDF==0]=np.nan

    return frameDF

def plot_trace_sample(fig,ax,U, SVT, behavior, sync_behavior, allen_area, atlas,localdisk, plot_sec=15):
    # first get all the frames for all the events and store inf frame_df
    use_trials = behavior.iloc[:90]
    frame_df = time_to_frameDF(use_trials,sync_behavior,localdisk)
    # for the first plot, get the first 10 seconds, and the events that happen then
    first_event = frame_df['stimOn_times'][0]
    first_10 = reconstruct(U,SVT[:,first_event:first_event+plot_sec*30]) # ten seconds at 30fps
    first_events = frame_df[frame_df <= first_event+plot_sec*30] - first_event
    first_events.drop(['goCue_times','goCueTrigger_times', 'feedback_times'],axis='columns',inplace=True)

    x = np.arange(plot_sec*30)
    area_r = allen_area.label*-1
    area_l = allen_area.label
    first_10_l = first_10[:,atlas==area_l].mean(axis=1)
    first_10_r = first_10[:,atlas==area_r].mean(axis=1)
    
    norm = mplc.Normalize(vmin=0,vmax=len(area_list))
    color1 = cm.tab20(norm(0))
    color2 = cm.tab20(norm(1))
    fig.suptitle(allen_area.acronym)
    ax.plot(x/30,first_10_l,color=color1)
    ax.plot(x/30,first_10_r,color=color2)

    event_colors = ['c','g','orange','k']
    cnt=0
    for (event,times) in first_events.iteritems():
        ax.vlines(times/30,np.min(first_10_r),np.max(first_10_l),colors = event_colors[cnt])
        cnt+=1

    colors = event_colors + [color1,color2]
    labels = first_events.keys().tolist()+['Right_hem','Left_hem']
    make_legend(ax,colors,labels)
    # ax.set_xlabel('time (s)')
    ax.set_ylabel('df/f')
    ax.set_xlim(0,plot_sec)
    # return np.vstack((first_10_l,first_10_r), first_events


def plot_trial_sample(fig, ax, U, SVT, behavior, sync_behavior, allen_area, allen_atlas, localdisk, trial_choice=np.arange(90)):
    
    use_behavior = behavior.iloc[trial_choice]
    mask = ['times' in key for key in use_behavior.keys()]
    time_df = use_behavior[use_behavior.keys()[mask]]
    frame_df = pd.DataFrame(columns=time_df.keys())
    for (columnName, columnData) in time_df.iteritems():
        frame_df[columnName] = time_to_frames(sync_behavior,localdisk,np.array(columnData),dropna=False)
    frame_df = frame_df.astype(np.int64)
    stimOns = np.array(frame_df.stimOn_times)
    trials = np.empty([len(frame_df),70])
    for i in tqdm(range(len(stimOns))):
        stim = stimOns[i]
        trial_im = reconstruct(U,SVT[:,stim-10:stim+60])
        trials[i,:] = trial_im[:,allen_atlas==allen_area.label].mean(axis=1)
    ax = sns.heatmap(trials,cmap='viridis',cbar=False,square=False)
    # norm = mplc.Normalize(vmin=np.min(trials),vmax=np.max(trials))

    # cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
    # cb.set_label('df/f')
    ax.axvline(10,0,90,color='k')
    ax.set_xticks(np.arange(0,70,10))
    ax.set_xlim(0,70)
    ax.set_xticklabels(np.around(np.arange(-10,60,10)/30,decimals=2))
    # ax.set_xlabel('frames @30fps')
    ax.set_ylabel('trials')
    return trials


def plot_signal_summary(fig,U, SVT, behavior, sync_behavior, allen_area, allen_atlas,localdisk,ccf_regions):
    """
    plot a summary of the signal through stages of processing for a single brain area. First a few
    traces of raw df/f, then some trials, and an average trace.
    Inputs:
        U: spatial components
        SVT: hemodynamically corrected temporal components
        behavior: dataframe with behavioral events and timestamps of len(nTrials)
        sync_behavior: dataframe with the sync information btw times and frames
        allen_area: pandas series taking one row of the ccf_regions dataframe
        allen_atlas: 540x640 array with each area marked by its unique label
    """
    # fig = plt.figure(constrained_layout=True)
    use_behavior = behavior.iloc[np.arange(90)]
    mask = ['times' in key for key in use_behavior.keys()]
    time_df = use_behavior[use_behavior.keys()[mask]]
    frame_df = pd.DataFrame(columns=time_df.keys())
    for (columnName, columnData) in time_df.iteritems():
        frame_df[columnName] = time_to_frames(sync_behavior,localdisk,np.array(columnData),dropna=False)
    frame_df = frame_df.astype(np.int64)
    stimOns = np.array(frame_df.stimOn_times)

    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    print('plotting a sample of the raw trace...')
    plot_trace_sample(fig,ax1,U, SVT, behavior, sync_behavior, allen_area, allen_atlas,localdisk)
    print('plotting the first 90 trials...')
    plot_trial_sample(fig,ax2,U, SVT, behavior, sync_behavior, allen_area, allen_atlas,localdisk)
    print('plotting the PSTHs...')
    x,psth,ste = whole_brain_peth(stimOns,U,SVT,window=[-10,60])
    plot_multi_psth(ax3,x,psth,ste,allen_atlas,[allen_area.label,allen_area.label*-1],ccf_regions)
    # plt.show(block=False)

def time_lag_corr(data_x, data_y, lag):
    """
    Calculate the correlation between two numpy arrays shifted by an integer lag
    inputs: 
        data_x: ndarray must be same shape as data_y, and the first dimension must be time
        data_y: ndarray must be same shape as data_x, and the first dimension must be time
        lag: integer amount of lag between the two, reccomended lag <= window/2
    outputs:
        correlation between the two arrays, lagnged at the set number of frames

    """
    # first pad the data_y before it is rolled
    if lag==0:
        return np.corrcoef(data_x,data_y) 
    if len(data_x.shape) == 3:
        pad = np.empty(lag,shape(data_x)[1],shape(data_x)[2])
    else:
        pad = np.zeros(abs(lag))
        # pad[:] = np.nan
    if lag > 0:
        padded = np.concatenate([pad, data_y.T])
        rolled = padded[:len(data_x)]
    elif lag < 0:
        padded = np.concatenate([data_y.T, pad])
        rolled = padded[abs(lag):]

    return np.corrcoef(data_x.T,rolled)

def windowed_lag_corr(U,SVT,mask_x, mask_y, events=[], roll_range=[-60,60]):
    """
    time shifted correlation analysis to see which of the two areas leads the interaction
    takes activity from the two areas x and y, and holds x still while shifting y from 
    roll_range[0]/2 to roll_range[1]/2, calculating a pearson's R at each time lag, one frame at a
    time. there are no edge effects because the roll_range is twice the size of the window for 
    which correlations are actually calculated. you can optionally specify events to have zero 
    aligned to a specific task event.

    Negative lag means that area_y leads, positive lag means that area_x leads

    inputs:
        U: spatial components
        SVT: temporal components
        mask_x: the 540x640 boolean mask for the first area to use activity from
        mask_y: the 540x640 boolean mask for the second area to use activity from
        events: default is empty, or no events, to use add a 1d array of frame integers
        roll_range: the range around the event (or 0) to load data, is 2x the length of the corr 
                    array that is returned
    """
    all_corrs = []
    # if len(events) == 0:
    #     data_end = SVT.shape[-1]
    #     roll_end = int(roll_range[1]+roll_range[1]/2)
    #     roll_start = 0

    #     load_frames = SVT[:,roll_start:roll_end]

    #     corrs = [time_lag_corr(load_frames,load_frames,lag)[0,1] for lag in np.arange(roll_range[0],roll_range[1])]
    #     all_corrs.append(corrs[int(roll_range[1]/2):int(roll_range[1]+roll_range[1]/2)])
    #     roll_start = int(roll_end - roll_range[1])
    #     roll_end += int(roll_range[1])
        
    # else:
    for event in tqdm(events):
        roll_start = event + roll_range[0]
        roll_end = event + roll_range[1]
        load_frames = SVT[:,roll_start:roll_end]
        frames_x = load_frames[:,mask_x].mean(axis=1)
        frames_y = load_frames[:,mask_y].mean(axis=1)
        corrs = [time_lag_corr(frames_x,frames_y,lag)[0,1] for lag in np.arange(roll_range[0],roll_range[1])]
        all_corrs.append(corrs[int(roll_range[1]/2):int(roll_range[1]+roll_range[1]/2)])
    return np.asarray(all_corrs)


def show():
    plt.show(block=False)

def plot_window_lag(corr_im, area1_label, area2_label,data_range=60):
    from scipy.stats import skew, skewtest
    fig,(ax1,ax2) = plt.subplots(1,2,constrained_layout=True)
    meanc = corr_im.mean(axis=0)
    skewness = skew(meanc)
    zstat, p = skewtest(meanc)
    ax2.plot(np.arange(corr_im.shape[1]),meanc)
    ax2.set_xticks(np.linspace(0,data_range,5))
    ax2.set_xticklabels(np.linspace((data_range-data_range*1.5)/30,(data_range-data_range*.5)/30,5))
    ax2.set_xlabel('lag (s)')
    ax2.text(.5,.8,'skew = {},\np = {}'.format(np.round(skewness,2),np.round(p,2)))
    ax2.axvline(np.mean((meanc[-1:29:-1] - meanc[:30])*30+30),0,1,color='k')

    plt.sca(ax1)
    plt.imshow(corr_im)
    cbar = plt.colorbar(ax=ax1)
    cbar.set_label("Pearson's r")
    ax1.set_xticks(np.linspace(0,data_range,5))
    ax1.set_xticklabels(np.linspace((data_range-data_range*1.5)/30,(data_range-data_range*.5)/30,5))
    ax1.set_xlabel('lag (s)')
    ax1.set_ylabel('Trial number')
    ax1.set_title(' {} leads <> {} leads'.format(area2_label,area1_label))
    show()

