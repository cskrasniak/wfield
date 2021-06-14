### messy temporary file to pilot analyses
### Chris Krasniak 2021-01-28

from labcams import parse_cam_log
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from  wfield import *
import imageio
import math
from matplotlib import cm, colorbar
import matplotlib.colors as mplc
from matplotlib.lines import Line2D
import seaborn as sns
from analyses import *

## Subjects to use
subjects = ['Gcamp_test1','CSK-im-001','CSK-im-002']
baseDir = r'F:\imaging_data'
## iterate through subjects
for subject in subjects:
    os.chdir(pjoin(baseDir,subject))
    dates = os.listdir()
    # create figure for each subject's PSTHs
    psth_fig,axs = plt.subplots(nrows=len(dates)-1,ncols=2,figsize=(10,8))
    psth_fig.suptitle(subject+' PSTHs')
    ## iterate through dates for each subject
    for idx,date in tqdm(enumerate(dates[:-1])):
        os.chdir(pjoin(baseDir,subject))
        localdisk = pjoin(os.getcwd(),date)
        os.chdir(localdisk)

        ## Check if the session has been preprocessed, and if behavior data has been downloaded
        if not os.path.isfile('SVTcorr.npy'):
            print('{} \nThis session has not yet been preprocessed, skipping'.format(localdisk))
            continue
        alf_folder = pjoin('G:\\FlatIron\\zadorlab\\Subjects\\',subject,date,'001','alf')
        if not os.path.isfile(pjoin(alf_folder,'_ibl_trials.stimOn_times.npy')):
            print('{} \nNo alf data, try running fetchONE to get it, or check extraction'.format(alf_folder))
            continue
        behavior = fetch_task_data(subject,date)
        behavior = behavior[behavior['choice']!=0].reset_index() #drop trials where there was no response
        U = np.load(pjoin(localdisk,'U.npy'))# load spatial components
        SVTcorr = np.load(pjoin(localdisk,'SVTcorr.npy'))# load hemodynamic corrected temporal components
        SVT = np.load(pjoin(localdisk,'SVT.npy'))# load  normal temporal components
        onset_times = extract_onset_times(localdisk)

        # the allen atlas and brain mask for finding which pixels corresponds to which area
        atlas, area_names, brain_mask = atlas_from_landmarks_file(pjoin(localdisk,'dorsal_cortex_landmarks.JSON'),do_transform=True)
        ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

        #the transform to apply to the images
        transform = load_allen_landmarks(pjoin(localdisk, 'dorsal_cortex_landmarks.JSON'))['transform']
        outline_im, outline = make_allen_outline(atlas, allen_transform=transform)
        lmarks = load_allen_landmarks(pjoin(localdisk, 'dorsal_cortex_landmarks.JSON'))
        nref_regions = allen_transform_regions(None,ccf_regions,
                                       resolution=lmarks['resolution'],
                                       bregma_offset=lmarks['bregma_offset'])
                
        stack = SVDStack(U,SVTcorr)
        stack.set_warped(True,M = lmarks['transform'])
        # mask the pixels outside of the brain outline.
        from wfield.imutils import mask_to_3d
        # transform the brain outline to image coordinates
        b_outline = brain_outline/lmarks['resolution'] + np.array(lmarks['bregma_offset'])
        mask = contour_to_mask(*b_outline.T,dims = U.shape[:-1])
        #create a 3d mask of the brain outline
        mask3d = mask_to_3d(mask,shape = np.roll(stack.U_warped.shape,1))
        # set pixels outside the brain outline to zero
        stack.U_warped[~mask3d.transpose([1,2,0])] = 0   

        #sync up the behavior and grab a few things for analysis
        sync_behavior = sync_to_task(localdisk)
        mask = ['times' in key for key in behavior.keys()]
        time_df = behavior[behavior.keys()[mask]]
        frame_df = pd.DataFrame(columns=time_df.keys())
        for (columnName, columnData) in time_df.iteritems():
            frame_df[columnName] = time_to_frames(sync_behavior,localdisk,np.array(columnData),dropna=False)
        frame_df = frame_df.astype(np.int64)
        stimOns = np.array(frame_df.stimOn_times)
        strong_r = behavior[behavior['signedContrast']>.4]['stimOn_times'] #positive values for right stimuli
        strong_r_frames = time_to_frames(sync_behavior,localdisk,np.array(strong_r))
        lick100 = time_to_frames(sync_behavior,localdisk,np.array(behavior['response_times'].iloc[:100]))
        correct = time_to_frames(sync_behavior,localdisk,np.array(behavior.loc[behavior['feedbackType'] == 1,'feedback_times']))
        error = time_to_frames(sync_behavior,localdisk,np.array(behavior.loc[behavior['feedbackType'] == -1,'feedback_times']))

        strong_l = behavior[behavior['signedContrast']<-.4]['stimOn_times']
        strong_l_frames = time_to_frames(sync_behavior,localdisk,np.array(strong_l))
        # Use unique to exclude the default value of frames that is assigned for nan values
        first_move_l = np.unique(time_to_frames(sync_behavior, localdisk, np.array(behavior[behavior['choice']==-1]['firstMovement_times'])))
        first_move_r = np.unique(time_to_frames(sync_behavior, localdisk, np.array(behavior[behavior['choice']==1]['firstMovement_times'])))
        l_block_stim = np.unique(time_to_frames(sync_behavior, localdisk, np.array(behavior[behavior['probabilityLeft'] >.5]['stimOn_times'])))
        r_block_stim = np.unique(time_to_frames(sync_behavior, localdisk, np.array(behavior[behavior['probabilityLeft'] <.5]['stimOn_times'])))
        unbiased_stim = np.unique(time_to_frames(sync_behavior, localdisk, np.array(behavior[behavior['probabilityLeft'] ==.5]['stimOn_times'])))

        surprise_r = np.array(behavior[np.logical_and(behavior['probabilityLeft'] <.5, behavior['signedContrast'] < 0)]['stimOn_times'])
        sur_r_frames = np.unique(time_to_frames(sync_behavior, localdisk,surprise_r))
        surprise_l = np.array(behavior[np.logical_and(behavior['probabilityLeft'] >.5, behavior['signedContrast'] > 0)]['stimOn_times'])
        sur_l_frames = np.unique(time_to_frames(sync_behavior, localdisk,surprise_l))
        unsurprise_r = np.array(behavior[np.logical_and(behavior['probabilityLeft'] <.5, behavior['signedContrast'] > 0)]['stimOn_times'])[-len(surprise_r):]
        unsur_r_frames = np.unique(time_to_frames(sync_behavior, localdisk,unsurprise_r))
        unsurprise_l = np.array(behavior[np.logical_and(behavior['probabilityLeft'] >.5, behavior['signedContrast'] < 0)]['stimOn_times'])[-len(surprise_l):]
        unsur_l_frames = np.unique(time_to_frames(sync_behavior, localdisk,unsurprise_l))

        # ## make PSTHs
        # ax = axs[idx,0]
        # x,stim_psth,ste = whole_brain_peth(strong_l_frames,stack.U_warped,SVTcorr,window=[-20,60])
        # plot_multi_psth(ax,x,stim_psth,ste,atlas,[33,-33],ccf_regions)
        
        # ax = axs[idx,1]
        # x1, move_psth,ste1 = whole_brain_peth(first_move_l,stack.U_warped,SVTcorr,window=[-20,60])
        # plot_multi_psth(ax,x1,move_psth,ste1,atlas,[-3,3,-18,18],ccf_regions)
        # # sum_fig = plt.figure()
        # # plot_signal_summary(sum_fig,stack.U_warped,SVT[:,1::2],behavior,sync_behavior, ccf_regions.iloc[30], atlas,
        # #                     localdisk,ccf_regions)
        # # sum_fig.savefig('uncorr_summary.pdf')
        sum_fig2 = plt.figure()
        plot_signal_summary(sum_fig2,stack.U_warped,SVTcorr,behavior,sync_behavior, ccf_regions.iloc[30], atlas,
                            localdisk,ccf_regions)
        sum_fig2.savefig('corr_summary.pdf')

        #contrast dependent psth in visp
        fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(8,14))
        ax = axs[0,0]
        psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, 33, ccf_regions,localdisk,ax=ax,split_on='contrastRight')
        ax.axes.get_xaxis().set_visible(False)
        ax.get_legend().remove()
        ax = axs[1,0]
        psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, -33, ccf_regions,localdisk,ax=ax,split_on='contrastLeft')
        ax.get_legend().remove()
        ax = axs[0,1]
        psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, 33, ccf_regions,localdisk,ax=ax,split_on='contrastLeft')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax = axs[1,1]
        psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, -33, ccf_regions,localdisk,ax=ax,split_on='contrastRight')
        ax.axes.get_yaxis().set_visible(False)
        fig.savefig('VISp_contrast_dependence')
        show()
    os.chdir(pjoin(baseDir,subject))
    psth_fig.savefig('PSTHs.pdf')

    show()


## Block analyses
x,un_psth,un_ste = whole_brain_peth(unbiased_stim,stack.U_warped,SVTcorr,window=[-20,60])
x,r_psth,r_ste = whole_brain_peth(r_block_stim,stack.U_warped,SVTcorr,window=[-20,60])
x,l_psth,l_ste = whole_brain_peth(l_block_stim,stack.U_warped,SVTcorr,window=[-20,60])
fig = plt.figure()
ax=fig.add_subplot(2,1,1)
ax.set_title('')
plot_multi_psth(ax,x,r_psth-un_psth,np.mean([r_ste,un_ste],axis=0),atlas,[33,31,3,4,18],ccf_regions)
ax=fig.add_subplot(2,1,2)
plot_multi_psth(ax,x,l_psth-un_psth,np.mean([l_ste,un_ste],axis=0),atlas,[33,31,3,4,18],ccf_regions)

show()

x,sur_r_psth,sur_r_ste = whole_brain_peth(sur_r_frames,stack.U_warped,SVTcorr,window=[-20,60])
x,un_r_psth,un_r_ste = whole_brain_peth(unsur_r_frames,stack.U_warped,SVTcorr,window=[-20,60])

x,sur_l_psth,sur_l_ste = whole_brain_peth(sur_l_frames,stack.U_warped,SVTcorr,window=[-20,60])
x,un_l_psth,un_l_ste = whole_brain_peth(unsur_l_frames,stack.U_warped,SVTcorr,window=[-20,60])
fig = plt.figure()
ax=fig.add_subplot(2,1,1)
ax.set_title('surprise L - expected L stim')
plot_multi_psth(ax,x,sur_l_psth-un_l_psth,np.mean([sur_l_ste,un_l_ste],axis=0),atlas,[33,31,30],ccf_regions)
ax=fig.add_subplot(2,1,2)
ax.set_title('surprise R - expected R stim')
plot_multi_psth(ax,x,sur_r_psth-un_r_psth,np.mean([sur_r_ste,un_l_ste],axis=0),atlas,[33,31,30],ccf_regions)
show()

## correct vs incorrect PSTH
x,correct_psth,correct_ste = whole_brain_peth(correct,stack.U_warped,SVTcorr,window=[-20,60])
x,error_psth,error_ste = whole_brain_peth(error,U,SVTcorr,window=[-20,60])
fig = plt.figure()
ax=fig.add_subplot(2,1,1)
ax.set_title('U_warped correct PSTH')
plot_multi_psth(ax,x,correct_psth,correct_ste,atlas,[20,21,31],ccf_regions)
ax=fig.add_subplot(2,1,2)
x,correct_psth,correct_ste = whole_brain_peth(correct,U,SVTcorr,window=[-20,60])

ax.set_title('U correct PSTH')
plot_multi_psth(ax,x,correct_psth,correct_ste,atlas,[20,21,31],ccf_regions)
show()

fig=plt.figure('first 25 U components')
cnt=1
for i in range(25):
    ax=fig.add_subplot(5,5,cnt)
    ax.imshow(U[:,:,cnt])
    ax.axis('off')
    cnt+=1
fig.suptitle('first 25 U components')
plt.tight_layout()
show()



