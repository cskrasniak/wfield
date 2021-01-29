## Functions to make various movies and gifs

from labcams import parse_cam_log
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from  wfield import *
import imageio
import math
from matplotlib import cm
import matplotlib.colors as mplc
from matplotlib.lines import Line2D
import seaborn as sns
from analyses import *
from sklearn.linear_model import LogisticRegression

subject = 'CSK-im-002'
baseDir = r'F:\imaging_data'
os.chdir(pjoin(baseDir,subject))
dates = os.listdir()
## iterate through dates for each subject
date = '2021-01-17'
#load the camera log
logdata,led,sync,ncomm = parse_cam_log(glob(pjoin(localdisk,'*.camlog'))[0],readTeensy=True)
# load the preprocessed stack
U = U
SVT = SVTcorr
# load the landmarks and the reference
lmarks = load_allen_landmarks(pjoin(localdisk, 'dorsal_cortex_landmarks.JSON'))
ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')
# convert the regions to pixels coordinates

nref_regions = allen_transform_regions(lmarks['transform'],ccf_regions,
                                       resolution=lmarks['resolution'],
                                       bregma_offset=lmarks['bregma_offset'])
# transform U (warp) 
stack = SVDStack(U,SVT)
stack.set_warped(True,M = lmarks['transform'])
# mask the pixels outside of the brain outline.
from wfield.imutils import mask_to_3d
# transform the brain outline to image coordinates
bout = brain_outline/lmarks['resolution'] + np.array(lmarks['bregma_offset'])
mask = contour_to_mask(*bout.T,dims = U.shape[:-1])
#create a 3d mask of the brain outline
mask3d = mask_to_3d(mask,shape = np.roll(stack.U_warped.shape,1))
# set pixels to zero
stack.U_warped[~mask3d.transpose([1,2,0])] = 0
# interpolate camera times
camtime = np.interp(np.arange(len(stack)),(np.array(led[led.led==4].frame)+1)/2,np.array(led[led.led==4].timestamp))/1000
unique_contrasts=[0,.0625,.125,.25,1]

npre=20
npost=60
x,rPSTHs,ste = psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, 33, ccf_regions,localdisk,ax=ax,split_on='contrastRight')
x, lPSTHs,ste = psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, 33, ccf_regions,localdisk,ax=ax,split_on='contrastLeft')

def stim_on_mov(rPSTHs,lPSTHs, nref_regions, fname=None):
    fig=plt.figure(figsize=[10.88,  3.71])
    fig.add_axes([0,0,1,1])
    dat = []
    for ii in np.arange(len(rPSTHs)):
        xoff = rPSTHs[ii].shape[2]*(ii)
        for i,r in nref_regions.iterrows():
            plt.plot(xoff+r['left_x'],r['left_y'],color='k',lw=0.3)
            plt.plot(xoff+r['right_x'],r['right_y'],color='k',lw=0.3)

            plt.plot(xoff+r['right_x'],np.array(r['right_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)
            plt.plot(xoff+r['left_x'],np.array(r['left_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)

        if ii == 0:
            plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'Contrast Percent',ha='center',va = 'bottom')    
        plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'{0:1.1f}'.format(unique_contrasts[ii]*100),ha='center',va = 'top')
        if ii == len(rPSTHs)-1:
            plt.text(10,rPSTHs[ii].shape[1]*.7,'Right Stimulus', rotation=90)
            plt.text(10,rPSTHs[ii].shape[1]*.7 + rPSTHs[ii].shape[1],'Left Stimulus', rotation=90)
        # tmp = np.concatenate([rPSTHs[ii]-np.mean(rPSTHs[ii][:npre],axis=0),
        #                               lPSTHs[ii]-np.mean(lPSTHs[ii][:npre],axis=0)],axis=1)
        tmp = np.concatenate([rPSTHs[ii], lPSTHs[ii]],axis=1)
        if len(dat)==0:
            dat = tmp.copy()
        else:
            dat = np.concatenate([dat,tmp.copy()],axis=2)
    #make dot for stim on
    dat[npre:npre+5,int(dat.shape[1]/2-20):int(dat.shape[1]/2+20),:] = -np.max(dat)
    tmp[npre:npre+30,:30,:30] = 1
    plt.axis('off')
    if fname is None:
        nb_play_movie(dat,
                      clim=[-0.04,0.04],cmap='RdBu_r')
    else:
        nb_save_movie(dat,filename=fname,
                      clim=[-0.04,0.04],cmap='RdBu_r')

## make movie for right vs left stimulus onset for diff contrast levels
x,rPSTHs,ste = psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, 33, ccf_regions,localdisk,ax=ax,split_on='contrastRight')
x, lPSTHs,ste = psth_from_df(stack.U_warped,SVTcorr,behavior,sync_behavior,atlas, 33, ccf_regions,localdisk,ax=ax,split_on='contrastLeft')
stim_on_mov(rPSTHs,lPSTHs,nref_regions,fname='stim_on.gif')

## make mov for comparing the two blocks and if the stimulus was surprising or expected
prob_left = [.2,.8]
x,sur_r_psth,sur_r_ste = whole_brain_peth(sur_r_frames,stack.U_warped,SVTcorr,window=[-20,60])
x,un_r_psth,un_r_ste = whole_brain_peth(unsur_r_frames,stack.U_warped,SVTcorr,window=[-20,60])

x,sur_l_psth,sur_l_ste = whole_brain_peth(sur_l_frames,stack.U_warped,SVTcorr,window=[-20,60])
x,un_l_psth,un_l_ste = whole_brain_peth(unsur_l_frames,stack.U_warped,SVTcorr,window=[-20,60])
rPSTHs = [un_r_psth,sur_r_psth]
lPSTHs = [sur_l_psth,un_l_psth]
block_diff_mov(rPSTHs,lPSTHs, nref_regions, fname='block_diff.avi')

def block_diff_mov(rPSTHs,lPSTHs, nref_regions, fname=None):
    fig=plt.figure(figsize=[10.88,  3.71])
    fig.add_axes([0,0,1,1])
    dat = []
    for ii in np.arange(len(rPSTHs)):
        xoff = rPSTHs[ii].shape[2]*(ii)
        for i,r in nref_regions.iterrows():
            plt.plot(xoff+r['left_x'],r['left_y'],color='k',lw=0.3)
            plt.plot(xoff+r['right_x'],r['right_y'],color='k',lw=0.3)

            plt.plot(xoff+r['right_x'],np.array(r['right_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)
            plt.plot(xoff+r['left_x'],np.array(r['left_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)
        if ii == 0:
            plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'Probability Left ',ha='center',va = 'bottom')    
        plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'{0:1.1f}'.format(prob_left[ii]),ha='center',va = 'top')
        if ii == len(rPSTHs)-1:
            plt.text(10,rPSTHs[ii].shape[1]*.7,'Right Stimulus', rotation=90)
            plt.text(10,rPSTHs[ii].shape[1]*.7 + rPSTHs[ii].shape[1],'Left Stimulus', rotation=90)
        # tmp = np.concatenate([rPSTHs[ii]-np.mean(rPSTHs[ii][:npre],axis=0),
        #                               lPSTHs[ii]-np.mean(lPSTHs[ii][:npre],axis=0)],axis=1)
        tmp = np.concatenate([rPSTHs[ii], lPSTHs[ii]],axis=1)
        if len(dat)==0:
            dat = tmp.copy()
        else:
            dat = np.concatenate([dat,tmp.copy()],axis=2)
    #make line for stim on
    dat[npre:npre+5,int(dat.shape[1]/2-20):int(dat.shape[1]/2+20),:] = -np.max(dat)
    tmp[npre:npre+30,:30,:30] = 1
    plt.axis('off')
    if fname is None:
        nb_play_movie(dat,
                      clim=[-0.04,0.04],cmap='RdBu_r')
    else:
        nb_save_movie(dat,filename=fname,
                      clim=[-0.04,0.04],cmap='RdBu_r')



def block_diff_sub_mov(rPSTHs,lPSTHs, nref_regions, fname=None):
    fig=plt.figure(figsize=[10.88,  3.71])
    fig.add_axes([0,0,1,1])
    dat = []
    for ii in np.arange(len(rPSTHs)):
        xoff = rPSTHs[ii].shape[2]*(ii)
        for i,r in nref_regions.iterrows():
            plt.plot(xoff+r['left_x'],r['left_y'],color='k',lw=0.3)
            plt.plot(xoff+r['right_x'],r['right_y'],color='k',lw=0.3)

            plt.plot(xoff+r['right_x'],np.array(r['right_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)
            plt.plot(xoff+r['left_x'],np.array(r['left_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)
        if ii == 0:
            plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'Unexpected-expected ',ha='center',va = 'bottom')    
        # plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'{0:1.1f}'.format(prob_left[ii]),ha='center',va = 'top')
        if ii == len(rPSTHs)-1:
            plt.text(10,rPSTHs[ii].shape[1]*.7,'Right Stimulus', rotation=90)
            plt.text(10,rPSTHs[ii].shape[1]*.7 + rPSTHs[ii].shape[1],'Left Stimulus', rotation=90)
        # tmp = np.concatenate([rPSTHs[ii]-np.mean(rPSTHs[ii][:npre],axis=0),
        #                               lPSTHs[ii]-np.mean(lPSTHs[ii][:npre],axis=0)],axis=1)
        tmp = np.concatenate([rPSTHs[ii], lPSTHs[ii]],axis=1)
        if len(dat)==0:
            dat = tmp.copy()
        else:
            dat = np.concatenate([dat,tmp.copy()],axis=2)
    #make line for stim on
    dat[npre:npre+5,int(dat.shape[1]/2-20):int(dat.shape[1]/2+20),:] = -np.max(dat)
    tmp[npre:npre+30,:30,:30] = 1
    plt.axis('off')
    if fname is None:
        nb_play_movie(dat,
                      clim=[-0.04,0.04],cmap='RdBu_r')
    else:
        nb_save_movie(dat,filename=fname,
                      clim=[-0.04,0.04],cmap='RdBu_r')

## make movie to subtract expected from unexpected responses for each stimulus side
rPSTHs = [sur_r_psth-un_r_psth]
lPSTHs = [sur_l_psth-un_l_psth]
block_diff_sub_mov(rPSTHs,lPSTHs, nref_regions, fname='block_diff_sub.avi')

npre = 20
## make a movie for stim side vs feedbacktype, need to be more careful choosing trials here so the contrasts match up
re_mask = (behavior['feedbackType'] == -1) & (behavior['contrastRight']>0)
le_mask = (behavior['feedbackType'] == -1) & (behavior['contrastLeft']>0)
lc_mask = (behavior['feedbackType'] == 1) & (behavior['contrastLeft']>0)
rc_mask = (behavior['feedbackType'] == 1) & (behavior['contrastRight']>0)
re_contrasts = behavior[re_mask].signedContrast
rc_df = behavior[rc_mask]
correct_r_times = []
for con in re_contrasts:# sample from the subset that matches the contrast of the error trials
    correct_r_times.append(rc_df[rc_df['signedContrast']==con].sample()['feedback_times'].iloc[0])
correct_r = time_to_frames(sync_behavior,localdisk,np.array(correct_r_times))

lc_df = behavior[lc_mask]
le_contrasts = behavior[le_mask].signedContrast
correct_l_times = []
for con in le_contrasts:# sample from the subset that matches the contrast of the error trials
    correct_l_times.append(lc_df[lc_df['signedContrast']==con].sample()['feedback_times'].iloc[0])
correct_l = time_to_frames(sync_behavior,localdisk,np.array(correct_l_times))

error_r = time_to_frames(sync_behavior,localdisk,np.array(behavior.loc[re_mask,'feedback_times']))
error_l = time_to_frames(sync_behavior,localdisk,np.array(behavior.loc[le_mask,'feedback_times']))

x,error_r_psth,ste = whole_brain_peth(error_r,stack.U_warped,SVTcorr,window=[-20,40])
x,correct_r_psth,ste = whole_brain_peth(correct_r,stack.U_warped,SVTcorr,window=[-20,40])
x,error_l_psth,ste = whole_brain_peth(error_l,stack.U_warped,SVTcorr,window=[-20,40])
x,correct_l_psth,ste = whole_brain_peth(correct_l,stack.U_warped,SVTcorr,window=[-20,40])
rPSTHs = [error_r_psth,correct_r_psth]
lPSTHs = [error_l_psth,correct_l_psth]
## make the movie
error_correct_mov(rPSTHs,lPSTHs, nref_regions, fname='correct_vs_error.gif')

def error_correct_mov(rPSTHs,lPSTHs, nref_regions, fname=None):
    fig=plt.figure(figsize=[6,  5])
    ax2 = fig.add_axes([.85,0.05,.025,.9])
    ax1 = fig.add_axes([0,0,.85,1])

    norm = mplc.Normalize(vmin=-.04,vmax=.04)
    colorbar.ColorbarBase(ax2,cmap=plt.get_cmap('RdBu_r'),
                                norm=norm, label='df/f')
    dat = []
    fb_type = ['Error','Correct']
    for ii in np.arange(len(rPSTHs)):
        xoff = rPSTHs[ii].shape[2]*(ii)
        for i,r in nref_regions.iterrows():
            plt.plot(xoff+r['left_x'],r['left_y'],color='k',lw=0.3)
            plt.plot(xoff+r['right_x'],r['right_y'],color='k',lw=0.3)

            plt.plot(xoff+r['right_x'],np.array(r['right_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)
            plt.plot(xoff+r['left_x'],np.array(r['left_y'])+rPSTHs[ii].shape[1],color='k',lw=0.3)

        # if ii == 0:
        #     plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'Feedback Type',ha='center',va = 'bottom')    
        plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,fb_type[ii],ha='center',va = 'top')
        if ii == len(rPSTHs)-1:
            plt.text(10,rPSTHs[ii].shape[1]*.7,'Right Stimulus', rotation=90)
            plt.text(10,rPSTHs[ii].shape[1]*.7 + rPSTHs[ii].shape[1],'Left Stimulus', rotation=90)
        # tmp = np.concatenate([rPSTHs[ii]-np.mean(rPSTHs[ii][:npre],axis=0),
        #                               lPSTHs[ii]-np.mean(lPSTHs[ii][:npre],axis=0)],axis=1)
        tmp = np.concatenate([rPSTHs[ii], lPSTHs[ii]],axis=1)
        if len(dat)==0:
            dat = tmp.copy()
        else:
            dat = np.concatenate([dat,tmp.copy()],axis=2)
    #make dot for stim on
    dat[npre:npre+5,60:80,60:80] = -np.max(dat)
    # tmp[npre:npre+30,:30,:30] = 1
    plt.axis('off')
    if fname is None:
        nb_play_movie(dat,
                      clim=[-0.04,0.04],cmap='RdBu_r')
    else:
        nb_save_movie(dat,filename=fname,
                      clim=[-0.04,0.04],cmap='RdBu_r')

behavior['choice'] = behavior['choice'].map({-1:'CCW',1:'CW'})
npre = 20

fig,ax = plt.subplots(1,1)
x, rPSTHs,ste = psth_from_df(U,SVTcorr,behavior,sync_behavior,atlas, [3,18], ccf_regions,localdisk,ax=ax, events='firstMovement_times', split_on='choice',window=[-20,40],smoothing=None)

L_R_mov(rPSTHs,rPSTHs, nref_regions, fname='L_vs_R_choice.gif')
show()

def L_R_mov(rPSTHs,lPSTHs, nref_regions, fname=None):
    # CHOICE=1, Left contrast, CW wheel turn; CHOICE=-1, Right Contrast, CCW wheel turn
    fig=plt.figure(figsize=[10.88,  3.71])
    ax2 = fig.add_axes([.85,0.05,.025,.9])
    ax1 = fig.add_axes([0,0,.85,1])

    norm = mplc.Normalize(vmin=-.04,vmax=.04)
    colorbar.ColorbarBase(ax2,cmap=plt.get_cmap('RdBu_r'),
                                norm=norm, label='df/f')
    dat = []
    for ii in np.arange(len(rPSTHs)):
        xoff = rPSTHs[ii].shape[2]*(ii)
        for i,r in nref_regions.iterrows():
            plt.plot(xoff+r['left_x'],r['left_y'],color='k',lw=0.3)
            plt.plot(xoff+r['right_x'],r['right_y'],color='k',lw=0.3)

        if ii == 0:
            plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'Right Contrast, CCW',ha='center',va = 'bottom')  
        else:
            plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'Left contrast, CW',ha='center',va = 'bottom') 
             
        # plt.text(xoff+rPSTHs[ii].shape[2]*0.8,50,'{0:1.1f}'.format(prob_left[ii]),ha='center',va = 'top')

        tmp = rPSTHs[ii]
        if len(dat)==0:
            dat = tmp.copy()
        else:
            dat = np.concatenate([dat,tmp.copy()],axis=2)
    #make dot for stim on
    dat[npre:npre+5,60:80,60:80] = -np.max(dat)
    # tmp[npre:npre+30,:30,:30] = 1
    plt.axis('off')
    if fname is None:
        nb_play_movie(dat,
                      clim=[-0.04,0.04],cmap='RdBu_r')
    else:
        nb_save_movie(dat,filename=fname,
                      clim=[-0.04,0.04],cmap='RdBu_r')
