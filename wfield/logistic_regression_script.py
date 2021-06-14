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
import matplotlib as mpl
import seaborn as sns
os.chdir(r'C:\Users\chris\int-brain-lab\wfield\wfield')
from analyses import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats 


subject = 'CSK-im-002'
baseDir = r'H:\imaging_data'
os.chdir(pjoin(baseDir,subject))
dates = os.listdir()
# dates= ['2021-01-17']
ims = []
trials = []
frames = []
for date in dates:
    try:
        behavior, frame_df, stack, ccf_regions, atlas = get_ses_data(subject,date)
    except:
        continue
    downsampled_im = spatial_down_sample(stack,20)
    #compile all the data for one mouse into one
    trials.append(behavior)
    ims.append(downsampled_im)
    frames.append(frame_df)
    im_shape = downsampled_im.shape[1:]

frame_df = pd.concat(frames).reset_index()
downsampled_im = np.concatenate(ims,axis=0)
behavior = pd.concat(trials).reset_index()

stims = behavior['signedContrast']
ustims,stims_choice1 = np.unique(stims[(behavior['signedContrast'] !=0) & (behavior['choice']==1)],return_counts=True)
u_stims,stims_choice0 = np.unique(stims[(behavior['signedContrast'] !=0) & (behavior['choice']==-1)],return_counts=True)
ustims = np.arange(-4,4)
fig,ax = plt.subplots(1,1)
ax.bar(ustims-.15,stims_choice0/7,width=.3,color='b')
ax.bar(ustims+.15,stims_choice1/7,width=.3,color='r')
ax.set_xticks(ustims)
ax.set_xticklabels(u_stims)
ax.set_title(subject)
ax.set_ylabel('count')
ax.set_xlabel('signed contrast')
show()

train_ses = 3
behavior = trials[train_ses]
downsampled_im = ims[train_ses]
frame_df = frames[train_ses]
# X,Y = format_for_choice_logreg(behavior,downsampled_im,frame_df)

C = .5
baseline_correct=True
rescale=True
X,Y = format_for_logreg(behavior,downsampled_im,frame_df,n_pre=0,n_post=3,use_choice=None,baseline_correct=baseline_correct,rescale=rescale)
# coefs = model.coef_.reshape(im_shape)

# toggle here for including abseline activity as seperate parameters
# Xb, _ = format_baseline_for_logreg(behavior,downsampled_im,frame_df,rescale=rescale)
# Xb = np.mean(Xb[:,mask],axis=1).reshape(-1,1) # To make a single parameter for Xb
# X = np.concatenate([X,Xb],axis=1)
results,SD,model = crossval_log_reg_pred(X,Y,C=C)
coefs = model.coef_[:,:np.product(im_shape)].reshape(im_shape)
# baseline_coefs = model.coef_[:,np.product(im_shape):].reshape(im_shape)

# shuf_res,shuf_sd,shuf_coefs = crossval_log_reg_pred(X[use_stims],np.random.permutation(Y_stim[use_stims]))
plt.figure()
im = plt.imshow(coefs, cmap='RdBu_r',vmax = np.max(coefs),vmin=-np.max(coefs))
plt.colorbar(im)
plt.title('C = {}, baseline_correct = {}, accuracy = {:.2f}, rescale = {}'.format(C,baseline_correct,results,rescale))

# plt.figure()
# im = plt.imshow(baseline_coefs, cmap='RdBu_r',vmax = np.max(baseline_coefs),vmin=-np.max(baseline_coefs))
# plt.colorbar(im)
# plt.title('Baseline Coefficients')
# print(results)
show()
# pause(.1)


ses_preds = np.zeros((len(ims),len(ims)))
coef_maps = []
#loop over sessions
for ses1 in tqdm(range(len(ims))):
    #format X and Y for logistic regression
    X1,Y1 = format_for_logreg(trials[ses1],ims[ses1],frames[ses1],n_pre = 0,n_post=3,con_cutoff=0,baseline_correct=True,rescale=True)
    behavior = trials[ses1]
    image=ims[ses1]
    frame_df=frames[ses1]
    # get train and test indices
    test_idx = np.random.choice(np.arange(len(Y1)),round(len(Y1)/3))
    train_idx = np.delete(np.arange(len(Y1)), test_idx)
    test_X1 = X1[test_idx]
    test_Y1  =Y1[test_idx]
    train_X1 = X1[train_idx]
    train_Y1 = Y1[train_idx]
    print(X1.shape)
    # train the model, TODO remove cross validation
    model = LogisticRegression(C=.5,solver='saga',max_iter=100000,penalty='l2',fit_intercept=True)
    model.fit(train_X1,train_Y1)
    # option to turn off part of the brain to see the effect on behavior
    # vis_off_im = inhibit_spot(image,[45,27],radius=3)
    # ax = inhibition_plot(model,vis_off_im,image,behavior.iloc[test_idx],frame_df.iloc[test_idx])
    # ax.set_title('Right vis off simulation')
    # make neurometric curves using only test data
    # plot_neurometric(model,image,behavior.iloc[test_idx],frame_df.iloc[test_idx])
    # block_neurometric_plot(model,image,behavior,frame_df)
    # results,SD,coef = crossval_log_reg_pred(X,Y_choice,C=.1)
    coefs = model.coef_.reshape(im_shape)
    coef_maps.append(coefs)
    # shuf_res,shuf_sd,shuf_coefs = crossval_log_reg_pred(X[use_stims],np.random.permutation(Y_stim[use_stims]))
    plt.figure()
    plt.imshow(coefs, cmap='RdBu_r',vmax = np.max(coefs),vmin=-np.max(coefs))
    plt.suptitle('decode stim day {}'.format(ses1))
    for ses2 in range(len(ims)):
        # block_neurometric_plot(model,ims[ses2],trials[ses2],frames[ses2])
        X2,Y2 = format_for_logreg(trials[ses2],ims[ses2],frames[ses2],n_post=3,baseline_correct=True,rescale=True)
        prediction = model.predict(X2)
        ses_preds[ses1,ses2] = np.mean(prediction==Y2)

plt.figure()
pred_mat = plt.imshow(ses_preds,cmap='jet',vmin=.5,vmax=1)
plt.colorbar(pred_mat)

plt.figure()
mean_coefs = np.mean(coef_maps,axis=0)
plt.imshow(mean_coefs,cmap='RdBu_r',vmax = np.max(mean_coefs),vmin=-np.max(mean_coefs))
plt.title('mean coefficient map for {}'.format(subject))
show()

## simulated inhibition scan:
scanX = []
scanY = []
scan_spots = np.vstack(scanX,scanY)
    X,Y = format_for_choice_logreg(behavior,iamge,frame_df)
    # get train and test indices
    test_idx = np.random.choice(np.arange(len(Y)),round(len(Y)/3))
    train_idx = np.delete(np.arange(len(Y)), test_idx)
    test_X1 = X[test_idx]
    test_Y1  =Y[test_idx]
    train_X1 = X[train_idx]
    train_Y1 = Y[train_idx]
    model = LogisticRegression(C=2,solver='saga',max_iter=100000,penalty='l2',fit_intercept=False)
    model.fit(train_X,train_Y)
for spot in scan_spot:
    inhibition_im = inhibit_spot(image,spot,radius=2)
    uCons, pred,pred_se,actual,actual_se = make_neurometric(model,inhibition_im,behavior[test_idx],frame_df[test_idx],baseline_correct=True)

### decode conditioned on choice

train_ses = 3
behavior = trials[train_ses]
downsampled_im = ims[train_ses]
frame_df = frames[train_ses]

C = .5
baseline_correct=True
rescale=True
for train_ses in range(len(trials)):
    behavior = trials[train_ses]
    downsampled_im = ims[train_ses]
    frame_df = frames[train_ses]
    X,Y = format_for_logreg(behavior,downsampled_im,frame_df,use_choice=1,n_pre=0,n_post=3,baseline_correct=baseline_correct,rescale=rescale)
    test_idx = np.random.choice(np.arange(len(Y)),round(len(Y)/4))
    train_idx = np.delete(np.arange(len(Y)), test_idx)
    test_X1 = X[test_idx]
    test_Y1  =Y[test_idx]
    train_X1 = X[train_idx]
    train_Y1 = Y[train_idx]
    model = LogisticRegression(C=.5,solver='liblinear',max_iter=100000,penalty='l2',fit_intercept=True,class_weight='balanced')
    model.fit(train_X1,train_Y1)

    coefs1 = model.coef_[:,:np.product(im_shape)].reshape(im_shape)

    X,Y = format_for_logreg(behavior,downsampled_im,frame_df,use_choice=0,n_pre=0,n_post=3,baseline_correct=baseline_correct,rescale=rescale)
    
    results = np.mean(model.predict(X)==Y)
    plt.figure()
    im = plt.imshow(coefs1, cmap='RdBu_r',vmax = np.max(coefs1),vmin=-np.max(coefs1))
    plt.colorbar(im)
    plt.title('Choice = 1, accuracy = {:.2f}'.format(results))

    show()

    X,Y = format_for_logreg(behavior,downsampled_im,frame_df,use_choice=0,n_pre=0,n_post=3,baseline_correct=baseline_correct,rescale=rescale)
    test_idx = np.random.choice(np.arange(len(Y)),round(len(Y)/4))
    train_idx = np.delete(np.arange(len(Y)), test_idx)
    test_X2 = X[test_idx]
    test_Y2  =Y[test_idx]
    train_X2 = X[train_idx]
    train_Y2 = Y[train_idx]
    model = LogisticRegression(C=.5,solver='liblinear',max_iter=100000,penalty='l2',fit_intercept=True,class_weight='balanced')
    model.fit(train_X2,train_Y2)

    coefs0 = model.coef_[:,:np.product(im_shape)].reshape(im_shape)

    X,Y = format_for_logreg(behavior,downsampled_im,frame_df,use_choice=1,n_pre=0,n_post=3,baseline_correct=baseline_correct,rescale=rescale)

    results = np.mean(model.predict(X)==Y)
    plt.figure()
    im = plt.imshow(coefs0, cmap='RdBu_r',vmax = np.max(coefs0),vmin=-np.max(coefs0))
    plt.colorbar(im)
    plt.title('Choice = 0, accuracy = {:.2f}'.format(results))

    show()
    testX = np.concatenate([test_X1,test_X2])
    testY = np.concatenate([test_Y1,test_Y2])
    model.coef_ = np.mean([coefs0,coefs1],axis=0).reshape(1,coefs0.shape[0]*coefs0.shape[1])
    results = np.mean(model.predict(testX)==testY)
    mean_coefs = model.coef_.reshape(im_shape)
    plt.figure()
    im = plt.imshow(mean_coefs, cmap='RdBu_r',vmax = np.max(coefs0),vmin=-np.max(coefs0))
    plt.colorbar(im)
    plt.title('Mean choice conditioned weights, accuracy = {:.2f}'.format(results))

    show()

########################################################
# make map of decoding ability with each pixel
#######################################################
pixel_decoding = np.zeros((len(ims),len(flat_mask)))
for ses1 in range(len(ims)):
    #format X and Y for logistic regression
    X1,Y1 = format_for_logreg(trials[ses1],ims[ses1],frames[ses1],use_choice=None,n_pre = 0,n_post=3,con_cutoff=0.1,baseline_correct=True,rescale=True)
    behavior = trials[ses1]
    image=ims[ses1]
    frame_df=frames[ses1]
    # get train and test indices
    test_idx = np.random.choice(np.arange(len(Y1)),round(len(Y1)/4))
    train_idx = np.delete(np.arange(len(Y1)), test_idx)
    test_X1 = X1[test_idx]
    test_Y1  =Y1[test_idx]
    train_X1 = X1[train_idx]
    train_Y1 = Y1[train_idx]
    print(X1.shape)
    for pixel in tqdm(range(train_X1.shape[1])):
        if flat_mask[pixel]:
            train_X = train_X1[:,pixel]
            test_X = test_X1[:,pixel]
            # train the model, TODO remove cross validation
            model = LogisticRegression(C=110,solver='saga',max_iter=100000,penalty='none',fit_intercept=True,class_weight='balanced')
            model.fit(train_X.reshape(-1,1),train_Y1)
            pixel_decoding[ses1,pixel] = model.score(test_X.reshape(-1,1),test_Y1)
        else:
            pixel_decoding[ses1,pixel] = np.nan
fig,axs = plt.subplots(2,4)
for i in range(pixel_decoding.shape[0]):
    ax = axs[i%2,int(i/2)]
    pix_im = ax.imshow(pixel_decoding[i].reshape(im_shape),cmap='viridis',vmin=.42,vmax=.75)
    ax.axis('off')
    if i== pixel_decoding.shape[0]-1:
        plt.colorbar(pix_im)
        
    show()

########################################################
# make histogram of decoding ability with each pixel
#######################################################
n_samples = 500
n_pixels = 15
min_dist = 0
sampled_accuracy = np.zeros((len(ims),n_samples))
colors = []
for ses1 in range(len(ims)):
    X1,Y1 = format_for_logreg(trials[ses1],ims[ses1],frames[ses1],use_choice=None,n_pre = 0,n_post=3,con_cutoff=0,baseline_correct=True,rescale=True)
    behavior = trials[ses1]
    image=ims[ses1]
    frame_df=frames[ses1]
    brain_mask = image[0,:,:] !=0
    flat_mask = brain_mask.flatten()
    # get train and test indices
    test_idx = np.random.choice(np.arange(len(Y1)),round(len(Y1)/3))
    train_idx = np.delete(np.arange(len(Y1)), test_idx)
    test_X1 = X1[test_idx]
    # test_X1 = test_X1.reshape((test_X1.shape[0],im_shape[0],im_shape[1]))
    test_Y1  =Y1[test_idx]
    train_X1 = X1[train_idx]
    # train_X1 = train_X1.reshape((train_X1.shape[0],im_shape[0],im_shape[1]))
    train_Y1 = Y1[train_idx]
    print(X1.shape)

    for sample_num in tqdm(range(n_samples)):
        use_pix = np.random.choice(np.arange(flat_mask.shape[0])[flat_mask],n_pixels,replace=False)
        # use_x = np.random.choice(np.arange(im_shape[0]),n_pixels,replace=False)
        # use_y = np.random.choice(np.arange(im_shape[1]),n_pixels,replace=False)
        # idxs = np.array([[x,y] for x,y in zip(use_x,use_y)])
        # while (np.sign(use_y[1]-16)==np.sign(use_y[0]-16)) | (not all([brain_mask[idxs[idx,0],idxs[idx,1]] for idx in range(idxs.shape[0])])):
        #     use_x = np.random.choice(np.arange(im_shape[0]),n_pixels,replace=False)
        #     use_y = np.random.choice(np.arange(im_shape[1]),n_pixels,replace=False)
        #     idxs = np.array([[x,y] for x,y in zip(use_x,use_y)])
        while not all(flat_mask[use_pix]):
            print('here')
            use_pix = np.random.choice(np.arange(flat_mask.shape[0]),n_pixels,replace=False)
        train_X = train_X1[:,use_pix]
        test_X = test_X1[:,use_pix]
        # train_X = train_X1[:,use_x,use_y]
        # test_X = test_X1[:,use_x,use_y]
        # train the model, TODO remove cross validation
        model = LogisticRegression(solver='saga',max_iter=100000,penalty='none',fit_intercept=True,class_weight='balanced')
        model.fit(train_X,train_Y1)
        score = np.mean(model.predict(test_X) == test_Y1)
        if (np.max(model.coef_) > 0) & (np.min(model.coef_) < 0):
            colors.append(1)
        else:
            colors.append(0)
        if score == 0:
            # score = np.nan
            print(np.mean(model.predict(test_X)))
        sampled_accuracy[ses1,sample_num] = score 
plt.figure()
# sns.histplot(data=sampled_accuracy.flatten().reshape(1,-1), hue=colors,multiple='stack')
data=sampled_accuracy.flatten()
colors=np.array(colors)
plt.hist([data[colors==0],data[colors==1]],bins=20,histtype='bar',label =['single sign coefs','multi sign coefs'] )
plt.title('{} pixel decoding'.format(n_pixels,min_dist))
plt.xlabel('test accuracy')
plt.ylabel('count')
plt.text(.6,600,'median = {}'.format(np.round(np.median(sampled_accuracy),3)))
plt.xlim([.375,.75])
plt.ylim([0,975])
plt.legend()
show()

def format_for_logreg(behavior,downsampled_im,frame_df, use_choice=1, n_pre=0, n_post=3,con_cutoff=0,baseline_correct=True,rescale=True):
    """
    behavior: (pd.DataFrame) len trials with behavioral events
    downsampled_im: (np.array) nframes x pixels_hight x pixels_wide
    frame_df: (pd.dataframe) len trials with frame number of behavior event
    con_cutoff: (float) the absolute value of contrasts greater than that value to use (abs(use_stim) > con_cutoff)
    """
    im_shape = downsampled_im.shape[1:]
    downsampled_im = downsampled_im.reshape((downsampled_im.shape[0],downsampled_im.shape[1]*downsampled_im.shape[2]))
    mask = downsampled_im[0]!=0
    flat_mask = mask.flatten()
    zero_mask = ~flat_mask
    # drop the .5 block
    # frame_df = frame_df[behavior['probabilityLeft']!=.5]
    # behavior = behavior[behavior['probabilityLeft']!=.5]

    # get the features from the downsampled image
    stim_frames = frame_df['stimOn_times']
    fm_frames = frame_df['firstMovement_times']
    #only select frames with first movement times > 120ms after onset
    # use_frames = (fm_frames-stim_frames)>n_post
    # get the actual blocks
    Y_prob = np.array(behavior['probabilityLeft'])
    Y_choice = np.array(behavior['choice'])
    Y_choice[Y_choice == -1] = 0
    if use_choice == None:
        use_frames = np.ones_like(Y_choice).astype(bool)
    else:
        use_frames = Y_choice==use_choice
    use_stims = (np.abs(behavior['signedContrast']) > con_cutoff ) & (use_frames)
    Y_stim = np.array(np.sign(behavior['signedContrast']))
    Y_stim[Y_stim == -1] = 0

    Y_prob[Y_prob == .2] = 0; Y_prob[Y_prob == .8] = 1 #left blocks are 1, right blocks are 0
    X = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
        spot_act = downsampled_im[:,spot]
        # get arrays nTrials averaged across the frames
        block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in stim_frames])
        X[:,spot] = block_act
    baseline = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    if baseline_correct:
        for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
            spot_act = downsampled_im[:,spot]
            # get arrays nTrials averaged across the frames
            block_act = np.array([np.sum(spot_act[frame-3:frame]) for frame in stim_frames])
            baseline[:,spot] = block_act
        X=X-baseline
    # scale X from -1 to 1 
    if rescale:
        average = (np.min(X) + np.max(X)) / 2
        rang = (np.max(X) - np.min(X)) / 2
        X = (X - average) / rang
        X[:,zero_mask] = 0
    # downsample to match conditions
    X = X[use_stims]
    Y = Y_stim[use_stims]
    # ys, y_counts = np.unique(Y,return_counts=True)
    # if y_counts[0]==y_counts[1]:
    #     return X, Y
    # else:
    #     min_counts = np.min(y_counts)
    #     min_y = np.argmin(y_counts)
    #     max_y = np.argmax(y_counts)
    #     overrepresented_y = ys[max_y]
    #     Y_idx1 = np.random.choice(np.arange(len(Y))[Y==max_y],min_counts,replace=False)
    #     Y_idx2 = np.random.choice(np.arange(len(Y))[Y==min_y],min_counts,replace=False)
    #     even_sampled = np.concatenate([Y_idx1,Y_idx2])
    #     return X[even_sampled], Y[even_sampled]
    return X,Y

def inhibit_spot(image,spot,radius=2,shape=[27,32]):
    """
    simulation of optogenetic inhibition by simply setting a chunk of activity to zero
    X: flattened image of shape n, N pixels 
    spot = list or tuple of X,Y coords for pixel to turn off
    radius (float) number of pixels to extend in either direction to include in inhibition
    shape (list len 2) of image shape for X
    """
    x_off = spot[0]+1
    y_off = spot[1]+1
    # X_im = image.reshape((image.shape[0],shape[0],shape[1]))
    off_mask = np.zeros_like(image).astype(bool)
    off_mask[:,round(y_off-radius):round(y_off+radius),round(x_off-radius):int(np.ceil(x_off+radius))] = True
    vis_off_im = image.copy()
    vis_off_im[off_mask] = 0
    # np.reshape(image,(X.shape[0],shape[0]*shape[1]))
    return vis_off_im

def inhibition_plot(model,inhibition_image,baseline_image,behavior,frame_df):
    ax = plot_neurometric(model,image,behavior,frame_df)
    uCons, pred,pred_se,actual,actual_se = make_neurometric_choice(model,inhibition_image,behavior,frame_df,baseline_correct=True)
    ax.plot(uCons,pred,'g')
    ax.fill_between(uCons,pred-pred_se,pred+pred_se,color='g',alpha=.3)
    return ax

def plot_neurometric(model,image,behavior,frame_df,plot_type = 'choice'):
    if plot_type == 'choice':
        uCons, pred,pred_se,actual,actual_se = make_neurometric_choice(model,image,behavior,frame_df,baseline_correct=True)
    if plot_type == 'stim':
        uCons, pred,pred_se,actual,actual_se = make_neurometric_stim(model,image,behavior,frame_df,baseline_correct=True)
    plt.figure()
    plt.suptitle('neurometric and psychometric')
    plt.plot(uCons,pred, 'b')
    plt.fill_between(uCons,pred-pred_se,pred+pred_se,'b',alpha=.3)
    plt.plot(uCons,actual,'r')
    plt.fill_between(uCons,actual-actual_se,actual+actual_se,'r',alpha=.3)
    plt.xlabel('Signed Contrast')
    plt.ylabel('Percent Choose Left')
    make_legend(plt.gca(), ['b','r'], ['neurometric','psychometric'], bbox_to_anchor=[.05,.95])
    return plt.gca()

def block_neurometric_plot(model,image,behavior,frame_df):
    plt.figure()
    plt.suptitle('neurometric and psychometric')
    colors=['r','b']
    for i,prob in enumerate([.2,.8]):
        use = behavior['probabilityLeft'] == prob 
        uCons, pred,pred_se,actual,actual_se = make_neurometric_choice(model,image,behavior[use],frame_df[use],baseline_correct=True)
        plt.plot(uCons,pred, colors[i],ls='--')
        plt.fill_between(uCons,pred-pred_se,pred+pred_se,color=colors[i],alpha=.3)
        plt.plot(uCons,actual,colors[i])
        plt.fill_between(uCons,actual-actual_se,actual+actual_se,color=colors[i],alpha=.3)
    plt.xlabel('Signed Contrast')
    plt.ylabel('Percent Choose Left')
    make_legend(plt.gca(), ['b','r'], ['right block','left block'], bbox_to_anchor=[.05,.95])
    show()    



def format_for_choice_logreg(behavior,downsampled_im,frame_df,npre = -3,npost=6,baseline_correct=True,rescale=True):
    """
    behavior: (pd.DataFrame) len trials with behavioral events
    downsampled_im: (np.array) nframes x pixels_hight x pixels_wide
    frame_df: (pd.dataframe) len trials with frame number of behavior event
    con_cutoff: (float) the absolute value of contrasts greater than that value to use (abs(use_stim) > con_cutoff)
    """
    im_shape = downsampled_im.shape[1:]
    downsampled_im = downsampled_im.reshape((downsampled_im.shape[0],downsampled_im.shape[1]*downsampled_im.shape[2]))
    mask = downsampled_im[0]!=0
    flat_mask = mask.flatten()
    zero_mask = ~flat_mask
    # drop the .5 block
    # frame_df = frame_df[behavior['probabilityLeft']!=.5]
    # behavior = behavior[behavior['probabilityLeft']!=.5]

    # get the features from the downsampled image
    move_frames = frame_df['firstMovement_times']
    stim_frames = frame_df['stimOn_times']
    # get the actual blocks
    Y_prob = np.array(behavior['probabilityLeft'])
    Y_choice = np.array(behavior['choice'])
    Y_choice[Y_choice == -1] = 0
    # use_stims = (np.abs(behavior['signedContrast']) > con_cutoff ) #& (behavior['feedbackType']==1)
    # Y_stim = np.array(np.sign(behavior['signedContrast']))

    # Y_stim[Y_stim == -1] = 0
    Y_prob[Y_prob == .2] = 0; Y_prob[Y_prob == .8] = 1 #left blocks are 1, right blocks are 0

    X = np.zeros((len(move_frames),downsampled_im.shape[1]))
    for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
        spot_act = downsampled_im[:,spot]
        # get arrays nTrials averaged across the frames
        block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in move_frames])
        X[:,spot] = block_act
        baseline = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    if baseline_correct:
        for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
            spot_act = downsampled_im[:,spot]
            # get arrays nTrials averaged across the frames
            block_act = np.array([np.sum(spot_act[frame-3:frame]) for frame in stim_frames])
            baseline[:,spot] = block_act
        X=X-baseline
    # scale X from -1 to 1 
    if rescale:
        average = (np.min(X) + np.max(X)) / 2
        rang = (np.max(X) - np.min(X)) / 2
        X = (X - average) / rang
        X[:,zero_mask] = 0
    return X,Y_choice


def format_baseline_for_logreg(behavior,downsampled_im,frame_df,con_cutoff=0,rescale=True):
    """
    behavior: (pd.DataFrame) len trials with behavioral events
    downsampled_im: (np.array) nframes x pixels_hight x pixels_wide
    frame_df: (pd.dataframe) len trials with frame number of behavior event
    con_cutoff: (float) the absolute value of contrasts greater than that value to use (abs(use_stim) > con_cutoff)
    """
    im_shape = downsampled_im.shape[1:]
    downsampled_im = downsampled_im.reshape((downsampled_im.shape[0],downsampled_im.shape[1]*downsampled_im.shape[2]))
    mask = downsampled_im[0]!=0
    flat_mask = mask.flatten()
    zero_mask = ~flat_mask
    # drop the .5 block
    # frame_df = frame_df[behavior['probabilityLeft']!=.5]
    # behavior = behavior[behavior['probabilityLeft']!=.5]

    # get the features from the downsampled image
    stim_frames = frame_df['stimOn_times']
    # get the actual blocks
    Y_prob = np.array(behavior['probabilityLeft'])
    Y_choice = np.array(behavior['choice'])
    Y_choice[Y_choice == -1] = 0
    use_stims = (np.abs(behavior['signedContrast']) > con_cutoff ) #& (behavior['feedbackType']==1)
    Y_stim = np.array(np.sign(behavior['signedContrast']))

    Y_stim[Y_stim == -1] = 0
    Y_prob[Y_prob == .2] = 0; Y_prob[Y_prob == .8] = 1 #left blocks are 1, right blocks are 0
    n_pre =  3  # 200ms before stim on
    n_post = 0
    X = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
        spot_act = downsampled_im[:,spot]
        # get arrays nTrials averaged across the frames
        block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in stim_frames])
        X[:,spot] = block_act
    # scale X from -1 to 1 
    if rescale:
        average = (np.min(X) + np.max(X)) / 2
        rang = (np.max(X) - np.min(X)) / 2
        X = (X - average) / rang
        X[:,zero_mask] = 0
    return X[use_stims], Y_stim[use_stims]



def crossval_log_reg_pred(X,Y,K=10,C=1):
    """
    Train and test a logistic regression model with the predictors being a vector of df/f values 
    through time for a single pixel
    X, matrix of predictors (np.array) n_samples,nfeatures
    Y, vector of labels (np.array) n_samples
    K, number of cross validation folds (int)
    C, regularization param, smaller #s = more regularization (int)
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    kf = KFold(K,shuffle=True)
    kf.get_n_splits(Y)
    k=0
    mods = []
    test_results = np.zeros(K)
    for train,test in tqdm(kf.split(Y)):
        train_X, test_X = X[train,:],X[test,:]
        train_Y, test_Y = Y[train],Y[test]
        
        model = LogisticRegression(C=C,solver='saga',max_iter=100000,penalty='l2',fit_intercept=True)
        model.fit(train_X,train_Y)
        print(np.mean(model.predict(train_X)==train_Y))
        prediction = model.predict(test_X)
        test_results[k] = np.mean(prediction == test_Y)
        mods.append(test_results)
        k+=1
    test_SD = test_results.std(axis=0)
    test_results = test_results.mean(axis=0)
    return test_results, test_X, model

# def make_neurometric(model,downsampled_im,behavior,frame_df,baseline_correct=True):

#     im_shape = downsampled_im.shape[1:]
#     downsampled_im = downsampled_im.reshape((downsampled_im.shape[0],downsampled_im.shape[1]*downsampled_im.shape[2]))
#     mask = downsampled_im[0]!=0
#     flat_mask = mask.flatten()
#     zero_mask = ~flat_mask

#     choices = behavior['choice']
#     choices[choices==-1]=0
#     signed_contrast = behavior['signedContrast']
#     uContrasts = np.unique(behavior['signedContrast'])
#     stim_ons = frame_df['stimOn_times']
#     predicted_choices = np.zeros(len(uContrasts))
#     actual_choices = np.zeros(len(uContrasts))
#     actual_se = np.zeros(len(uContrasts))
#     predicted_se = np.zeros(len(uContrasts))
#     for i,con in enumerate(uContrasts):
#         stim_frames = stim_ons[signed_contrast==con]
#         n_pre =  -3  # 200ms before stim on
#         n_post = 6
#         X = np.zeros((len(stim_frames),downsampled_im.shape[1]))
#         for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
#             spot_act = downsampled_im[:,spot]
#             # get arrays nTrials averaged across the frames
#             block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in stim_frames])
#             X[:,spot] = block_act
#         baseline = np.zeros((len(stim_frames),downsampled_im.shape[1]))
#         if baseline_correct:
#             for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
#                 spot_act = downsampled_im[:,spot]
#                 # get arrays nTrials averaged across the frames
#                 block_act = np.array([np.sum(spot_act[frame-3:frame]) for frame in stim_frames])
#                 baseline[:,spot] = block_act
#             X=X-baseline
#         # scale X from -1 to 1 
#         average = (np.min(X) + np.max(X)) / 2
#         rang = (np.max(X) - np.min(X)) / 2
#         X = (X - average) / rang
#         X[:,zero_mask] = 0
        
#         predicted_choices[i] = 1-np.mean(model.predict(X))
#         predicted_se[i] = np.std(model.predict(X))/np.sqrt(len(stim_frames))
#         actual_choices[i] = 1-np.mean(choices[signed_contrast==con])
#         actual_se[i] = np.std(choices[signed_contrast==con])/np.sqrt(len(stim_frames))
#     return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se

def make_neurometric_choice(model,downsampled_im,behavior,frame_df,return_type='mean',baseline_correct=True):

    choices = behavior['choice']
    choices[choices==-1]=0
    signed_contrast = behavior['signedContrast']
    uContrasts,nContrasts = np.unique(behavior['signedContrast'],return_counts=True)
    stim_ons = frame_df['stimOn_times']
    if return_type == 'mean':
        predicted_choices = np.zeros(len(uContrasts))
        actual_choices = np.zeros(len(uContrasts))
        actual_se = np.zeros(len(uContrasts))
        predicted_se = np.zeros(len(uContrasts))
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,_ = format_for_choice_logreg(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],baseline_correct=baseline_correct)
            
            predicted_choices[i] = 1-np.mean(model.predict(X))
            predicted_se[i] = np.std(model.predict(X))/np.sqrt(len(stim_frames))
            actual_choices[i] = 1-np.mean(choices[signed_contrast==con])
            actual_se[i] = np.std(choices[signed_contrast==con])/np.sqrt(len(stim_frames))
        return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se
    elif return_type == 'raw':
        predicted_choices = []
        actual_choices = []
        actual_se = []
        predicted_se = []
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,_ = format_for_choice_logreg(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],baseline_correct=baseline_correct)
            
            predicted_choices.append(1-model.predict(X))
            predicted_se.append(None)
            actual_choices.append(1-choices[signed_contrast==con])
            actual_se.append(None)
    return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se


def make_neurometric_stim(model,downsampled_im,behavior,frame_df,return_type='mean',baseline_correct=True):
    frame_df = frame_df[behavior['signedContrast'] !=0].reset_index()  
    behavior = behavior[behavior['signedContrast'] !=0].reset_index()
    choices = behavior['choice']
    choices[choices==-1]=0
    signed_contrast = behavior['signedContrast'][behavior['signedContrast'] !=0]
    uContrasts,nContrasts = np.unique(behavior['signedContrast'],return_counts=True)
    stim_ons = frame_df['stimOn_times']
    if return_type == 'mean':
        predicted_choices = np.zeros(len(uContrasts))
        actual_choices = np.zeros(len(uContrasts))
        actual_se = np.zeros(len(uContrasts))
        predicted_se = np.zeros(len(uContrasts))
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,_ = format_for_logreg(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],use_choice=None,baseline_correct=baseline_correct)
            
            predicted_choices[i] = np.mean(model.predict(X))
            predicted_se[i] = np.std(model.predict(X))/np.sqrt(len(stim_frames))
            actual_choices[i] = 1-np.mean(choices[signed_contrast==con])
            actual_se[i] = np.std(choices[signed_contrast==con])/np.sqrt(len(stim_frames))
        return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se
    elif return_type == 'raw':
        predicted_choices = []
        actual_choices = []
        actual_se = []
        predicted_se = []
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,_ = format_for_logreg(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],use_choice=None,baseline_correct=baseline_correct)
            
            predicted_choices.append(model.predict(X))
            predicted_se.append(None)
            actual_choices.append(1-choices[signed_contrast==con])
            actual_se.append(None)
    return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se

## make a block triggered average of the baseline activity



train_ses = 6
behavior = trials[train_ses]
downsampled_im = ims[train_ses]
if len(downsampled_im.shape) ==3:
    downsampled_im=downsampled_im.reshape((downsampled_im.shape[0],im_shape[0]*im_shape[1]))
frame_df = frames[train_ses]
mask = downsampled_im[0]!=0
flat_mask = mask.flatten()
zero_mask = ~flat_mask

# get the features from the downsampled image
stim_frames = frame_df['stimOn_times']
# get the actual blocks
Y_prob = np.array(behavior['probabilityLeft'])
Y_choice = np.array(behavior['choice'])
Y_choice[Y_choice == -1] = 0
n_pre =  3  # 200ms before stim on
n_post = 0
baseline = np.zeros((len(stim_frames),downsampled_im.shape[1]))
for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
    spot_act = downsampled_im[:,spot]
    # get arrays nTrials averaged across the frames
    block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in stim_frames])
    baseline[:,spot] = block_act

mean_baseline = np.mean(baseline[:,flat_mask],axis=1) # To make a single parameter for Xb
baseline=baseline.reshape((baseline.shape[0],im_shape[0],im_shape[1]))
#toggle to select individual spots instead of the whole cortex
# mean_baseline = np.mean(baseline[:,11:14,18:21],axis=(1,2))
block_changes = np.concatenate([[0],np.diff(Y_prob)])
max_block = find_max_len_block(block_changes)+1
bc_type,bc_count = np.unique(block_changes,return_counts=True)
Lblock_num =np.sum(bc_count[np.where(bc_type<0)[0]])
Rblock_num = np.sum(bc_count[np.where(bc_type>0)[0]])
even_block = np.ones(max_block) *np.nan
left_blocks = np.ones((Lblock_num,max_block)) *np.nan
right_blocks = np.ones((Rblock_num,max_block)) *np.nan
block_type = 'even'
j = -1
k = -1
l=0
m=0
for i,trial in enumerate(block_changes):
    if trial < 0:
        block_type = 'left'
        j+=1
        l=0
    elif trial > 0 :
        block_type = 'right'
        k+=1
        m=0
    if block_type == 'even':
        even_block[i] = mean_baseline[i]
    elif block_type == 'left':
        left_blocks[j,l] = mean_baseline[i]
        l+=1
    elif block_type == 'right':
        right_blocks[k,m] = mean_baseline[i]
        m+=1
left_block = np.nanmean(left_blocks,axis=0)
right_block = np.nanmean(right_blocks,axis=0)
plt.figure()
plt.plot(np.arange(max_block),even_block,'k')
plt.plot(np.arange(max_block),left_block,'b')
plt.plot(np.arange(max_block),right_block,'r')
make_legend(plt.gca(),['k','b','r'],['even_block','left_block','right_block'])
plt.title('mean baseline activity between block types')
plt.xlabel('trials since block change')
plt.ylabel('mean prestimulus df/f')
plt.tight_layout()
show()


def find_max_len_block(block_changes):
    length = 0
    lengths = [0]
    for change in block_changes:
        if change ==0:
            length +=1
        if (change!=0):
            lengths.append(length)
            length=0
    return max(lengths)
        