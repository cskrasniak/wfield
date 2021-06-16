from labcams import parse_cam_log
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
os.chdir('/grid/zador/home/ckrasnia/wfield')
from  wfield import *
from sklearn.metrics import mean_squared_error
from matplotlib import cm
import matplotlib.colors as mplc
from matplotlib.lines import Line2D
import matplotlib as mpl
import seaborn as sns
os.chdir('/grid/zador/home/ckrasnia/wfield/wfield')
from analyses import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats 


def scale_to_range(measurements, range_max, range_min):
    meas_min = np.min(measurements)
    meas_max = np.max(measurements)
    return ((measurements-meas_min)/(meas_max-meas_min))*(range_max-range_min)+range_min

def format_for_logreg(behavior,downsampled_im,frame_df, use_choice=None, n_pre=0, n_post=3,con_cutoff=0,baseline_correct=True,rescale=True,split=False):
    """
    behavior: (pd.DataFrame) len trials with behavioral events
    downsampled_im: (np.array) nframes x pixels_hight x pixels_wide
    frame_df: (pd.dataframe) len trials with frame number of behavior event
    con_cutoff: (float) the absolute value of contrasts greater than that value to use (abs(use_stim) > con_cutoff)
    """
    im_shape = downsampled_im.shape[1:]
    if len(downsampled_im.shape) == 3:
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
    # Y_prob = np.array(behavior['probabilityLeft'])
    Y_choice = np.array(behavior['choice'])
    Y_choice[Y_choice == -1] = 0
    if use_choice == None:
        use_frames = np.ones_like(Y_choice).astype(bool)
    else:
        use_frames = Y_choice==use_choice
    use_stims = (np.abs(behavior['signedContrast']) > con_cutoff ) & (use_frames)
    Y_stim = np.array(np.sign(behavior['signedContrast']))
    Y_stim[Y_stim == -1] = 0

    # Y_prob[Y_prob == .2] = 0; Y_prob[Y_prob == .8] = 1 #left blocks are 1, right blocks are 0
    X = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
        spot_act = downsampled_im[:,spot]
        if np.count_nonzero(spot_act)==0:
            continue
        # get arrays nTrials averaged across the frames
        block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in stim_frames])
        X[:,spot] = block_act
    baseline = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    if baseline_correct:
        for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
            spot_act = downsampled_im[:,spot]
            if np.count_nonzero(spot_act)==0:
                continue
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
    if split:
        train_idx,test_idx = train_test_split(X,Y,behavior[use_stims].reset_index())
        return [X[train_idx],X[test_idx]], [Y[train_idx],Y[test_idx]]
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

def format_for_logreg_choice(behavior,downsampled_im,frame_df, n_pre=0, n_post=3,con_cutoff=0,baseline_correct=True,rescale=True,split=False):
    """
    behavior: (pd.DataFrame) len trials with behavioral events
    downsampled_im: (np.array) nframes x pixels_hight x pixels_wide
    frame_df: (pd.dataframe) len trials with frame number of behavior event
    con_cutoff: (float) the absolute value of contrasts greater than that value to use (abs(use_stim) > con_cutoff)
    """
    im_shape = downsampled_im.shape[1:]
    if len(downsampled_im.shape) == 3:
        downsampled_im = downsampled_im.reshape((downsampled_im.shape[0],downsampled_im.shape[1]*downsampled_im.shape[2]))

    mask = downsampled_im[0]!=0
    flat_mask = mask.flatten()
    zero_mask = ~flat_mask
    Y_choice = np.array(behavior['choice']).astype(float)
    Y_choice[Y_choice==0]=np.nan
    na_drop = ~np.isnan(Y_choice)
    behavior=behavior.iloc[na_drop].reset_index()
    frame_df=frame_df.iloc[na_drop].reset_index()
    Y_choice=Y_choice[na_drop]
    # get the features from the downsampled image
    stim_frames = frame_df['stimOn_times']

    # Y_choice[Y_choice == 1] = 0
    Y_choice[Y_choice == -1] = 0

    use_stims = (np.abs(behavior['signedContrast']) >= con_cutoff )
    Y_stim = np.array(np.sign(behavior['signedContrast']))
    Y_stim[Y_stim == -1] = 0

    # Y_prob[Y_prob == .2] = 0; Y_prob[Y_prob == .8] = 1 #left blocks are 1, right blocks are 0
    X = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
        spot_act = downsampled_im[:,spot]
        if np.count_nonzero(spot_act)==0:
            continue
        # get arrays nTrials averaged across the frames
        block_act = np.array([np.sum(spot_act[frame-n_pre:frame+n_post]) for frame in stim_frames])
        X[:,spot] = block_act
    baseline = np.zeros((len(stim_frames),downsampled_im.shape[1]))
    if baseline_correct:
        for spot in range(downsampled_im.shape[1]):# loop over each pixel in the image
            spot_act = downsampled_im[:,spot]
            if np.count_nonzero(spot_act)==0:
                continue
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
    Y = Y_choice[use_stims]
    if split:
        train_idx,test_idx = train_test_split(X,Y,behavior[use_stims].reset_index())
        return [X[train_idx],X[test_idx]], [Y[train_idx],Y[test_idx]]
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

def train_test_split(X,Y, behavior, include0 = False):
    """
    splits X and Y into a train and test set with an even number of trials 
    of each contrast level in the test set
    """
    uCons,counts = np.unique(behavior['signedContrast'],return_counts=True)
    if not include0:
        counts = counts[uCons!=0]
        uCons=uCons[uCons!=0]

    test_num = int(np.min(counts)/4)
    test_idx = np.zeros((len(uCons))*test_num).astype(int)
    for i, con in enumerate(uCons):
        con_trials = behavior[behavior['signedContrast'] == con].index.astype(int)
        test_idx[test_num*i:test_num*(i+1)] = np.random.choice(con_trials,test_num,replace=False)
    train_idx = np.arange(len(Y))[np.array([y not in test_idx for y in np.arange(len(Y))])]
    assert len(Y) == len(train_idx)+len(test_idx), 'train ({}) and test ({}) index lengths do not add up to Y ({}) length'.format(len(train_idx),len(test_idx),len(Y))
    return train_idx, test_idx


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
    plt.ylim([0,1])
    make_legend(plt.gca(), ['b','r'], ['neurometric','psychometric'], bbox_to_anchor=[.05,.95])
    return plt.gca()

def block_neurometric_plot(model,image,behavior,frame_df,plot_type='choice'):
    plt.figure()
    plt.suptitle('neurometric and psychometric')
    colors=['r','b']
    for i,prob in enumerate([.2,.8]):
        use = behavior['probabilityLeft'] == prob 
        if plot_type == 'choice':
            uCons, pred,pred_se,actual,actual_se = make_neurometric_choice(model,image,behavior[use],frame_df[use],baseline_correct=True)
        if plot_type == 'stim':
            uCons, pred,pred_se,actual,actual_se = make_neurometric_stim(model,image,behavior[use],frame_df[use],baseline_correct=True)
        plt.plot(uCons,pred, colors[i],ls='--')
        plt.fill_between(uCons,pred-pred_se,pred+pred_se,color=colors[i],alpha=.3)
        plt.plot(uCons,actual,colors[i])
        plt.fill_between(uCons,actual-actual_se,actual+actual_se,color=colors[i],alpha=.3)
    plt.xlabel('Signed Contrast')
    plt.ylabel('Percent Choose Left')
    make_legend(plt.gca(), ['b','r'], ['right block','left block'], bbox_to_anchor=[.05,.95])
    show()    


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
        
        model = LogisticRegression(C=C,solver='saga',max_iter=100000,penalty='l2',fit_intercept=True,class_weight='balanced')
        model.fit(train_X,train_Y)
        print(np.mean(model.predict(train_X)==train_Y))
        prediction = model.predict(test_X)
        test_results[k] = np.mean(prediction == test_Y)
        mods.append(model)
        k+=1
    test_SD = test_results.std(axis=0)
    test_results = test_results.mean(axis=0)
    model.coef_ = np.mean([mod.coef_ for mod in mods],axis=0)
    return test_results, test_SD, model

def make_neurometric_choice(model,downsampled_im,behavior,frame_df,n_pre=0,n_post=3,return_type='mean',baseline_correct=True):
    # frame_df = frame_df[behavior['signedContrast'] !=0].reset_index()  
    # behavior = behavior[behavior['signedContrast'] !=0].reset_index()
    choices = behavior['choice']
    choices = np.array(behavior['choice']).astype(float)
    choices[choices==0]=np.nan
    na_drop = ~np.isnan(choices)
    behavior=behavior.iloc[na_drop].reset_index()
    frame_df=frame_df.iloc[na_drop].reset_index()
    choices=choices[na_drop]    # choices[choices == 1] = 0
    choices[choices == -1] = 0
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
            X,_ = format_for_logreg_choice(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],n_pre=n_pre,n_post=n_post,baseline_correct=baseline_correct)
            
            predicted_choices[i] = np.mean(model.predict(X))
            predicted_se[i] = np.std(model.predict(X))/np.sqrt(len(stim_frames))
            actual_choices[i] = np.mean(choices[signed_contrast==con])
            actual_se[i] = np.std(choices[signed_contrast==con])/np.sqrt(len(stim_frames))
        return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se
    elif return_type == 'raw':
        predicted_choices = []
        actual_choices = []
        actual_se = []
        predicted_se = []
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,_ = format_for_logreg_choice(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],n_pre=n_pre,n_post=n_post,baseline_correct=baseline_correct)
            
            predicted_choices.append(model.predict(X))
            predicted_se.append(None)
            actual_choices.append(choices[signed_contrast==con])
            actual_se.append(None)
    return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se


def make_neurometric_stim(model,downsampled_im,behavior,frame_df,n_pre=0,n_post=3,return_type='mean',baseline_correct=True):
    frame_df = frame_df[behavior['signedContrast'] !=0].reset_index()  
    behavior = behavior[behavior['signedContrast'] !=0].reset_index()
    choices = behavior['choice']
    choices[choices==-1]=0
    signed_contrast = behavior['signedContrast'][behavior['signedContrast'] !=0]
    uContrasts,nContrasts = np.unique(signed_contrast,return_counts=True)
    stim_ons = frame_df['stimOn_times']
    if return_type == 'mean':
        predicted_choices = np.zeros(len(uContrasts))
        actual_choices = np.zeros(len(uContrasts))
        actual_se = np.zeros(len(uContrasts))
        predicted_se = np.zeros(len(uContrasts))
        accuracy = np.zeros(len(uContrasts))
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,Y = format_for_logreg(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],n_pre=n_pre,n_post=n_post,use_choice=None,baseline_correct=baseline_correct)
            
            predicted_choices[i] = np.mean(model.predict(X))
            accuracy[i] = model.score(X,Y)
            
            predicted_se[i] = np.std(model.predict(X))/np.sqrt(len(stim_frames))
            actual_choices[i] = 1-np.mean(choices[signed_contrast==con])
            actual_se[i] = np.std(choices[signed_contrast==con])/np.sqrt(len(stim_frames))
        print(np.mean(accuracy))
        return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se
    elif return_type == 'raw':
        predicted_choices = []
        actual_choices = []
        actual_se = []
        predicted_se = []
        for i,con in enumerate(uContrasts):
            stim_frames = stim_ons[signed_contrast==con]
            X,_ = format_for_logreg(behavior.iloc[stim_frames.index],downsampled_im,frame_df.iloc[stim_frames.index],n_pre=n_pre,n_post=n_post,use_choice=None,baseline_correct=baseline_correct)
            
            predicted_choices.append(model.predict(X))
            predicted_se.append(None)
            actual_choices.append(1-choices[signed_contrast==con])
            actual_se.append(None)
    return uContrasts, predicted_choices, predicted_se, actual_choices,actual_se

##################### start annalysis###########################################################
##load data
subject = 'CSK-im-002'
baseDir = '/grid/zador/data_nlsas_norepl/Chris/imaging_data'
os.chdir(pjoin(baseDir,subject))
dates = os.listdir()
# dates= ['2021-01-17']
ims = []
trials = []
frames = []
stacks = []
plot_dates = []
ds_factor=5
for date in dates:
    try:
        behavior, frame_df, stack, ccf_regions, atlas = get_ses_data(subject,date,baseDir=baseDir,FlatIron='/grid/zador/home/ckrasnia/FlatIron/zadorlab/Subjects')
    except:
        print('data loading failed, skipping session')
        continue
    plot_dates.append(date)
    downsampled_im = spatial_down_sample(stack,ds_factor)
    #compile all the data for one mouse into one
    trials.append(behavior)
    ims.append(downsampled_im)
    frames.append(frame_df)
    im_shape = downsampled_im.shape[1:]
    stacks.append(stack)
    brain_mask = downsampled_im[0]!=0
brain_map = ccf_regions[['left_x','left_y','right_x','right_y']]
ds_map = brain_map.applymap(np.array)
ds_map = ds_map / ds_factor
ds_atlas = downsample_atlas(atlas,ds_factor)

train_ses = 2
behavior = trials[train_ses]
downsampled_im = ims[train_ses]
frame_df = frames[train_ses]
brain_mask = downsampled_im[0,:,:] !=0
flat_mask = brain_mask.flatten()
[train_X,test_X],[train_Y,test_Y] = format_for_logreg(behavior,downsampled_im,frame_df,use_choice=None, split=True)

c_list = [.2,.3,.4,.5,.6,.7,.8,.9,1,5,7]
n_samples = 1000
pix_accuracy = np.zeros((len(c_list),n_samples))
mirror_accuracy = np.zeros((len(c_list),n_samples))
L1_accuracy = np.zeros((len(c_list),n_samples))
num_pix = []

for i,C in enumerate(c_list):
    model = LogisticRegression(C=C,solver='liblinear',max_iter=100000,penalty='l1',fit_intercept=True,class_weight='balanced')
    for sample_num in tqdm(range(n_samples)):
        [train_X0,test_X0],[train_Y0,test_Y0] = format_for_logreg(behavior,downsampled_im,frame_df,use_choice=None, split=True)
        model.fit(train_X0,train_Y0)
        L1_accuracy[i,sample_num] = model.score(test_X0,test_Y0)

    pix_num = np.sum(model.coef_!=0)
    coefs = model.coef_.reshape(im_shape)
    num_pix.append(pix_num)

    for sample_num in range(n_samples):
        if pix_num ==0:
            pix_accuracy[i,sample_num] = .5
            continue
        # first do for random pixels
        use_pix = np.random.choice(np.arange(flat_mask.shape[0])[flat_mask],pix_num,replace=False)
        train_X1 = train_X0[:,use_pix]
        test_X1 = test_X0[:,use_pix]
        model = LogisticRegression(C=1,solver='liblinear',max_iter=100000,penalty='l2',fit_intercept=True,class_weight='balanced')
        model.fit(train_X1,train_Y0)
        pix_accuracy[i,sample_num] = model.score(test_X1,test_Y0)
        # now fit for random mirrored pixels
        if pix_num ==1:
            mirror_accuracy[i,sample_num] = .5
            continue
        half_pix_num = int(pix_num/2)
        use_half_pix = np.random.choice(use_pix,half_pix_num,replace=False)
        mirror_pix = []
        for pix in use_half_pix:
            pix_x = pix % im_shape[1]
            pix_y = int(pix / im_shape[1])
            new_x = 32-pix_x
            new_pix = 32*pix_y + new_x-1
            mirror_pix.append(new_pix)
        mirror_pix = np.concatenate([mirror_pix,use_half_pix])
        train_X2 = train_X[:,mirror_pix]
        test_X2 = test_X[:,mirror_pix]
        model = LogisticRegression(C=1,solver='liblinear',max_iter=100000,penalty='l2',fit_intercept=True,class_weight='balanced')
        model.fit(train_X2,train_Y)
        # mirror_map = model.coef_.reshape(im_shape)
        empty_brain = flat_mask.astype(int)
        empty_brain[mirror_pix] = 2
        mirror_brain = empty_brain.reshape(im_shape)
        mirror_accuracy[i,sample_num] = model.score(test_X2,test_Y)
np.save('L1_accuracy.npy',L1_accuracy)
np.save('pix_accuracy.npy',pix_accuracy)
np.save('mirror_accuracy.npy',mirror_accuracy)

L1_mean = L1_accuracy.mean(axis=1)
L1_std = L1_accuracy.std(axis=1)
pix_mean = pix_accuracy.mean(axis=1)
pix_std = pix_accuracy.std(axis=1)
mirror_mean = mirror_accuracy.mean(axis=1)
mirror_std = mirror_accuracy.std(axis=1)
fig = plt.figure()
plt.plot(num_pix,L1_mean, label='L1 accuracy',color='red')
plt.fill_between(num_pix,L1_mean+L1_std,L1_mean-L1_std,color='red',alpha=.3)
plt.plot(num_pix,pix_mean, label='random accuracy',color='blue')
plt.fill_between(num_pix,pix_mean+pix_std,pix_mean-pix_std,color='blue',alpha=.3)
plt.plot(num_pix,mirror_mean, label='bilateral random accuracy',color='green')
plt.fill_between(num_pix,mirror_mean+mirror_std,mirror_mean-mirror_std,color='green',alpha=.3)
plt.legend()
plt.xlabel('number of pixels')
plt.ylabel('accuracy')
fig.savefig('random vs L1.png')
# show()
