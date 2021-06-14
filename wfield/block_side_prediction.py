## Try to predict which block the mouse is in from whole brain trials

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
from analyses import *
from sklearn.linear_model import LogisticRegression
from scipy import stats 

subject = 'CSK-im-001'
baseDir = r'F:\imaging_data'
os.chdir(pjoin(baseDir,subject))
dates = os.listdir()
## iterate through dates for each subject
date = '2021-01-17'
area_activities = []
X_svt_list = []
U_list = []
con_list = []
y_list = []
stim_ons = []
perf_list = []
block_sides = []
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
    SVT = SVTcorr# load  normal temporal components
    onset_times = extract_onset_times(localdisk)

    # the allen atlas and brain mask for finding which pixels corresponds to which area
    atlas, area_names, brain_mask = atlas_from_landmarks_file(pjoin(localdisk,'dorsal_cortex_landmarks.JSON'),do_transform=False)
    ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

    #the transform to apply to the images
    transform = load_allen_landmarks(pjoin(localdisk, 'dorsal_cortex_landmarks.JSON'))['transform']
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

    stimOn_frames = np.array(frame_df['stimOn_times'])
    block_side = np.array(behavior['probabilityLeft'])
    pre = 5
    post = 15
    X_svt = np.empty((len(stimOn_frames),200,pre+post))
    for i,frame in enumerate(stimOn_frames):
        X_svt[i,:] = SVTcorr[:,frame-pre:frame+post]
    y = np.sign(behavior['signedContrast'])#np.sign(behavior['signedContrast'])
    perf = np.array(behavior['feedbackType'])
    cons = behavior['signedContrast']
    y=np.array(y)
    cons = np.array(cons)
    
    y[y==-1]=0

    X_svt_list.append(X_svt)
    con_list.append(cons)
    y_list.append(y)
    perf_list.append(perf)
    U_list.append(stack.U_warped)
    stim_ons.append(stimOn_frames)
    block_sides.append(block_side)
    unique_areas =np.unique(atlas)
    #### THIS TAKES AWHILE! this creates the extracted activity for each brain area per frame
    #### Only use this if it hasn't been done yet, otherwise 
    allen_list = ccf_regions['acronym']
    right_areas = [string + ' right' for string in allen_list.iloc[::-1].tolist()]
    left_areas = [string + ' left' for string in allen_list.tolist()]
    full_allen_list = right_areas + ['None'] + left_areas
    area_activity = pd.DataFrame(np.empty((SVT.shape[1],len(unique_areas))),columns=full_allen_list)
    old=0 
    for i in tqdm(np.linspace(SVT.shape[1]/200,SVT.shape[1],200).astype(int)):
        frames = SVT[:,old:i]
        
        vid = reconstruct(stack.U_warped,frames)
        area_activity.iloc[old:i] = np.array([vid[:,atlas==area].mean(axis=1) for area in np.arange(-33,34)]).T
        old = i
    area_activity.to_csv('area_activity')

    area_activities.append(area_activity)

#load area_activities if it doesnt exist
if not area_activities:
    area_activities = []
    for idx,date in enumerate(dates[:-1]):
        os.chdir(pjoin(baseDir,subject))
        localdisk = pjoin(os.getcwd(),date)
        os.chdir(localdisk)
        if not os.path.isfile('area_activity'):
            continue
        area_activities.append(pd.read_csv('area_activity'))

tot_trials=0
for i in y_list:
    tot_trials+=len(i)


## get all the trials etc
pre = 20
post = 40
trials = np.zeros((pre+post,tot_trials,67))
Ys = np.zeros(tot_trials)
cons = np.zeros(tot_trials)
perf = np.zeros(tot_trials)
blocks = np.zeros(tot_trials)
start = 0
stop=0
for ses in range(len(y_list)):
    so = stim_ons[ses]   
    stop += len(so)
    Ys[start:stop] = y_list[ses]
    cons[start:stop] = con_list[ses]
    perf[start:stop] = perf_list[ses]
    act_temp = area_activities[ses]
    if 'Unnamed: 0' in act_temp.keys():
        act_temp = act_temp.drop('Unnamed: 0',axis=1)
    blocks[start:stop] = block_sides[ses]
    blocks[blocks==.2]=1 #right block
    blocks[blocks == .8]=0 #left Block
    # act_temp = act_temp.drop('Unnamed: 0',axis=1)
    for i,trial in enumerate(so):
        trials[:,i+start,:] = act_temp.iloc[trial-pre:trial+post,-67:]
    start=stop
perf[perf==-1]=0
##### Select which contrasts to use#######
condition = (cons!=0) & (abs(cons)>.5) 
Xs = trials[:,condition,:]
Ys = np.sign(cons[condition]) #cons>0 = right
Ys[Ys==-1]=0
perf = perf[condition]

#subsample to match number in each condition, or else just permute the samples
if abs(.5-np.mean(Ys))>.05:
    vals, cnts = np.unique(Ys,return_counts=True)
    least = np.argmin(cnts)
    most = np.argmax(cnts)
    common = np.argwhere(Ys==vals[most]).flatten()
    rare = np.argwhere(Ys==vals[least]).flatten()
    keepers = np.concatenate((rare,np.random.choice(common,cnts[least],replace=False)))
    p = np.random.permutation(len(keepers))
    keepers = keepers[p]
    Xs = Xs[:,keepers,:]
    Ys = Ys[keepers]
else:
    p = np.random.permutation(len(Ys))
    Xs = Xs[:,p,:]
    Ys = Ys[p]
##### exclude the 50/50 blocks ###########
# Ys = blocks[blocks!=.5]
# Xs = trials[:,blocks!=.5,:]

u_areas = np.unique(atlas)

############ fit logreg model w/ kfold cross-val
from sklearn.model_selection import KFold
K = 20
kf = KFold(K)
kf.get_n_splits(Ys)
C=7

area_mask = u_areas!=0
k=0
mods = []
test_results = np.zeros((K,train_X.shape[0]))
for train,test in tqdm(kf.split(Ys)):
    fmods = []
    train_X, test_X = Xs[:,train,:],Xs[:,test,:]
    train_Y, test_Y = Ys[train],Ys[test]
    for frame in range(train_X.shape[0]):
        model = LogisticRegression(C=C,solver='liblinear',max_iter=100000,penalty='l1')
        model.fit(train_X[frame,:,area_mask].T,train_Y)
        prediction = model.predict(test_X[frame,:,area_mask].T)
        test_results[k,frame] = np.mean(prediction == test_Y)
        fmods.append(model)
    mods.append(fmods)
    k+=1
test_SD = test_results.std(axis=0)
test_results = test_results.mean(axis=0)

##### now shuffle the labels and do the same fitting to have a floor level

p2 = np.random.permutation(len(Ys))
Ys_shuff = Ys[p2]

k=0
mods = []

shuff_results = np.zeros((K,train_X.shape[0]))
for train,test in tqdm(kf.split(Ys_shuff)):
    fmods = []
    train_X, test_X = Xs[:,train,:],Xs[:,test,:]
    train_Y, test_Y = Ys_shuff[train],Ys_shuff[test]
    for frame in range(train_X.shape[0]):
        model = LogisticRegression(C=C,solver='liblinear',max_iter=100000,penalty='l1')
        model.fit(train_X[frame,:,area_mask].T,train_Y)
        prediction = model.predict(test_X[frame,:,area_mask].T)
        shuff_results[k,frame] = np.mean(prediction == test_Y)
        fmods.append(model)
    mods.append(fmods)
    k+=1
shuff_SD = shuff_results.std(axis=0)
shuff_results = shuff_results.mean(axis=0)

###### Plot the training results ##################
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(pre+post),test_results, 'r')
ax.fill_between(np.arange(pre+post), test_results+test_SD, test_results-test_SD,alpha=.5,color='r')
ax.plot(np.arange(pre+post),shuff_results,'b')
ax.fill_between(np.arange(pre+post), shuff_results+shuff_SD, shuff_results-shuff_SD,color='b',alpha=.5)
ax.set_xlabel('time (s)')
ax.set_xticks(np.linspace(0,pre+post,6))
ax.set_xticklabels(((np.linspace(-pre,post,6)/30).round(2)))
ax.set_ylabel('Decoding Accuracy on test set')
ax.set_ylim(.4,1)
ax.axvline(pre,color='k')
ax.axhline(np.mean(perf),color='k',ls = '--')
make_legend(ax,['r','b'],['test','shuffled'],location='upper right',bbox_to_anchor=(.99,.99))
show()

##### map the weights onto the allen atlas and show them ###########
mods = np.array(mods)
best_coefs = np.array([mod.coef_[0] for mod in mods[:,np.argmax(test_results)]]).mean(axis=0)
# best_coefs = np.array([mod.coef_[0] for mod in mods[:,10]]).mean(axis=0)
allen_list = ccf_regions['acronym']
right_areas = [string + ' right' for string in allen_list.iloc[::-1].tolist()]
left_areas = [string + ' left' for string in allen_list.tolist()]
full_allen_list = right_areas + ['None'] + left_areas
full_allen_list = np.array(full_allen_list)[area_mask]
acronym_dict = {}
for i,area in enumerate(full_allen_list):
    acronym_dict[area] = best_coefs[i]
coef_dict = {}
coef_atlas = np.copy(atlas)
cnt=0
for area in np.arange(-33,34)[area_mask]: 
    if area in u_areas[area_mask]:
        coef_atlas[coef_atlas == area] = best_coefs[cnt]
        cnt+=1
    else:
        coef_atlas[coef_atlas == area] = 0.0
    
fig=plt.figure(figsize=[6,  5])
ax2 = fig.add_axes([.85,0.05,.025,.9])
ax1 = fig.add_axes([0.0,0.0,.8,1])
norm = mplc.TwoSlopeNorm(vcenter=0,vmax = np.max(coef_atlas))#vmin = np.min(coef_atlas),
ax1.imshow(coef_atlas, cmap='RdBu_r',norm=norm)
cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.get_cmap('RdBu_r'),
                                norm=norm)
cb1.set_label('Regression weight')
ax1.axis('off')
show()

## visualize what this activity actually looks like when you break it down into areas like this
mean_act_l = Xs[:,Ys=0,:].mean(axis=1)
mean_act_r = Xs[:,Ys==1,:].mean(axis=1)

movie_l = np.empty((mean_act_l.shape[0],540,640))
movie_r = np.empty((mean_act_r.shape[0],540,640))
for area in np.arange(-33,34)[area_mask]: 
    for frame in range(movie_l.shape[0]):
        movie_l[frame,area==atlas] = mean_act_l[frame,area]
    for frame in range(movie_r.shape[0]):
        movie_r[frame,area==atlas] = mean_act_r[frame,area]
movie_r[pre:pre+3,0:50,0:50] = .03
movie_l[pre:pre+3,0:50,0:50] = .03
nb_save_movie(movie_l,filename='test.avi',clim=[-.03,.03],cmap='RdBu_r')


### Do the same predictions, but use LocaNMF components as predictors and look at their weights
### could also do leave-one-out and look for unique contribution
subject = 'CSK-im-001'
date = '2021-01-27'
localdisk = r'F:\imaging_data\CSK-im-001\2021-01-27'
os.chdir(localdisk)
C = np.load('C.npy')
A = np.load('A.npy')
areas = np.load('LocaNMF_areas.npy')
lambdas = np.load('lambdas.npy')

behavior = fetch_task_data(subject,date)
behavior = behavior[behavior['choice']!=0].reset_index() #drop trials where there was no response

# the allen atlas and brain mask for finding which pixels corresponds to which area
atlas = np.load('atlas.npy')
ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

#sync up the behavior and grab a few things for analysis
sync_behavior = sync_to_task(localdisk)
mask = ['times' in key for key in behavior.keys()]
time_df = behavior[behavior.keys()[mask]]
frame_df = pd.DataFrame(columns=time_df.keys())
for (columnName, columnData) in time_df.iteritems():
    frame_df[columnName] = time_to_frames(sync_behavior,localdisk,np.array(columnData),dropna=False)
frame_df = frame_df.astype(np.int64)

stimOn_frames = np.array(frame_df['stimOn_times'])
block_side = np.array(behavior['probabilityLeft'])
pre = 5
post = 15
y = np.sign(behavior['signedContrast'])#np.sign(behavior['signedContrast'])
perf = np.array(behavior['feedbackType'])
cons = behavior['signedContrast']
y=np.array(y)
cons = np.array(cons)
y[y==-1]=0
#create an array for training that is len trials x len window, for each component
p = np.random.permute(len(stimOn_frames))
x = np.empty((len(stimOn_frames),pre+post,C.shape[0]))
for comp in range(C.shape[0]):
    for i,frame in enumerate(stimOn_frames):
        x[i,:,comp] = C[comp,frame-pre:frame+post]
X = x[p];train_X = X[:-100]; test_X = X[100:]
Y = y[p];train_Y = Y[:-100]; test_Y = Y[100:]

test_results = np.empty(train_X.shape[1])
for frame in range(train_X.shape[1]):
    model = LogisticRegression(C=100,solver='liblinear',max_iter=100000,penalty='l2')
    model.fit(train_X[:,frame,:],train_Y)
    prediction = model.predict(test_X[:,frame,:])
    test_results[frame] = np.mean(prediction == test_Y)
plt.figure()
plt.plot(np.arange(train_X.shape[1]),test_results)
show()
