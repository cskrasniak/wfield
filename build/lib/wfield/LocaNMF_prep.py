###script for getting everything in place to run locaNMF

import os
import numpy as np
import pandas as pd
from  wfield import *
subject = 'CSK-im-001'
date = '2021-01-20'
localdisk = 'F:\\imaging_data\\'+'\\'+subject+'\\'+date
# localdisk = r'F:\imaging_data\CSK-im-001\2021-01-27'
os.chdir(localdisk)

# the allen atlas and brain mask for finding which pixels corresponds to which area
atlas, area_names, brain_mask = atlas_from_landmarks_file(pjoin(localdisk,'dorsal_cortex_landmarks.JSON'),do_transform=False)
trans_atlas, trans_names, trans_mask = atlas_from_landmarks_file(pjoin(localdisk,'dorsal_cortex_landmarks.JSON'),do_transform=True)
ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

lmarks = load_allen_landmarks(pjoin(localdisk, 'dorsal_cortex_landmarks.JSON'))
nref_regions = allen_transform_regions(None,ccf_regions,
                                resolution=lmarks['resolution'],
                                bregma_offset=lmarks['bregma_offset'])
os.chdir(localdisk)

U = np.load('U.npy')
SVT = np.load('SVTcorr.npy')
stack = SVDStack(U,SVT)
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
np.save('U_warped.npy',stack.U_warped)
np.save('brain_mask.npy',mask)
np.save('atlas.npy',atlas)