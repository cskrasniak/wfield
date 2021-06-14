### select the mask for which there is actually reasonable signal with interactive plotting


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
