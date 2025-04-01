from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import h5py
from scipy.signal import resample_poly
sys.path.insert(0, '..')
# sys.path.append('./sleep_general')
# from mgh_sleeplab import read_psg_from_bdsp, load_mgh_signal, annotations_preprocess, vectorize_respiratory_events, vectorize_sleep_stages, vectorize_arousals, vectorize_limb_movements
import numpy as np
import pandas as pd
import os
import h5py
import sys
import warnings
import matplotlib.pyplot as plt
from shutil import copyfile, rmtree
import mne
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from collections import Counter
from sklearn.preprocessing import RobustScaler
import sys
sys.path.insert(1, './CAISR/feature_extraction/graphsleepnet')
from DE_PSD import *
from gcn_features import *
sys.path.insert(1, './CAISR/feature_extraction')
from channel_selection import *
from utils_ss_koges import *

input_folder = '/bdsp/staging/BDSP-Sleep/CAISR/data/prepared_data/'
save_folder = '/bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/'
os.makedirs(save_folder, exist_ok=True)


dataset = 'koges_prepared'
# dataset = 'koges_robert_single'
# SET DATASET DEPENDENT VARIABLES
dataset_folder = dataset if dataset != 'mesa' else 'mesa_v2'
dataset_folder = dataset_folder if dataset != 'mros' else 'mros_v2'
dataset_folder = dataset_folder if dataset != 'mgh' else 'mgh_v3'

# set all out paths 
input_path = input_folder + dataset + '/' 
files = os.listdir(input_path)


graph_out_paths = set_out_paths(save_folder, dataset)
files_is_already_extracted = os.listdir(graph_out_paths[0])

result_files = [x for x in files if x not in files_is_already_extracted]
# import pdb; pdb.set_trace()
for num, each_id in enumerate(result_files):
    path = input_path + each_id
    print('\n==========================\n(#%s/%s) %s'%(num+1, len(result_files), dataset), each_id)
    out_paths = [p + each_id for p in graph_out_paths] 
    # if not os.path.isfile(out_paths[0]):
    try:
        signals, stages, Fs, sig_tages = \
            select_signals(dataset, dataset_folder, path)
        # import pdb; pdb.set_trace()
    except Exception as e:
        print(e)
        continue

    # segment EEG plus signals
    segs = segment_data_unseen_koges(signals, stages)
    # import pdb; pdb.set_trace()
    window = 30 # sec
    # import pdb; pdb.set_trace()
    
    try:
        out_paths = [p + each_id for p in graph_out_paths] 
        MYpsd, MYde = graph_feat_extraction_unseen(segs, sig_tages, Fs, window, out_paths)

        # save features
        dtype = 'float32'
        with h5py.File(out_paths[0], 'w') as f:
            dMYpsd = f.create_dataset('psd', shape=MYpsd.shape, maxshape=(MYpsd.shape), dtype=dtype, compression="gzip") 
            dMYde = f.create_dataset('de', shape=MYde.shape, maxshape=(MYde.shape), dtype=dtype, compression="gzip") 
            dMYpsd[:] = MYpsd
            dMYde[:] = MYde

        # save labels
        # dtype = 'int32'
        # with h5py.File(out_paths[1], 'w') as f:
        #     d = label
        #     dfile = f.create_dataset('label', shape=d.shape, maxshape=(d.shape), dtype=dtype, compression="gzip") 
        #     dfile[:] = d
    except Exception as error:
        print('(%s - %s) Failure feature extraction GRAPH: %s'%(num, dataset, error))
print("Graph features processed!")