from __future__ import print_function
from re import L

import warnings
# from hmm import *
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import shutil
import random
import pickle
# from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import glob

# import tf and keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import set_session, clear_session, get_session
from tensorflow.keras import backend as K
from tensorflow import keras  # NOTE: for using TEST venv you should import keras by tensorflow backend

# import sklearn 
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import sklearn.metrics as metrics
from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import ADASYN
# from sklearn.preprocessing import OneHotEncoder
from random import shuffle
from get_mad_dbx import *
mad_pfx, dbx_pfx, feature_folder, mem_lim = get_mad_dbx()
# feature_folder = '/media/snasiri/Expansion/CAISR data/Feature_files_Robert_v2/'
dbx_pfx = '/home/ubuntu/snasiri/CAISR/Sleep Staging/'
sys.path.insert(1, dbx_pfx + 'CAISR_Codes/main_CAISR_functions/')
from load_features_and_labels_updated import *
from retrieve_LOIO_patient_fold import *
# from load_features_and_labels_only_EEG import *
from scipy import signal
# import arnn_model as arnn

# sys.path.insert(1, dbx_pfx + 'CAISR_Codes/GraphNN_SleepStaging/')
from ProductGraphSleepNet import *
from utils import *

from get_mad_dbx import *
_, dbx_pfx, feature_folder, _ = get_mad_dbx()

fold = 0 
path1 = '/bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/'
model_type =  'graphsleepnet' 

# define global vars
optimizer = 'adam'
# learn_rate = 0.001
# lr_decay = 0.0
# l1, l2 = 0, 0

learn_rate = 0.0001
lr_decay = 0.0
l1, l2 = 0.001, 0.001

# check optimizer（opt）
if optimizer=="adam":
    opt = keras.optimizers.Adam(lr=learn_rate, decay=lr_decay, clipnorm=1)
elif optimizer=="RMSprop":
    opt = keras.optimizers.RMSprop(lr=learn_rate, decay=lr_decay)
elif optimizer=="SGD":
    opt = keras.optimizers.SGD(lr=learn_rate, decay=lr_decay)
else:
    assert False,'Config: check optimizer'

# set l1、l2（regularizer）
if l1!=0 and l2!=0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1!=0 and l2==0:
    regularizer = keras.regularizers.l1(l1)
elif l1==0 and l2!=0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None

# dataset = ['penn']
# if dataset[0] == 'penn':
#     w = 2  # width
# else:
#     w = 6
w = 7
h = 9   # height
context = 7 # 21 Note that it should be odd number 
sample_shape = (context, w, h)  
conf_adj = 'GL'
GLalpha = 0.0
num_of_chev_filters = 128 # 32 or 64, 128
num_of_time_filters = 128   # 32 or 64 , 128 
time_conv_strides = 1   
time_conv_kernel = 3
num_block = 1
cheb_k = 3   # 3, 5
cheb_polynomials = None  
dropout = 0.60   # 0.6, 0.75 or 0.8
GRU_Cell = 256  # 256, 512  or 1024
attn_heads = 40   # 20, 64 or 128, 256

    
def load_prepared_data(file_path:str, get_signals:list=[]):
    """ 
    load prepared data format (post March 22, 2023). Currently, all signals are read. 
    Inputs:
    file_path: path to prepared .h5 file.
    """
    # init DF
    Xy = pd.DataFrame([])
    
    # read file
    with h5py.File(file_path, 'r') as f:
        # Loop over each group and dataset within each group and get data
        group_names = list(f.keys())
        for group_name in group_names:        
            group = f[group_name]
            # list all existing datasets
            dataset_names = list(group.keys())
            # get all, or take from get_signals
            get_names = dataset_names if len(get_signals)==0 else [c for c in dataset_names if c in get_signals]
            for dataset_name in get_names:
                dataset = group[dataset_name][:]
                assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
                Xy[dataset_name] = dataset.flatten()
        
        # save attributes
        params = {}
        for attrs in f.attrs.keys():
            params[attrs] = f.attrs[attrs]
    return Xy, params

def evaluations_matrix(gs, pred):
    fpr, tpr, thresholds = metrics.roc_curve(gs, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(gs, pred)
    return auroc, auprc

def evaluations_matrix_binary(gs, pred):
    """
    Params
    ------
    gs: gold standards
    pred: predicted binary labels 
    Yields
    ------
    sensitivity
    specificity
    precision
    F1-score
    auprc_baseline: P/N in all samples
    """
    # kappa = metrics.cohen_kappa_score(gs, pred)
    conf_mat = confusion_matrix(gs, pred)
    tn, fp, fn, tp = conf_mat.ravel()
    auprc_baseline = (fn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(tp+fn) # recall
    specificity =tn/(tn+fp)
    precision = tp/(tp+fp)
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    F1= 2*(sensitivity*precision)/(sensitivity+precision)
    return conf_mat, sensitivity, specificity, precision, accuracy, F1, auprc_baseline

def predict(model_path, spec_path, sig_path, lab_path, fold):

    dataset = ['koges_robert_triple']

    os.makedirs(spec_path, exist_ok=True) 
    os.makedirs(spec_path + 'csv/', exist_ok=True) 
    # (tr_line, tr_cohort), (va_line, va_cohort), (test_line, test_cohort_line) = get_LOIO_cohorts(LOIO=fold)
    # df_self_data = pd.read_csv('/home/snasiri/Dropbox (Partners HealthCare) (copy)/CAISR_Codes/csv_files/files_per_dataset/files_few_shot_robert_fold2.csv')
    
    path = '/bdsp/staging/BDSP-Sleep/CAISR/data/prepared_data/'
    # test_line = os.listdir(path + dataset[0])

    # path_to_csv_few_shot = '/bdsp/staging/BDSP-Sleep/CAISR/data/train_val_test_splits/Koges_few_shot/'
    # df_fold = pd.read_csv(path_to_csv_few_shot + 'fold_2.csv')
    # test_line = df_fold['prepared_file'].values.tolist()

    path_to_csv_few_shot = '/bdsp/staging/BDSP-Sleep/CAISR/data/train_val_test_splits/Koges_few_shot_triple/'
    df_fold = pd.read_csv(path_to_csv_few_shot + 'fold_2.csv')
    test_line = df_fold['files'].values.tolist()

    test_cohort_line = dataset * len(test_line)
    # os.chdir(path)
    # files = glob.glob('*.h5')
    cohort = 'koges_robert_triple'
    df_self_data = pd.DataFrame({'0': test_line, 'dataset': test_cohort_line})
    # test_line = df_self_data['0'].values.tolist()
    # test_cohort_line = df_self_data['dataset'].values.tolist()
    # import pdb; pdb.set_trace()
    model = build_ProductGraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, 
                                        time_conv_kernel, sample_shape, num_block, opt, conf_adj=='GL', GLalpha, regularizer, 
                                            GRU_Cell, attn_heads, dropout)

    model.load_weights(model_path + 'weights_fold_3.h5')
    
    pred_all = []
    gs_all = []
    sleep_conf_mat = np.zeros((5,5)) 
    labels = ['N3', 'N2', 'N1', 'REM', 'Wake'] 
    # import pdb; pdb.set_trace()
    for i in range(0, len(test_line)):
        # import pdb; pdb.set_trace()
        sample = test_line[i]
        the_id=sample.split('/')[-1]
        the_id = the_id[:-3]
        dataset = test_cohort_line [i]
        print('sample: %s, cohort: %s, i=%s'%(sample, dataset, i))
        data_input, params = load_prepared_data(path + dataset + '/' + the_id + '.h5')
        # import pdb; pdb.set_trace()
        # try:
        # all_test_eva = open(spec_path + dataset + '_'+ the_id + '_eva_test_graphsleepnet.txt', 'w')
        # all_test_eva.write('id\tlabel\tAUROC\tAUPRC\tSensitivity\tSpecificity\tPrecision\tAccuracy\tF1\tauprc_baseline\ttotal_length\n')
        try:
            # 1: N3, 2: N2, 3: N1, 4: REM, 5: Wake
            image, label,  _ = load_features_and_labels(sig_path, lab_path, dataset, the_id)
            
            # import pdb; pdb.set_trace()
            experts = label
            # label = np.squeeze(label[:,0])
            
            # conmvert Nan Values to 0
            # for i in range(len(label)):
            #     if label[i] not in [1,2,3,4,5]:
            #         label[i] = 0
            # import pdb; pdb.set_trace()
            # if dataset in ['mgh', 'robert']:
            #     channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
            #     # common_channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
            #     common_channels = channels
            #     idx = [i for i, e in enumerate(channels) if e in set(common_channels)] 
            #     image = image [:, idx, :]
            # indices = np.where(label == 0) [0].tolist()
            #after adding context; we are not caring about 
            # cut = int(context/2)
            # remining_index_after_adding_context =np.arange(cut,label.shape[0]-cut)


            # sleep_stage_max = 6
            # label = np.eye(sleep_stage_max)[label.tolist()] 
            # label = label [:,1:6]
            # import pdb; pdb.set_trace()

            image = AddContext (image , context)
            image = np.squeeze(np.array(image))
            #  print (image.shape)

            # label = AddContext (label , context, label = True)
            # label = np.squeeze(np.array(label))
            # print (label.shape)
        except:
            continue

        # evaluation
        prediction = model.predict(image)
        # import pdb; pdb.set_trace()
        # np.save(spec_path + the_id +'_stage_prediction', prediction)
        # np.save(spec_path + dataset + '_'+ the_id+'_sleep_actual_graphsleepnet', label)
        # np.save(spec_path + the_id +'_stage_gold', experts)
        # import pdb; pdb.set_trace()
        # gs_per_subject = gt_subj.argmax(axis = 1) + 1
        pred_per_subject = prediction.argmax(axis = 1) + 1
        
        fs_input = 200 
        output_Hz = 1
        task = 'stage'
        stage_length = 30 # sec

        # gs_per_subject = np.concatenate([[np.nan]*3, gs_per_subject, [np.nan]*3])
        pred_per_subject = np.concatenate([[np.nan]*3, pred_per_subject, [np.nan]*3])
        
        # gs_per_subject = np.repeat(gs_per_subject, stage_length, axis=0)
        pred_per_subject = np.repeat(pred_per_subject,stage_length,axis=0)
        
        nan_row = np.empty((1, prediction.shape[1]))
        nan_row[:] = np.nan
        # import pdb; pdb.set_trace()
        indices = [0,1,2] # graph doesn't predict for the first/last three epochs
        # Iterate over the list of indices
        for index in indices:
            # gt_subj = np.insert(gt_subj, index, nan_row, axis=0)
            prediction = np.insert(prediction, index, nan_row, axis=0) 
        # import pdb; pdb.set_trace()
        for i in range(3):
            # gt_subj = np.vstack((gt_subj, nan_row)) 
            prediction = np.vstack((prediction, nan_row)) 
        # import pdb; pdb.set_trace()
        # gt_subj = np.repeat(gt_subj, stage_length, axis=0)
        prediction = np.repeat(prediction,stage_length,axis=0)
        
        t1 = ((np.arange(len(prediction)))*fs_input).astype(int)
        t2 = ((np.arange(len(prediction))+1)*fs_input).astype(int)
        # import pdb; pdb.set_trace()
        df = pd.DataFrame({'start_idx':t1,'end_idx':t2,'stage':pred_per_subject,'prob_n3':prediction[:,0],'prob_n2':prediction[:,1],'prob_n1':prediction[:,2],'prob_r':prediction[:,3],'prob_w':prediction[:,4]})
        
        t1 = ((np.arange(len(data_input)/fs_input))*fs_input).astype(int)
        t2 = ((np.arange(len(data_input)/fs_input)+1)*fs_input).astype(int)

        # import pdb; pdb.set_trace()
        df_matched = pd.DataFrame({'start_idx':t1,'end_idx':t2,'stage':np.nan,'prob_n3':np.nan,'prob_n2':np.nan,'prob_n1':np.nan,'prob_r':np.nan,'prob_w':np.nan})
        df_matched.iloc[0:len(df)] = df
        spec_path = '/bdsp/staging/BDSP-Sleep/CAISR/caisr_output/koges_robert_triple/stage/graph_2023_05_17_fold_2/'
        os.makedirs(spec_path, exist_ok=True) 
        df_matched.to_csv(spec_path + the_id + '_stage.csv', index = False)
        # df.to_csv(spec_path + the_id + '_stage.csv')
        # import pdb; pdb.set_trace()


def main():
        parser = argparse.ArgumentParser(description = 'sleep staging model test program',
            usage = 'use "python %(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('--model_path', type=str, default = '/home/ubuntu/snasiri/Stage/Models/graphsleepnet_ self-testing_SS_korean_triple_fold_2_context_7_batch_10_atten_40_epoch_500/',
        help = 'signal path, default = /home/ubuntu/snasiri/Stage/Models/graphsleepnet_ self-testing_SS_korean_triple_fold_2_context_7_batch_10_atten_40_epoch_500/')
        
        # parser.add_argument('--model_path', type=str, default = '/home/snasiri/Documents/Projects_New/zz_SLEEP/CAISR1/Models/graphsleepnet_ self-testing_few_shot_dice_tr_RT_sep26_fold_2_calibrate_context_7_batch_10_atten_40_epoch_1000/',
        # help = 'signal path, default = /home/snasiri/Documents/Projects_New/zz_SLEEP/CAISR1/Models/graphsleepnet_ self-testing_few_shot_dice_tr_RT_sep26_fold_2_calibrate_context_7_batch_10_atten_40_epoch_1000/')
        
        # parser.add_argument('--model_path', type=str, default = '/home/snasiri/Documents/models/graphsleepnet_ self-testing test_RT_few_shot_context_7_batch_10_atten_40_epoch_1000/',
        # help = 'signal path, default = /home/snasiri/Documents/models/graphsleepnet_ self-testing test_RT_few_shot_context_7_batch_10_atten_40_epoch_1000/')
        # parser.add_argument('--model_path', type=str, default = '/home/snasiri/Documents/models/graphsleepnet_ self-testing test__RT_context_7_batch_20_atten_40_v3/',
        # help = 'signal path, default = /home/snasiri/Documents/models/graphsleepnet_ self-testing test__RT_context_7_batch_20_atten_40_v3/')

        

        parser.add_argument('--spec_path', type=str, default = '/home/ubuntu/snasiri/CAISR_pred/gcn_ss/koges_robert_triple_20230517_fold_2/',
        help = 'spec path, default = /home/ubuntu/snasiri/CAISR_pred/gcn_ss/koges_robert_triple_20230517_fold_2/')

        parser.add_argument('--sig_path', type=str, default = '/bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/',
        help = 'signal path, default = /bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/')

        # parser.add_argument('--sig_path', type=str, default = 's3://bdsp-opendata-staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/',
        # help = 'signal path, default = s3://bdsp-opendata-staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/')

        # parser.add_argument('--sig_path', type=str, default = '/media/snasiri/Expansion/CAISR data/Feature_files_Robert_v2/',
        # help = 'signal path, default = /media/snasiri/Expansion/CAISR data/Feature_files_Robert_v2/')

        parser.add_argument('--lab_path', type=str, default = model_type ,
        help = 'label path, default = graphsleepnet')   
    
        parser.add_argument('-fold', type=int, 
        help = 'fold number; also used as seed to split train and val', default =0)

        args = parser.parse_args()
        opts = vars(args)

        predict(**opts)

if __name__ == '__main__':
    main()