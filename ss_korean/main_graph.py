import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras

from ProductGraphSleepNet import build_ProductGraphSleepNet

from get_mad_dbx import *
_, dbx_pfx, feature_folder, _ = get_mad_dbx()

dbx_pfx = '/home/ubuntu/snasiri/koges_caisr/'

# /media/snasiri/Expansion/Samaneh/Dropbox (Partners HealthCare) (copy)/CAISR_Codes/GraphNN_SleepStaging
sys.path.insert(1, dbx_pfx + 'main_CAISR_functions/')
from train_ss_task_koges import *
# from test_on_cohorts import *
from reset_keras import *

# define global vars
optimizer = 'adam'
# learn_rate = 0.0001
# lr_decay = 0.0
# l1, l2 = 0.001, 0.001

learn_rate = 0.001
lr_decay = 0.0
l1, l2 = 0, 0

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
    

def run(ifold, n, sig_path, lab_path, train_path):
    # set global paths
    # model_type = 'graphsleepnet_ self-testing_SS_korean_baseModel_Oct16_single_context_7_batch_5_atten_40_epoch_100'   # define model type: Note: modelName_...
    num_epochs = 100    # number of epochs
    batch_size = 10
    
   
    # build model
    print("*Build Graph NN")
    # w = 11  # width ONLY for MGH and Robert ( = #channels)
    # w = 7  # width ONLY for those that have 2 eeg channels
    # common_channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
    w = 7  # number of channels 
    h = 9   # height
    context = 7 # Note that it should be odd number 
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
    model_type = f'graphsleepnet_ self-testing_SS_koges_single_Nov7_2023_context_{context}_batch_{batch_size}_atten_{attn_heads}_epoch_{num_epochs}'   # define model type: Note: modelName_...
    
    model = build_ProductGraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, 
                                        time_conv_kernel, sample_shape, num_block, opt, conf_adj=='GL', GLalpha, regularizer, 
                                            GRU_Cell, attn_heads, dropout)

    # run over each of the LOIO folds
    # feature_folder = '/media/snasiri/Expansion/CAISR data/Feature_files_revised_v3/'
    # feature_folder = '/media/snasiri/Expansion/CAISR/Feature_files_Robert_v3/'
    feature_folder = '/bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/'
    # /media/snasiri/Seagate/CAISR/Resp/Feature_Files_NaN/graphsleepnet/features
    folds = []
    for ifold in range(3,4):
        # print (feature_folder)
        # import pdb; pdb.set_trace()
        train_ss_task(ifold, model, num_epochs, batch_size, feature_folder, model_type)
        folds.append(ifold)
    reset_keras()
    print('\n*Done with all folds %s'%folds)

def main():
    parser = argparse.ArgumentParser(description = 'multi_task train program',
        usage = 'use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-i','--ifold', type=int, 
        help = 'fold number; also used as seed to split train and val', default =0)

    parser.add_argument('-n', type=int,
        help = 'number of epoch', default = 30)  

    # parser.add_argument('--sig_path', type=str, default = '/media/snasiri/Expansion/CAISR data/Feature_files_revised_v3/',
    #     help = 'signal path, default = /media/snasiri/Expansion/CAISR data/Feature_files_revised_v3/')

    # parser.add_argument('--sig_path', type=str, default = '/media/snasiri/Expansion/CAISR/Feature_files_Robert_v3/',
    #     help = 'signal path, default = /media/snasiri/Expansion/CAISR/Feature_files_Robert_v3/')

    parser.add_argument('--sig_path', type=str, default = '/bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/',
        help = 'signal path, default = /bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/')

    parser.add_argument('--lab_path', type=str, default = 'graphsleepnet',
        help = 'label path, default = graphsleepnet')
    
    parser.add_argument('--train_path', type=str, default = 'without_balancing_gcn_ss_whole_train_70.txt',
        help = 'train path, default = without_balancing_gcn_ss_whole_train_70.txt')

    args = parser.parse_args()
    opts = vars(args)

    run(**opts)

if __name__ == '__main__':
    main()
