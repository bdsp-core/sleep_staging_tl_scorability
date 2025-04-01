import numpy as np
import sys, os, random
from collections import Counter
from get_mad_dbx import *
# mad_pfx, dbx_pfx, feature_folder, mem_lim = get_mad_dbx()
dbx_pfx = '/media/exx/Expansion1/codes/koges_caisr/'
sys.path.insert(1, dbx_pfx + 'main_CAISR_functions/')
from load_features_and_labels_updated import *
# from load_features_and_labels import *  # if two eeg channels
# from load_features_and_labels_only_EEG import *  # if 6 channels


sys.path.insert(1, os.getcwd())
from utils import AddContext
context = 7

def generate_test_data(test_line, test_cohort_line, path1, path2):
    # First shuffle the lines
    ids = list(range(len(test_line)))
    random.shuffle(ids)
    ids = np.array(ids)

    test_line = np.array(test_line)[ids].tolist()
    test_cohort_line = np.array(test_cohort_line)[ids].tolist()

    ##### augmentation parameters ######
    image_batch = []
    label_batch = []
   
    for i in range(0, len(test_line)):
        sample = test_line[i]
        the_id=sample.split('/')[-1]
        
        # the_id = the_id[:-3]  ## mgh and robert (if using load_features_and_labels_updated)
        if dataset in ['mros', 'mesa', 'shhs', 'robert_v6', 'qa_robert_v4', 'stanford_triple_scored']:
            the_id = the_id[:-3]
        dataset = test_cohort_line [i]
        print('sample: %s, cohort: %s, i=%s'%(sample, dataset, i))
        try:
            image, label, _ = load_features_and_labels(path1, path2, dataset, the_id)
            # import pdb; pdb.set_trace()
            # Get the mode value in each row of the array
            if dataset in ['robert_v6', 'koges_robert_triple', 'qa_robert_v4', 'stanford_triple_scored']:
                allowed_values = [1,2,3,4,5]
                mask = ~np.isin(label, allowed_values)  # get the indices of values not in allowed_values
                label = np.delete(label, np.where(mask), axis = 0)  # remove elements at those indices from arr
                majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=label)
                # Create a new array with the majority values
                label = majority.reshape(-1, 1)
                label = np.squeeze(label)
                image = np.delete(image, np.where(mask), axis = 0)
            if dataset in ['koges_robert_single', 'mgh', 'mros', 'mesa', 'shhs']:
                allowed_values = [1,2,3,4,5]
                mask = ~np.isin(label, allowed_values)
                label = np.delete(label, np.where(mask))

                image = np.delete(image, np.where(mask), axis = 0)

            elif dataset == 'penn_v2':
                allowed_values = [1,2,3,4,5]
                mask = ~np.isin(label, allowed_values) 
                label = np.delete(label, np.where(mask), axis = 0)
                label = label [:,6]
                image = np.delete(image, np.where(mask), axis = 0)
                # majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=label)
                # # Create a new array with the majority values
                # label = majority.reshape(-1, 1)
                label = np.squeeze(label)
            
            # zero_indices = np.isclose(image, 0).all(axis=(1, 2))
            # image = image[~zero_indices]
            # label = label[~zero_indices]
            # nan_indices = np.isnan(image).any(axis=(1, 2))
            # image = image[~nan_indices]
            # label = label[~nan_indices]
            # allowed_values = [1,2,3,4,5]
            # mask = ~np.isin(label, allowed_values)  # get the indices of values not in allowed_values
            # label = np.delete(label, np.where(mask))  # remove elements at those indices from arr
            # image = np.delete(image, np.where(mask), axis = 0)
            ## for robert dataset if we saved all experts' label
            # label = np.squeeze(label[:,0])
            
            # if dataset in ['mgh', 'robert', 'qa_robert_v2']:
            #     channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
                # common_channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
                # idx = [i for i, e in enumerate(channels) if e in set(common_channels)] 
                # image = image [:, idx, :]
            sleep_stage_max = 6
            label = np.eye(sleep_stage_max)[label.tolist()] 
            label = label [:,1:6]
        except:
            continue
        ### remove the undefined class
        image_batch.append(image)
        label_batch.append(label)

    image_batch = np.concatenate(image_batch, axis=0)
    label_batch = np.concatenate(label_batch, axis=0)
    label_batch[np.isnan(label_batch)] = 0 

    image_batch = AddContext(image_batch, context)
    image_batch = np.squeeze(np.array(image_batch))

    label_batch = AddContext(label_batch, context, label = True)
    label_batch = np.squeeze(np.array(label_batch))

    return image_batch, label_batch


def generate_data(train_line, train_cohort_line, batch_size, path1, path2):
    # First shuffle the lines
    ids = list(range(len(train_line)))
    random.shuffle(ids)
    ids = np.array(ids)
    # import pdb; pdb.set_trace()
    train_line = np.array(train_line)[ids].tolist()
    train_cohort_line = np.array(train_cohort_line)[ids].tolist()

    ##### augmentation parameters ######
    i = 0
    # import pdb; pdb.set_trace()
    while True:
        image_batch = []
        label_batch = []
        while len(image_batch) < batch_size:
            if i == len(train_line):
                i = 0
                
            sample = train_line[i]
            the_id = sample.split('/')[-1]
            # import pdb; pdb.set_trace()
            # the_id = the_id[:-3] ## mgh and robert (if using load_features_and_labels_updated)
            dataset = train_cohort_line[i]
            if dataset in ['mros', 'mesa', 'shhs', 'robert_v6', 'qa_robert_v4', 'stanford_triple_scored']:
                the_id = the_id[:-3]
            # print('sample: %s, cohort: %s, i=%s'%(sample, dataset, i))
            i += 1
            # import pdb; pdb.set_trace()
            # print(path1)
            # print (the_id)
            try:
                image, label, _ = load_features_and_labels(path1, path2, dataset, the_id)
                # import pdb; pdb.set_trace()
                if dataset in [ 'robert_v6', 'koges_robert_triple', 'qa_robert_v4', 'stanford_triple_scored']:
                    allowed_values = [1,2,3,4,5]
                    mask = ~np.isin(label, allowed_values)  # get the indices of values not in allowed_values
                    label = np.delete(label, np.where(mask), axis = 0)  # remove elements at those indices from arr
                    majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=label)
                    # Create a new array with the majority values
                    label = majority.reshape(-1, 1)
                    label = np.squeeze(label)
                    image = np.delete(image, np.where(mask), axis = 0)
                    
                if dataset in ['koges_robert_single', 'mgh', 'mros', 'mesa', 'shhs']:
                    allowed_values = [1,2,3,4,5]
                    mask = ~np.isin(label, allowed_values)
                    label = np.delete(label, np.where(mask))

                    image = np.delete(image, np.where(mask), axis = 0)
                    
                    # import pdb; pdb.set_trace()

                elif dataset == 'penn_v2':
                    allowed_values = [1,2,3,4,5]
                    mask = ~np.isin(label, allowed_values) 
                    label = np.delete(label, np.where(mask), axis = 0)
                    label = label [:,6]
                    image = np.delete(image, np.where(mask), axis = 0)
                    # import pdb; pdb.set_trace()
                    # majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=label)
                    # # Create a new array with the majority values
                    # label = majority.reshape(-1, 1)
                    label = np.squeeze(label)
                # zero_indices = np.isclose(image, 0).all(axis=(1, 2))
                # image = image[~zero_indices]
                # label = label[~zero_indices]
                # nan_indices = np.isnan(image).any(axis=(1, 2))
                # image = image[~nan_indices]
                # label = label[~nan_indices]
                # allowed_values = [1,2,3,4,5]
                # mask = ~np.isin(label, allowed_values)  # get the indices of values not in allowed_values
                # label = np.delete(label, np.where(mask))  # remove elements at those indices from arr
                # image = np.delete(image, np.where(mask), axis = 0)
                ## for robert dataset if we saved all experts' label
                # label = np.squeeze(label[:,0])
                # if dataset in ['mgh', 'robert']:
                #     channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
                    # common_channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg'] 
                    # idx = [i for i, e in enumerate(channels) if e in set(common_channels)] 
                    # image = image [:, idx, :]
                sleep_stage_max = 6
                label = np.eye(sleep_stage_max)[label.tolist()] 
                label = label [:,1:6]
                # print (image.shape)
                # print (label.shape)
                # import pdb; pdb.set_trace()
            except:
                continue
            # print (label.shape)
            # print(image.shape)
            ### remove the undefined class
            image_batch.append(image)
            label_batch.append(label)

        image_batch = np.concatenate(image_batch, axis=0)
        label_batch = np.concatenate(label_batch, axis=0)
        # label_batch[np.isnan(label_batch)] = 0 

        # balancing the #samples per class
        # label_categorical = np.argmax(label_batch, axis = 1) # Categorical label
        # unique, counts = np.unique(label_categorical, return_counts=True)
        # # print ("source data = ", dict(zip(unique, counts)))
        # # for using fit_transform, the input should be 2D array
        # image_batch_reshape = image_batch.reshape(len(image_batch),-1)  
        # # Adaptive Synthetic Sampling (ADASYN)
        # # generating synthetic samples inversely proportional to the density of the examples in the minority class
        # # oversample = ADASYN()
        # # oversample_ = RandomOverSampler()

        # oversample_ = SMOTE()
        # X, y = oversample_.fit_resample(image_batch_reshape, label_categorical)
        # unique, counts = np.unique(y, return_counts=True)
        # # print ("oversampled = ", dict(zip(unique, counts)))
        # onehot_encoder = OneHotEncoder(sparse=False)
        # integer_encoded = y.reshape(len(y), 1)
        # label_batch = onehot_encoder.fit_transform(integer_encoded)
        # image_batch = X.reshape ((X.shape[0], image_batch.shape[1], image_batch.shape[2]))
        ###########################

        image_batch = AddContext (image_batch , context)
        image_batch = np.squeeze(np.array(image_batch))

        label_batch = AddContext (label_batch , context, label = True)
        label_batch = np.squeeze(np.array(label_batch))

        # print (label_batch.shape)
        yield image_batch, label_batch