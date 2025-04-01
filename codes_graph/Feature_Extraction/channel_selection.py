import glob
import numpy as np
import pandas as pd
import os
from shutil import copyfile, rmtree

mad_pfx = '/media/mad3/'
if '/media/cdac/' in os.getcwd():
    dbx_pfx = '/media/cdac/hdd/Dropbox (Partners HealthCare)/'
else:
    dbx_pfx = '/home/thijs/Dropbox (Partners HealthCare)/'
    
def find_rec_in_table(file_path, table_path):
    table = pd.read_csv(table_path)

    folders = table.path_preprocessed
    loc = np.where(file_path == folders)[0]

    if len(loc) == 0: 
        return None
    elif len(loc) > 1:
        print('Warning, multiple matching files are found in MGH table')
    
    info = table.loc[loc[0]]

    return info

def determine_cohort(path):
    if '/mgh_v3/' in path:
        cohort = 'mgh'
    elif '/shhs/' in path:
        cohort = 'shhs'
    elif '/mros/' in path:
        cohort = 'mros'
    elif '/robert/' in path:
        cohort = 'robert'
    elif '/mesa/' in path:
        cohort = 'mesa'
    elif '/penn/' in path:
        cohort = 'penn'
    else:
        print(path)
        import pdb; pdb.set_trace()

    return cohort
    
def get_channel_idxs(channels, sig_tags):
    idxs = []
    for sig in sig_tags:
        idxs.append(np.where(sig==np.array(channels))[0][0])

    assert(len(idxs) == len(sig_tags))
    
    return np.array(idxs)


def check_processed_files(id, unet, graph, mtt, arnn):
    checks, done = [], ''
    tags = ['Unet', 'Graph', 'Multitaper', 'Arnn']
    for i, paths in enumerate([unet, graph, mtt, arnn]):
        check = False
        out_paths = [p + id + '.h5' for p in paths] 
        if np.all([os.path.exists(p) for p in out_paths]):
            check = True
            # make_copies_mad(out_paths)
            done += tags[i] + ' / '
        checks.append(check)

    if len(done) > 0:
        print('* ' + done[:-2] + '-> features already computed!')

    return checks


def make_copies(out_paths):
    for save_path in out_paths:
        if 'Expansion1' in save_path:
            copy_path = save_path.replace('Expansion1', 'Expansion')
        else:
            assert 'Expansion' in save_path
            copy_path = save_path.replace('Expansion', 'Expansion1')

        if not os.path.exists(copy_path):
            copyfile(save_path, copy_path)

def make_copies_mad(out_paths):
    for save_path in out_paths:
        
        copy_path = save_path.split('FeatureFiles')[-1]
        copy_path = mad_pfx + 'Projects/SLEEP/CAISR1/FeatureFiles' + copy_path

        if not os.path.exists(copy_path):
            copyfile(save_path, copy_path)
