import pandas as pd
import os
import numpy as np

def name_of_files(list_of_paths):
    filenames = []
    for string in list_of_paths:
        split_list = string.split('/')
        filename = split_list[-1]
        filenames.append(filename)
    return filenames


path = '/bdsp/staging/BDSP-Sleep/CAISR/data/koges_staging/'
df_sid_hard = pd.read_csv(path + 'sids_hard.csv')
hard_examples = df_sid_hard['path_prepared'].values.tolist()
hard_examples = name_of_files(hard_examples)
df_sids_single_only = pd.read_csv(path + 'sids_single_only.csv')
single_only  = df_sid_hard['path_prepared'].values.tolist()
single_only  = name_of_files(single_only )
df_sids_triple = pd.read_csv(path + 'sids_triple.csv')
sids_triple  = df_sid_hard['path_prepared'].values.tolist()
sids_triple  = name_of_files(sids_triple)
import pdb; pdb.set_trace()
# path = '/media/mad3/Projects_New/SLEEP/CAISR1/ground_truth_label_RT_Penn/robert/'
# files = os.listdir (path)

# for file in files:
#     print(file)
#     df = pd.read_csv(path + file, index_col=0)
#     df.columns = ['Kelley (1) sleep_stages', 'Laura (2) sleep_stages', 'Scorer (3) sleep_stages']
#     df = df.T
#     df.to_csv(path+ file)

# import pdb; pdb.set_trace()

os.makedirs(path_to_csv_few_shot, exist_ok=True)
path = '/bdsp/staging/BDSP-Sleep/CAISR/data/extracted_features/sleep_staging/graphsleepnet/features/koges_robert_triple/'
files = os.listdir(path)
cohort = 'koges_robert_triple'
df_few_shot = pd.DataFrame({'files': files, 'dataset': [cohort] * len(files)})

# filename = 'koges_hard_sample_list.csv'
# df_few_shot = pd.read_csv(path_to_csv_few_shot + filename)
# df_1 = pd.read_csv(path_to_csv_few_shot + 'fold_1.csv')

# # merge df2 with df1, excluding rows in df1
# df2_excl_df1 = df_few_shot.merge(df_1, how='outer', indicator=True)\
#                  .query('_merge == "right_only"').drop('_merge', 1)
# import pdb; pdb.set_trace()
threshold_confidence = 0.5
# df_few_shot_interest = df_few_shot[df_few_shot['confidence_score'] < threshold_confidence]
# msk = np.random.rand(len(df_few_shot_interest)) < 0.25
nsubjects = int(len(df_few_shot) * 0.50)
sampled_rows = df_few_shot.sample(n = nsubjects, random_state=42)
df1 = df_few_shot.drop(sampled_rows.index)
df2 = sampled_rows.copy()
import pdb; pdb.set_trace()
df1.to_csv(path_to_csv_few_shot + 'fold_1.csv', index = False)
df2.to_csv(path_to_csv_few_shot + 'fold_2.csv', index = False)

import pdb; pdb.set_trace()
