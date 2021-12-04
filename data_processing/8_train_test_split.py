import math
import pandas as pd

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.5

# %% Read data
# TODO: probably want to take arguments to handle multiple file names

ranklib_data = pd.read_csv('data/ranklib_data.txt', header=None, sep=" ", index_col=False)
ranklib_data['qid'] = ranklib_data[1].str.replace('qid:', '')
ranklib_data['qid'] = ranklib_data['qid'].astype(int)
ranklib_data.set_index('qid', inplace=True)

# %% Compute split indices

TRAIN_INDEX = list(ranklib_data.index).index(ranklib_data.index[math.floor(len(ranklib_data) * TRAIN_SPLIT)])
VALIDATION_INDEX = list(ranklib_data.index).index(ranklib_data.index[math.floor(len(ranklib_data) * VALIDATION_SPLIT + TRAIN_INDEX)]+1)
TEST_INDEX = len(ranklib_data)

if len({TRAIN_INDEX, VALIDATION_INDEX, TEST_INDEX}) < 3:
    raise Exception("Not enough data to create splits")

# %% Write output
ranklib_data.iloc[0:TRAIN_INDEX].to_csv('ranklib/train.txt', header=False, index=None, sep=" ")
ranklib_data.iloc[TRAIN_INDEX:VALIDATION_INDEX].to_csv('ranklib/valid.txt', header=False, index=None, sep=" ")
ranklib_data.iloc[VALIDATION_INDEX:TEST_INDEX].to_csv('ranklib/test.txt', header=False, index=None, sep=" ")
