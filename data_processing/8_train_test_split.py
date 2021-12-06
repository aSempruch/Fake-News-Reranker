import math
import pandas as pd

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.15
# Test split is remainder

# Files must have .txt extension
files = ['adjusted', 'baseline']
dfs = [pd.read_csv(f'ranklib/{file}.txt', header=None, sep=" ", index_col=False) for file in files]

# %% Read data
# TODO: probably want to take arguments to handle multiple file names

ranklib_data = dfs[0]
ranklib_data['qid'] = ranklib_data[1].str.replace('qid:', '')
ranklib_data['qid'] = ranklib_data['qid'].astype(int)
ranklib_data.set_index('qid', inplace=True)

# %% Compute split indices

TRAIN_INDEX = list(ranklib_data.index).index(ranklib_data.index[math.floor(len(ranklib_data) * TRAIN_SPLIT)])
VALIDATION_INDEX = list(ranklib_data.index).index(ranklib_data.index[math.floor(len(ranklib_data) * VALIDATION_SPLIT + TRAIN_INDEX)]+1)
TEST_INDEX = len(ranklib_data)

print(f'train_samples: {TRAIN_INDEX}\n'
      f'validation_samples: {VALIDATION_INDEX - TRAIN_INDEX}\n'
      f'test_samples: {TEST_INDEX - VALIDATION_INDEX}')

if len({TRAIN_INDEX, VALIDATION_INDEX, TEST_INDEX}) < 3:
    raise Exception("Not enough data to create splits")

# %% Write output
for idx, df in enumerate(dfs):
    file = files[idx]
    df.iloc[0:TRAIN_INDEX].to_csv(f'ranklib/{file}_train.txt', header=False, index=None, sep=" ")
    df.iloc[TRAIN_INDEX:VALIDATION_INDEX].to_csv(f'ranklib/{file}_valid.txt', header=False, index=None, sep=" ")
    df.iloc[VALIDATION_INDEX:TEST_INDEX].to_csv(f'ranklib/{file}_test.txt', header=False, index=None, sep=" ")

print("Files written")
