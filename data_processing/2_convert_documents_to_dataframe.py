import pandas as pd

# %% Read true and false documents

true_df = pd.read_csv('datasets/True.csv')
true_df['truth'] = 1

fake_df = pd.read_csv('datasets/Fake.csv')
fake_df['truth'] = 0

merged_df = true_df.append(fake_df, ignore_index=True)
# %% output to file

merged_df.to_csv('data/news_dataframe.csv')