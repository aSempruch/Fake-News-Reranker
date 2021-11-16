import pandas as pd

# %% Load csv

df = pd.read_csv('datasets/trends.csv')
# Filter out non global or US queries
# TODO: this still contains some spanish queries
df.query('location == "Global" or location == "United States"', inplace=True)
unique_queries = df['query'].unique()

# %% Write to file

pd.DataFrame(unique_queries).to_csv('data/queries_raw.txt', header=False, sep='\t')
