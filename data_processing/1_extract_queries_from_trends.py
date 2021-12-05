import pandas as pd

# %% Load csv

df = pd.read_csv('datasets/trends.csv')
# Filter out non global or US queries
# TODO: this still contains some spanish queries
df.query('location == "Global" or location == "United States"', inplace=True)
unique_queries = pd.DataFrame(df['query'].unique())
unique_queries.replace(r'[^a-zA-Z\s:]', '', regex=True, inplace=True)
unique_queries.replace(r'^$', float('NaN'), regex=True, inplace=True)
unique_queries.replace(' ', float('NaN'), regex=False, inplace=True)
unique_queries.dropna(inplace=True)
unique_queries = unique_queries.applymap(str.strip)

# %% Write to file

pd.DataFrame(unique_queries).to_csv('data/queries.txt', header=False, sep='\t')

