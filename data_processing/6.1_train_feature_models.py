import pandas as pd
import gensim
from tqdm import tqdm

# %% Load documents
df = pd.read_csv('data/news_dataframe.csv', index_col=0)

# %% Build corpus
print("Building corpus")
train_corpus = []

for idx, doc in tqdm(df.iterrows(), total=len(df)):
    for key in ['title', 'text']:
        tokens = gensim.utils.simple_preprocess(doc[key])
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(tokens, [idx]))

# %% Train model
print("Training doc2vec model")
model = gensim.models.doc2vec.Doc2Vec(vector_size=128, min_count=2, epochs=100)
model.build_vocab(train_corpus)

# %% Save model
print("Saving model")
model.save('data/doc2vec_model')
