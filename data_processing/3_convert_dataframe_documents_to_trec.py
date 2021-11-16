import pandas as pd
from xml.dom import minidom

# %% Load documents
df = pd.read_csv('data/news_dataframe.csv', index_col=0)

# %% Convert to trec format

root = minidom.Document()

with open('data/collection.trec', 'w', encoding='utf8') as file:
    for idx, doc_df in df.iterrows():
        doc_element = root.createElement('DOC')

        doc_no_element = root.createElement('DOCNO')
        doc_no_element.appendChild(root.createTextNode(str(idx)))
        doc_element.appendChild(doc_no_element)

        doc_text_element = root.createElement('TEXT')
        doc_text_element.appendChild(root.createTextNode(doc_df['text']))
        doc_element.appendChild(doc_text_element)

        doc_title_element = root.createElement('TITLE')
        doc_title_element.appendChild(root.createTextNode(doc_df['title']))
        doc_element.appendChild(doc_title_element)

        doc_truth_element = root.createElement('TRUTH')
        doc_truth_element.appendChild(root.createTextNode(str(doc_df['truth'])))
        doc_element.appendChild(doc_truth_element)

        file.write(doc_element.toprettyxml())
