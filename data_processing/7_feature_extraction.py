import pandas as pd

"""
FEATURES
    - stream length - t
    - IDF - q,d
    - sum of term frequency
    - min of term frequency
    - max of term frequency
    - mean of term frequency
    - variance of term frequency
    - sum of stream length normalized term frequency
    - min of stream length normalized term frequency
    - max of stream length normalized term frequency
    - mean of stream length normalized term frequency
    - variance of stream length normalized term frequency
    - sum of tf*idf
    - min of tf*idf
    - max of tf*idf
    - mean of tf*idf
    - variance of tf*idf
"""


class Features:

    def __init__(self, galago_index_path):
        pass

    @staticmethod
    def stream_length(text: str) -> int:
        return len(text)


class GET:

    @staticmethod
    def title(doc_id):
        return NEWS_DF.loc[int(doc_id)]['title']

    @staticmethod
    def text(doc_id):
        return NEWS_DF.loc[int(doc_id)]['text']

    @staticmethod
    def combined(doc_id):
        return GET.title(doc_id) + ' ' + GET.text(doc_id)

    @staticmethod
    def subject(doc_id):
        return NEWS_DF.loc[int(doc_id)]['subject']


NEWS_DF = pd.read_csv('data/news_dataframe.csv', index_col=0)
features = Features('data/index')

galago_output_file = open('data/galago_output.txt', mode='r', encoding='utf-16-le')
output_file = open('data/ranklib_data.txt', mode='w', encoding='utf-8')


""" Single input features """
general_features = [Features.stream_length]

for galago_output_line in galago_output_file:
    qid, _, doc_id, rank, score, _ = galago_output_line.split(' ')

    # TODO: relevancy
    rel = 0

    # output_line = [rel, f'qid:{qid}']
    feature_values = []

    for general_feature in general_features:
        for getter in [GET.title, GET.text, GET.combined]:
            feature_values.append(general_feature(getter(doc_id)))

    output_line = [rel, f'qid:{qid}', *[f'{idx}:{feature_value}' for (idx, feature_value) in enumerate(feature_values)], '\n']
    output_file.write(' '.join(map(str, output_line)))


galago_output_file.close()
output_file.close()
