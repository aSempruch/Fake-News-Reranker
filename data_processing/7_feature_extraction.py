from util.FeatureExtraction import Features, GET

# Set to None to disable debug mode
DEBUG_MODE = 30
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

get = GET('data/news_dataframe.csv')
features = Features()

galago_output_file = open('data/galago_output.txt', mode='r', encoding='utf-16-le')
output_file = open('data/ranklib_data.txt', mode='w', encoding='utf-8')


""" Single input features """
general_features = [Features.stream_length]

""" Query/Document input features """
query_document_features = [Features.sum_of_term_frequency]


for galago_output_line in galago_output_file:
    query_id, _, doc_id, rank, score, _ = galago_output_line.split(' ')
    # Remove BOM encoding added to start of file by powershell
    query_id = query_id.replace('\ufeff', '')

    # TODO: relevancy
    rel = 0

    # output_line = [rel, f'qid:{qid}']
    feature_values = []

    for general_feature in general_features:
        for getter in [get.title, get.text, get.combined]:
            feature_values.append(general_feature(getter(doc_id)))

    for query_document_feature in query_document_features:
        feature_values.append(query_document_feature(get.query(query_id), get.text(doc_id)))

    output_line = [rel, f'qid:{query_id}', *[f'{idx}:{feature_value}' for (idx, feature_value) in enumerate(feature_values)], '\n']
    output_file.write(' '.join(map(str, output_line)))

    if DEBUG_MODE is not None:
        DEBUG_MODE -= 1
        if DEBUG_MODE == 0:
            break


galago_output_file.close()
output_file.close()
