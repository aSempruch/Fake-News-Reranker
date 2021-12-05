from util.FeatureExtraction import Features, GET
import math

# Number of lines to parse. Set to None to disable debug mode
DEBUG_MODE = 1000
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

get = GET()
features = Features(get=get)

galago_output_file = open('data/galago_output.txt', mode='r', encoding='utf-16-le')
output_file = open('data/ranklib_data.txt', mode='w', encoding='utf-8')


""" Single input features """
general_features = [features.stream_length]

""" Query input features """
query_features = [features.idf]

""" Query/Document input features """
query_document_features = [features.sum_of_term_frequency]

feature_info = []
feature_info_created = False


def process_batch(lines: list):
    """
    Processes batch of galago outputs. query_id should be the same for all lines!
    :param lines: list of (query_id, doc_id, rank, score) tuples
    """
    global feature_info_created

    for line in lines:
        query_id, doc_id, rank, score = line

        rel = math.ceil((len(lines)-int(rank)+1) / (len(lines)/4))-1

        feature_values = []

        for general_feature in general_features:
            for getter in [get.title, get.text, get.combined]:
                feature_values.append(general_feature(getter(doc_id)))

                if not feature_info_created:
                    feature_info.append([general_feature.__name__, getter.__name__])

        for query_feature in query_features:
            for getter in [get.title, get.text, get.combined]:
                feature_values.append(query_feature(query_id))

                if not feature_info_created:
                    feature_info.append([query_feature.__name__, getter.__name__])

        for query_document_feature in query_document_features:
            for getter in [get.title, get.text, get.combined]:
                feature_values.append(query_document_feature(query_id, getter(doc_id)))
                if not feature_info_created:
                    feature_info.append([query_document_feature.__name__, getter.__name__])

        output_line = [rel, f'qid:{query_id}', *[f'{idx+1}:{feature_value}' for (idx, feature_value) in enumerate(feature_values)]]
        output_file.write(' '.join(map(str, output_line)) + '\n')

        feature_info_created = True


batch_lines = []

for galago_output_line in galago_output_file:
    query_id, _, doc_id, rank, score, _ = galago_output_line.split(' ')
    # Remove BOM encoding added to start of file by powershell
    query_id = query_id.replace('\ufeff', '')

    if len(batch_lines) == 0 or batch_lines[0][0] == query_id:
        batch_lines.append((query_id, doc_id, rank, score))
    else:
        process_batch(batch_lines)
        batch_lines.clear()
        batch_lines.append((query_id, doc_id, rank, score))

    if DEBUG_MODE is not None:
        DEBUG_MODE -= 1
        if DEBUG_MODE == 0:
            break


print(*[f'{idx+1}:{val[0]}, {val[1]}\n' for idx, val in enumerate(feature_info)])

galago_output_file.close()
output_file.close()
