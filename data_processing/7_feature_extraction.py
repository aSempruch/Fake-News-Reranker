from util.FeatureExtraction import Features, GET
import math

# Number of lines to parse. Set to None to disable debug mode
DEBUG_MODE = 1000
"""
FEATURES
    [X] stream length - t
    [X] IDF - q,d
    [X] sum of term frequency
    [X] min of term frequency
    [X] max of term frequency
    [X] mean of term frequency
    [X] variance of term frequency
    [ ] sum of stream length normalized term frequency
    [ ] min of stream length normalized term frequency
    [ ] max of stream length normalized term frequency
    [ ] mean of stream length normalized term frequency
    [ ] variance of stream length normalized term frequency
    [X] sum of tf*idf
    [X] min of tf*idf
    [X] max of tf*idf
    [X] mean of tf*idf
    [X] variance of tf*idf
"""

get = GET()
features = Features(get=get)

galago_output_file = open('data/galago_output.txt', mode='r', encoding='utf-8')
output_file = open('data/ranklib_data.txt', mode='w', encoding='utf-8')


""" Single input features """
general_features = [features.stream_length]

""" Query input features """
query_features = [features.idf]

""" Query/Document input features """
query_document_features = [features.term_frequency, features.tfidf, features.stream_length_normalized_tf]

feature_info = []
feature_info_created = False


def process_batch(lines: list):
    """
    Processes batch of galago outputs. query_id should be the same for all lines!
    :param lines: list of (query_id, doc_id, rank, score) tuples
    """
    global feature_info, feature_info_created

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
                ret = query_document_feature(query_id, getter(doc_id))
                feature_values += ret.values()
                if not feature_info_created:
                    feature_info += [[query_document_feature.__name__, feature_name,  getter.__name__] for feature_name in ret.keys()]

        output_line = [rel, f'qid:{query_id}', *[f'{idx+1}:{feature_value}' for (idx, feature_value) in enumerate(feature_values)]]
        output_file.write(' '.join(map(str, output_line)) + '\n')

        feature_info_created = True


batch_lines = []

for galago_output_line in galago_output_file:
    query_id, _, doc_id, rank, score, _ = galago_output_line.split(' ')

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


print(*[f'{idx+1}:{", ".join(values)}\n' for idx, values in enumerate(feature_info)])

galago_output_file.close()
output_file.close()
