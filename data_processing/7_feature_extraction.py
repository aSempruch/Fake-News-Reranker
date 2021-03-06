from util.FeatureExtraction import Features, GET
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

# Number of lines to parse. Set to None to disable debug mode
DEBUG_MODE = None

get = GET()
features = Features(get=get)

galago_output_file = open('data/galago_output.txt', mode='r', encoding='utf-8')
output_file_baseline = open('ranklib/baseline.txt', mode='w', encoding='utf-8')
output_file_adjusted = open('ranklib/adjusted.txt', mode='w', encoding='utf-8')

galago_output_line_count = sum(1 for _ in galago_output_file)
galago_output_file.seek(0)

""" Single input features """
general_features = [
    features.stream_length,
    features.spelling
]

""" Text input features (only take `text` portion of document) """
text_input_features = [
    features.polarity_and_subjectivity,
    features.term_stats
]

""" Query input features """
query_features = [features.idf]

""" Query/Document input features """
query_document_features = [
    features.term_frequency,
    features.tfidf,
    features.stream_length_normalized_tf,
    features.doc2vec_query_document
]

""" Title/Text input features """
title_text_features = [features.tf_title_in_text, features.doc2vec_document]

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

        rel_baseline = math.ceil((len(lines)-int(rank)+1) / (len(lines)/4))-1
        rel_adjusted = np.max([rel_baseline - 2*get.truth(doc_id), 0])

        feature_values = []

        for general_feature in general_features:
            for getter in [get.title, get.text, get.combined]:
                ret = general_feature(getter(doc_id))
                feature_values += ret.values()

                if not feature_info_created:
                    feature_info += [[general_feature.__name__, feature_name,  getter.__name__] for feature_name in ret.keys()]

        for text_input_feature in text_input_features:
            ret = text_input_feature(get.text(doc_id))
            feature_values += ret.values()

            if not feature_info_created:
                feature_info += [[text_input_feature.__name__, feature_name,  get.text.__name__] for feature_name in ret.keys()]

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

        for title_text_feature in title_text_features:
            ret = title_text_feature(get.title(doc_id), get.text(doc_id))
            feature_values += ret.values()
            if not feature_info_created:
                feature_info += [[title_text_feature.__name__, feature_name, 'title+text'] for feature_name in ret.keys()]

        output_line = [f'qid:{query_id}', *[f'{idx+1}:{feature_value}' for (idx, feature_value) in enumerate(feature_values)], '#', get.truth(doc_id), doc_id]
        output_file_baseline.write(' '.join(map(str, [rel_baseline, *output_line])) + '\n')
        output_file_adjusted.write(' '.join(map(str, [rel_adjusted, *output_line])) + '\n')

        feature_info_created = True


batch_lines = []

# Open progress bar
with tqdm(total=galago_output_line_count if DEBUG_MODE is None else DEBUG_MODE, unit='lines') as progress_bar:
    for galago_output_line in galago_output_file:
        query_id, _, doc_id, rank, score, _ = galago_output_line.split(' ')

        if len(batch_lines) == 0 or batch_lines[0][0] == query_id:
            batch_lines.append((query_id, doc_id, rank, score))
        else:
            process_batch(batch_lines)
            # progress_bar.update(cur_line_number)
            batch_lines.clear()
            batch_lines.append((query_id, doc_id, rank, score))

        progress_bar.update(1)

        if DEBUG_MODE is not None:
            DEBUG_MODE -= 1
            if DEBUG_MODE == 0:
                break


print(*[f'{idx+1}:{", ".join(values)}\n' for idx, values in enumerate(feature_info)])

pd.DataFrame(feature_info, index=[idx+1 for idx, _ in enumerate(feature_info)]).to_csv('ranklib/feature_info.csv')

galago_output_file.close()
output_file_baseline.close()
output_file_adjusted.close()
