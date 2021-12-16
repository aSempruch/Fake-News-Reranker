import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# %% Import Feature weighting data
FEATURE_COUNT = 65
all_feature_data = {}

TRAIN_PREFIXES = {
    'baseline': 'BASELINE_',
    'adjusted': ''
}

for model_type, model_type_fname in TRAIN_PREFIXES.items():
    all_feature_data[model_type] = {}
    for file_suffix in ['DCG', 'ERR', 'MAP', 'NDCG', 'P', 'RR']:
        with open(f'models/feature_fin/{model_type_fname}test_{file_suffix}.txt', mode='r', encoding='utf-8') as file:
            file.readline()
            file.readline()

            feature_data = {}

            for line in file:
                split = line.split()
                if len(split) < 3:
                    break
                feature_id = int(split[1].replace(']', '').replace(':', ''))
                max = int(split[-5])
                min = int(split[-3])
                avg = float(split[-1])

                feature_data[feature_id] = {
                    'max': max,
                    'min': min,
                    'avg': avg
                }

            all_feature_data[model_type][file_suffix] = feature_data

# %% PLot feature weightings
model = 'DCG'
EMPTY_FEAT = {'max': 0, 'min': 0, 'avg': 0}
for model_type in TRAIN_PREFIXES:
    feature_data = all_feature_data[model_type][model]
    x = list(range(1, 66))
    y = [feature_data[feature_idx] if feature_idx in feature_data else EMPTY_FEAT for feature_idx in x]
    plt.clf()
    plt.bar(x, [feat['avg'] for feat in y], color='blue')
    plt.xticks(x, fontsize=4.3)
    plt.title(f'Feature weightings of model trained on {model_type} data')
    plt.savefig(f'figures/feature_weights_{model_type}.png')


# %% Get performance data

PLOTTED_TEST_METRICS = {
    'NCDG@10',
    'MAP',
    'precision',
    'RR'
}

PLOTTED_TRAIN_METRICS = {
    *PLOTTED_TEST_METRICS
}

DIRECTORIES = {
    'd': 'DCG@10',
    'e': 'ERR@10',
    'm': 'MAP',
    'n': 'NDCG@10',
    'p': 'precision',
    'r': 'RR'
}

TRAIN_METRICS = {
    'DCG': DIRECTORIES['d'],
    'ERR': DIRECTORIES['e'],
    'MAP': DIRECTORIES['m'],
    'NDCG': DIRECTORIES['n'],
    'P': DIRECTORIES['p'],
    'RR': DIRECTORIES['r']
}

NORMALIZATIONS = [
    'linear',
    'no_norm',
    'sum',
    'zscore'
]

all_performance_data = {}

for test_dir_name in TRAIN_PREFIXES.keys():
    all_performance_data[test_dir_name] = {}
    for score_dir_name, score in DIRECTORIES.items():
        all_performance_data[test_dir_name][score] = {}
        for metric_fname, train_metric in TRAIN_METRICS.items():
            all_performance_data[test_dir_name][score][train_metric] = {}
            for train_type, train_fname in TRAIN_PREFIXES.items():
                all_performance_data[test_dir_name][score][train_metric][train_type] = {}
                for normalization in NORMALIZATIONS:
                    with open(f'models/analyze/{test_dir_name}/{score_dir_name}/{train_fname}{normalization}_Tr-{metric_fname}_Te-DCG', mode='r', encoding='utf-8') as file:
                        for line in file:
                            split = line.split()
                            if split[1] == 'all':
                                value = float(split[-1])
                                all_performance_data[test_dir_name][score][train_metric][train_type][normalization] = value
                                break
                        if normalization not in all_performance_data[test_dir_name][score][train_metric][train_type]:
                            raise Exception(f'Could not find "all" value in file {file.name}')

# %% Plot performance data

clean_name = lambda n: n.replace("@", "")

for test_type_name, test_type in all_performance_data.items():
    for test_metric_name, test_metric in test_type.items():
        if test_metric_name not in PLOTTED_TEST_METRICS: continue
        for train_metric_name, train_metric in test_metric.items():
            if train_metric_name not in PLOTTED_TRAIN_METRICS: continue
            plt.clf()
            # for train_metric_name, train_metric in train_metric.items():
            X = NORMALIZATIONS
            Y_baseline = train_metric['baseline'].values()
            Y_adjusted = train_metric['adjusted'].values()

            X_axis = np.arange(len(NORMALIZATIONS))

            plt.bar(X_axis - 0.2, Y_baseline, 0.4, label='baseline')
            plt.bar(X_axis + 0.2, Y_adjusted, 0.4, label='adjusted')

            ax = plt.gca()
            concat_data = [*Y_baseline, *Y_adjusted]
            ax.set_ylim([np.min(concat_data)-0.05, np.min([np.max(concat_data)+0.05, 1])])

            plt.xticks(X_axis, X)
            plt.xlabel('Normalization method')
            plt.ylabel(test_metric_name)
            plt.title(f'{test_metric_name} of model\ntested on {test_type_name} data\ntrained using {train_metric_name} metric')
            plt.legend(loc='lower right')
            plt.savefig(f'figures/{test_type_name}_te_{clean_name(test_metric_name)}_tr-{clean_name(train_metric_name)}.jpeg')
    #         break
    #     break
    # break
