import pandas as pd
import numpy as np
import re
from util.Galago import Galago
import math
import statistics


class GET:

    def __init__(self, news_df_path='data/news_dataframe.csv', queries_path='data/queries.txt', term_stats_path='data/galago_term_stats.txt'):
        self.NEWS_DF = pd.read_csv(news_df_path, index_col=0)
        self.QUERIES_DF = pd.read_csv(queries_path, index_col=0, sep="\t", header=None)
        self.TERM_STATS_DF = pd.read_csv(term_stats_path, index_col=0, sep="\t", header=None)
        # self.TFIDF_MODEL = TfidfVectorizer().fit(self.NEWS_DF[['text', 'title']])

    def title(self, doc_id):
        return self.NEWS_DF.loc[int(doc_id)]['title']

    def text(self, doc_id):
        return self.NEWS_DF.loc[int(doc_id)]['text']

    def combined(self, doc_id):
        return self.title(doc_id) + ' ' + self.text(doc_id)

    def subject(self, doc_id):
        return self.NEWS_DF.loc[int(doc_id)]['subject']

    def query(self, query_id: str) -> str:
        return self.QUERIES_DF.loc[int(query_id)].values[0]

    def term_frequency(self, term: str) -> int:
        return self.TERM_STATS_DF.loc[term.lower()][2] if term in self.TERM_STATS_DF.index else 0


class Features:

    def __init__(self, galago: Galago = None, get: GET = None):
        self.galago = Galago() if galago is None else galago
        self.get = GET()

    @staticmethod
    def stream_length(text: str) -> int:
        return len(text)

    def idf(self, query_id: str) -> float:
        query = self.get.query(query_id)
        idf_sum = 0
        number_of_docs = len(self.get.NEWS_DF)
        for query_term in query.split(' '):
            # num_docs_containing_term = int(self.galago.exec_str(f'doccount --index=data/index --x+{query_term}').split('\t')[0])
            idf_sum += math.log2((number_of_docs+1) / (self.get.term_frequency(query_term)+1))
        return idf_sum / len(query.split(' '))

    def term_frequency(self, query_id: str, document: str) -> dict:
        query = self.get.query(query_id)
        term_frequency = []
        for query_term in query.split(' '):
            term_frequency.append(sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(query_term.lower()), document.lower())) / len(document.split(' ')))

        return {
            'sum': sum(term_frequency),
            'min': min(term_frequency),
            'max': max(term_frequency),
            'mean': statistics.mean(term_frequency),
            'variance': statistics.variance(term_frequency) if len(term_frequency) > 2 else 0
        }

    def tfidf(self, query_id: str, document: str) -> dict:
        tf = self.term_frequency(query_id, document)
        idf = self.idf(query_id)
        return {k: v*idf for k, v in tf.items()}


    # This is for features 9-12 -- term frequency normalized by stream length, or divided by stream length, and then obtaining the
    # max, min, mean, and variance of those terms.
    def stream_length_normalized_tf(self, query_id: str, document: str) -> float:
        query = self.get.query(query_id)
        
        # Here, I obtain the normalized tf_vals.
        # I get the doc length, and then for every query term, I add that divided by the stream length
        # to an array.
        doc_len = self.stream_length(document)
        norm_tf_vals = []
        for query_term in query.split(' '):
            temp = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(query_term.lower()), document.lower())) / len(document.split(' '))
            norm_tf_vals.append( temp / doc_len )

        # Now that I have the normalized term frequency, it is time to get the values for it.
        norm_tf_sum = sum(norm_tf_vals)
        norm_tf_min = min(norm_tf_vals)
        norm_tf_max = max(norm_tf_vals)
        norm_tf_mean = norm_tf_sum/len(norm_tf_vals)
        norm_tf_var = np.var(norm_tf_vals)

        norm_tf_set ={
            "sum":  norm_tf_sum,
            "min":  norm_tf_min,
            "max":  norm_tf_max,
            "mean": norm_tf_mean,
            "var":  norm_tf_var
        }

        return norm_tf_set
