import pandas as pd
import re
from util.Galago import Galago


class Features:

    def __init__(self, galago: Galago = None):
        self.galago = Galago() if galago is None else galago

    @staticmethod
    def stream_length(text: str) -> int:
        return len(text)

    @staticmethod
    def sum_of_term_frequency(query: str, document: str):
        term_frequency = 0
        for query_term in query.split(' '):
            term_frequency += sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(query_term.lower()), document.lower()))
            # TODO: get term frequency from galago?
            # term_frequency = self.galago.exec_str('some command')

        return term_frequency


class GET:

    def __init__(self, df_path: str):
        self.NEWS_DF = pd.read_csv('data/news_dataframe.csv', index_col=0)
        self.QUERIES_DF = pd.read_csv('data/queries.txt', index_col=0, sep="\t", header=None)

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
