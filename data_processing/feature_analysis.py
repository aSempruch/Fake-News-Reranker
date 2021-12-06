import os

# %%
RANKLIB_JAR = os.getenv('RANKLIB_JAR')
JAVA_HOME = os.getenv('JAVA_HOME')

if None in {RANKLIB_JAR, JAVA_HOME}:
    raise Exception('Missing environment variables')

DEFAULT_PARAMS = {
    'ranker': 6,
    'metric2T': 'NDCG@10',
    'metric2t': 'NDCG@10',
    'norm': 'zscore'
}


def ranklib_evaluate(train: str, validate: str, test: str, params: dict = DEFAULT_PARAMS):
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -train {train} -validate {validate} -test {test} {" ".join([f"-{param} {value}" for param, value in params.items()])}'
    output = os.popen(cmd).read()
    split_output = output.strip().split('\n')

    def extract_score(line: str) -> float:
        return float(line[line.index(':')+2:])

    return {
        'train': extract_score(split_output[-4]),
        'validation': extract_score(split_output[-3]),
        'test': extract_score(split_output[-1])
    }


if __name__ == '__main__':
    print(ranklib_evaluate('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt'))
