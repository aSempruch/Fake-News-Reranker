import os

# %%
RANKLIB_JAR = os.getenv('RANKLIB_JAR')
JAVA_HOME = os.getenv('JAVA_HOME')
MATH3_JAR = os.getenv('MATH3_JAR')

print(MATH3_JAR)
print()

if None in {RANKLIB_JAR, JAVA_HOME, MATH3_JAR}:
    raise Exception('Missing environment variables')

DEFAULT_PARAMS = {
    'ranker': 6,
    'metric2T': 'NDCG@10',
    'metric2t': 'NDCG@10',
    'norm': 'zscore'
}

def ranklib_featureEval(modelFile: str):
    modelLoc = 'models/' + modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe"   -cp \"{MATH3_JAR}\";\"{RANKLIB_JAR}\" \"ciir.umass.edu.features.FeatureManager\" -feature_stats {modelLoc}'
    output = os.popen(cmd).read()

    outputFile = 'models/' + modelFile + '_FeatureStats.txt'
    f = open(outputFile, "w")
    f.write(output)
    f.close()

    return


def ranklib_eval(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    if(modelFile):
        modelLoc = 'models/' + modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -load {modelLoc} -train {train} -validate {validate} -test {test} -metric2T {params["metric2T"]}'
    #cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" ciir.umass.edu.features.FeatureManager"'

    output = os.popen(cmd).read()

    # If we where given a model to train, save the output of training here.
    if(modelFile):
        modelEvalLoc = 'models/' + modelFile + '_EvalOutput.txt'
        f = open(modelEvalLoc, "w")
        f.write(output)
        f.close()


    split_output = output.strip().split('\n')

    def extract_score(line: str) -> float:
        return float(line[line.index(':')+2:])

    return {
        'train': extract_score(split_output[-4]),
        'validation': extract_score(split_output[-3]),
        'test': extract_score(split_output[-1])
    }

def ranklib_train(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    if(modelFile):
        modelLoc = 'models/' + modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -train {train} -validate {validate} -test {test} -save {modelLoc} {" ".join([f"-{param} {value}" for param, value in params.items()])}'
    #os.system(cmd)
    output = os.popen(cmd).read()

    # If we where given a model to train, save the output of training here.
    if(modelFile):
        modelTrainLoc = 'models/' + modelFile + '_TrainingOutput.txt'
        f = open(modelTrainLoc, "w")
        f.write(output)
        f.close()

    split_output = output.strip().split('\n')

    def extract_score(line: str) -> float:
        return float(line[line.index(':')+2:])

    return {
        'train': extract_score(split_output[-6]),
        'validation': extract_score(split_output[-5]),
        'test': extract_score(split_output[-3])
    }


if __name__ == '__main__':
    #print(ranklib_train('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile.txt'))
    #print(ranklib_eval('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile.txt'))
    ranklib_featureEval('testFile')
