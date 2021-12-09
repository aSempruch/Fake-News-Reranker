import os
import pandas as pd
import re

# %%
RANKLIB_JAR = os.getenv('RANKLIB_JAR')
JAVA_HOME = os.getenv('JAVA_HOME')
# If you did not compile RANKLIB with ant, then you need apache commons-math3.  Current version is 3.6.1
MATH3_JAR = os.getenv('MATH3_JAR')


if None in {RANKLIB_JAR, JAVA_HOME, MATH3_JAR}:
    raise Exception('Missing environment variables')

DEFAULT_PARAMS = {
    'ranker': 6,
    'metric2T': 'NDCG@10',
    'metric2t': 'NDCG@10',
    'norm': 'zscore'
}

# TODO:  auto-training, auto-evaluation, print out and store extracted features with names

# Evaluates the features on the model
def ranklib_featureEval(modelFile: str):
    modelLoc = 'models/' + modelFile + '.txt'
    # NOTE -- the ';' in between MATH3_JAR and 'RANKLIB_JAR' is for windows systems only.  Try using ':' instead of it doesn't work.
    cmd = f'"{JAVA_HOME}/bin/java.exe"   -cp \"{MATH3_JAR}\";\"{RANKLIB_JAR}\" \"ciir.umass.edu.features.FeatureManager\" -feature_stats {modelLoc}'
    output = os.popen(cmd).read()

    # This is ABSOLUTELY ESSENTIAL -- not all features are always used!  This will let us know which ones weren't and which are
    # very important!
    outputFile = 'models/' + modelFile + '_FeatureStatsRaw.txt'
    f = open(outputFile, "w")
    f.write(output)
    f.close()

    return

def ranklib_featureProcess(modelFile: str):

    # First the number of features
    featureNames = pd.read_csv("ranklib/feature_info.csv")
    numFeatures = len(featureNames)

    # Then the feature frequency values
    featureFile = 'models/' + modelFile + '_FeatureStatsRaw.txt'
    f = open(featureFile, "r")
    rawFeatures = f.read()
    f.close()

    # Use regex's to extract the features used, and their frequences
    featureNumRegex = '(?<=\[)\d+(?=\])'
    featureFreqRegex = '(?<=\s\s)\d+(?=\n)'
    featuresUsed = re.findall(featureNumRegex, rawFeatures)
    featuresFreq = re.findall(featureFreqRegex, rawFeatures)
    # ...Afture adjusting them to hold intigers instead of strings...
    featuresUsed = list(map(int, featuresUsed))
    featuresFreq = list(map(int, featuresFreq))
    # ...and then into a list of tuples...
    featuresTuples = list(zip(featuresUsed, featuresFreq))
    # ...and THEN sorted by the SECOND value in the tuple.  We want largest first, so it's reversed, as well.
    featuresTuples.sort(key = lambda x: x[1], reverse = True)

    # Now that we have that, method to create the string we want given the tuple:
    def featureString(featureVals: tuple[int,int]) -> str:
        # The first int is the tuple number, the second, the feature frequency.
        # But first, we need to get the string of the feature information correct
        featureFrame = featureNames.loc[featureVals[0]].fillna("")
        #featureNameString = featureFrame.loc['0'] + " - " + featureFrame.loc['1'] + ' - ' + str(featureFrame.loc['2'])

        # Now to slowly build up the string, using ljust for formatting

        toReturnString = "Feature[" + (str(featureVals[0]) + "]:").rjust(5)
        toReturnString = toReturnString.ljust(17)

        # Should be starting at 15 characters
        toReturnString += featureFrame.loc['0']
        toReturnString = toReturnString.ljust(50)

        # Should be starting at 45 characters
        toReturnString += featureFrame.loc['1']
        toReturnString = toReturnString.ljust(65)

        toReturnString += featureFrame.loc['2']
        toReturnString = toReturnString.ljust(75)
        
        toReturnString += "-- FREQ:" + (str(featureVals[1])).rjust(5)
        return toReturnString

    for t in featuresTuples:
        print(featureString(t))

    #print(featureString(featuresTuples[0]))


    

    return

# Evaluates the models
def ranklib_eval(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    modelLoc = 'models/' + modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -load {modelLoc} -train {train} -validate {validate} -test {test} -metric2T {params["metric2T"]}'

    output = os.popen(cmd).read()

    # Save the evaluation output in case we want to read it later
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

# Trains the models
def ranklib_train(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    if(modelFile):
        modelLoc = 'models/' + modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -train {train} -validate {validate} -test {test} -save {modelLoc} {" ".join([f"-{param} {value}" for param, value in params.items()])}'
    output = os.popen(cmd).read()

    # If we where given a location to save the model, save the normal console output at it's name '_TrainingOutput.txt'.
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

# Basic shuffling
def ranklib_shuffle(train: str):
    cmd = f'"{JAVA_HOME}/bin/java.exe"  -cp \"{RANKLIB_JAR}\" \"ciir.umass.edu.features.FeatureManager\" -input {train} -output ranklib/ -shuffle'
    os.popen(cmd)
    return

# Trains with kfold training
def ranklib_kfold_train(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    if(modelFile):
        modelLoc = 'models/' + modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -train {train} -kcv 5 -kcvmd models/kft/ -kcvmn ca -validate {validate} -test {test} {" ".join([f"-{param} {value}" for param, value in params.items()])}'
    output = os.popen(cmd).read()

    # If we where given a location to save the model, save the normal console output at it's name '_TrainingOutput.txt'.
    if(modelFile):
        modelTrainLoc = 'models/' + modelFile + '_TrainingOutput.txt'
        f = open(modelTrainLoc, "w")
        f.write(output)
        f.close()

    split_output = output.strip().split('\n')

    print(split_output[-11])
    print(split_output[-10])
    print(split_output[-3])

if __name__ == '__main__':
    #print(ranklib_train('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile'))
    #print(ranklib_eval('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile'))
    #ranklib_featureEval('testFile')
    #ranklib_shuffle('ranklib/adjusted_train.txt')
    #ranklib_kfold_train('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile')
    ranklib_featureEval('testFile')
    ranklib_featureProcess('testFile')

# %%
