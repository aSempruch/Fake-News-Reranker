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


# Variables for use in functions below
FLDR_TRAIN = 'ranklib/'
FLDR_OUTPUT = 'models/'
FLDR_KF = FLDR_OUTPUT + 'kft/'
FLDR_SUMMARY = FLDR_OUTPUT + 'summary/'
KFLD_SUFFIX = '.ca'


# TODO:  auto-training, auto-evaluation, print out and store extracted features with names

# This will take the evaluation files and keep only what's important in them.
def ranklib_kfoldEvalProcess(modelFile: str = "TEST_NAME", kfoldNum: int = 1):
    
    # Output file
    outputFileName = FLDR_SUMMARY + modelFile + "_EvalSummary.txt"

    # Regix files!
    regTrain = '(?<=on training data: ).+'
    regValidation = '(?<=on validation data: ).+'
    regTest = '(?<=on test data: ).+'
    
    regTrainOn = '.+(?=on training data: )'
    regValidationOn = '.+(?=on validation data: )'
    regTestOn = '.+(?=on test data: )'

    # We need what they trained on, what they where evaluated on, and values for training and evaluation
    typeTrain = ""
    typeVal = ""
    typeTest = ""

    # The lists for training, validation, and test
    trainingVals = [0] * kfoldNum
    validationVals = [0] * kfoldNum
    testVals = [8] * kfoldNum

    # Now, for each of the kfoldNum, we need the training, validation, and test vals as floats in their arrays.
    for i in range(kfoldNum):
        folderName = FLDR_KF + 'f' + str(i+1) + KFLD_SUFFIX + '_EvalOutput.txt'
        f = open(folderName,"r")
        fileText = f.read()
        f.close()
        trainingVals[i] = float( re.search(regTrain, fileText)[0] )
        validationVals[i] = float( re.search(regValidation, fileText)[0] )
        testVals[i] = float( re.search(regTest, fileText)[0] )

        # In addition, if this is the first time, collect certain values.
        if(i == 0):
            typeTrain = re.search(regTrainOn, fileText)[0]
            typeVal = re.search(regValidationOn, fileText)[0]
            typeTest = re.search(regTestOn, fileText)[0]

    # Now that we have the vals as floats, average them out, and keep them limited to within 4 decimal places.
    avgTrain = 0
    avgVal = 0
    avgTest = 0

    for i in range(kfoldNum):
        avgTrain += trainingVals[i]
        avgVal += validationVals[i]
        avgTest += testVals[i]

    avgTrain = avgTrain/kfoldNum
    avgTrain = round(avgTrain, 4)

    avgVal = avgVal/kfoldNum
    avgVal = round(avgVal, 4)

    avgTest = avgTest/5
    avgTest = round(avgTest, 4)

    # Now we have all of the correct values.  Time to prepare the output file.
    outputFile = open(outputFileName, "w")
    outputFile.write( "Average evaluation vals for model " + modelFile + "\nKfold num: " + str(kfoldNum) + "\n" )
    outputFile.write("Train: " + typeTrain + " " + str(avgTrain) + "\n" )
    outputFile.write("Train: " + typeVal + " " + str(avgVal) + "\n" )
    outputFile.write("Train: " + typeTest + " " + str(avgTest) + "\n" )
    
    return


# Evaluates the features on the model
def ranklib_featureEval(modelFile: str):
    modelLoc = modelFile#'models/' + modelFile + '.txt'
    # NOTE -- the ';' in between MATH3_JAR and 'RANKLIB_JAR' is for windows systems only.  Try using ':' instead of it doesn't work.
    cmd = f'"{JAVA_HOME}/bin/java.exe"   -cp \"{MATH3_JAR}\";\"{RANKLIB_JAR}\" \"ciir.umass.edu.features.FeatureManager\" -feature_stats {modelLoc}'
    output = os.popen(cmd).read()

    # This is ABSOLUTELY ESSENTIAL -- not all features are always used!  This will let us know which ones weren't and which are
    # very important!
    outputFile = modelFile + '_FeatureStatsRaw.txt'
    f = open(outputFile, "w")
    f.write(output)
    f.close()

    return

# Turns all raw stats into a summary file
def ranklib_kfoldFeatureProcess(modelFile: str, kfoldNum: int = 1):
    # Preset variables used later on in the function 
    # 
    # Regex's used to extract the features used, and their frequences
    featureNumRegex = '(?<=\[)\d+(?=\])'
    featureFreqRegex = '(?<=\s\s)\d+(?=\n)'

    # First the number of features
    featureNames = pd.read_csv("ranklib/feature_info.csv")
    numFeatures = len(featureNames)

    # Now, we need to store the values of the features -- max, min, and average -- in a list of the appropiate size.
    featureList = [[0,0,0,0]] * numFeatures

    # Now to set the list so that the feature number is correct for all of them!
    for i in range(numFeatures):
        featureList[i]=[i+1,0,0,0]

    # Here we have a split based on if we have done any kfold training.  For now, we're going to assume we have.
    # If we have, then we want to do a loop based on the number of kfold trainngs we have done.
    for i in range(kfoldNum):
        # First, we want to get the correct files out.
        featureFile = FLDR_KF + 'f' + str(i+1) + KFLD_SUFFIX + '_FeatureStatsRaw.txt'
        f = open(featureFile, "r")
        rawFeatures = f.read()
        f.close()

        # Now we want to extract all of the correct features into a nice list of tuples.
        # Extract them with regexes, and then...
        featuresUsed = re.findall(featureNumRegex, rawFeatures)
        featuresFreq = re.findall(featureFreqRegex, rawFeatures)
        # ...After adjusting them to hold intigers instead of strings...
        featuresUsed = list(map(int, featuresUsed))
        featuresFreq = list(map(int, featuresFreq))
        # ...and turn them into a list of tuples.
        featuresTuples = list(zip(featuresUsed, featuresFreq))

        # Now, we take those features tuples, and for each them, do some work with the features list.
        for tuple in featuresTuples:
            # Getting the actual feature value
            curFeature = tuple[0] - 1

            # Adding the tuple value to the total feature - used count.
            featureList[curFeature][3] += tuple[1]

            # Checking for maximum
            if(tuple[1] > featureList[curFeature][1] ):
                featureList[curFeature][1] = tuple[1]

            # Checking for minimum -- OR if the current minimum is 0
            if((tuple[1] < featureList[curFeature][2]) or ( not featureList[curFeature][2])):
                featureList[curFeature][2] = tuple[1]

    # Now that that's done, if we have done any kfold training, we need to adjust the values
    if(kfoldNum):
        for i in range(numFeatures):
            curVal = featureList[i][3]
            curVal = curVal/kfoldNum
            curVal = round(curVal,2)
            featureList[i][3] = curVal

    # TODO:  NONE K-FOLD TRAINING PROCESSING

    # Now that the stats are properly processed, time to order the array.
    featureList.sort(key = lambda x: x[3], reverse = True)

    # Now that we have that, method to create the string we want given the tuple:
    def featureString(featureVals: list[int,int,int,float]) -> str:

        # The list has the feature ID, the max, min, and then the average frequency.
        # First, however, we need the feature name.
        featureFrame = featureNames.loc[featureVals[0]-1].fillna("")

        namePart_1 = featureFrame.loc['0']
        namePart_2 = featureFrame.loc['1']
        namePart_3 = featureFrame.loc['2']

        # Now to slowly build up the string, using ljust for formatting
        toReturnString = "Feature[" + (str(featureVals[0]) + "]:").rjust(5)
        toReturnString = toReturnString.ljust(17)

        # Should be starting at 15 characters
        toReturnString += namePart_1
        toReturnString = toReturnString.ljust(50)

        # Should be starting at 45 characters
        toReturnString += namePart_2
        toReturnString = toReturnString.ljust(65)

        toReturnString += namePart_3
        toReturnString = toReturnString.ljust(75)
        
        toReturnString += " -- " + (str(featureVals[1])).rjust(5)
        toReturnString += " -- " + (str(featureVals[2])).rjust(5)
        toReturnString += " -- " + (str(featureVals[3])).rjust(8)
        return toReturnString

    finalList = ["ERROR"] * numFeatures

    # First make the full list of features that we currently are using
    for i in range(numFeatures):
        finalList[i] = featureString(featureList[i])
    
    finDocName = FLDR_SUMMARY + modelFile + "_FeatureStatsProcessed.txt"
    featuresFinalDoc = open(finDocName, "w")

    # First off, the starting line of the summary document
    startLine = ("ID".ljust(17) + "NAME").ljust(50)
    startLine = (startLine + "TYPE").ljust(65)
    startLine = (startLine + "PART").ljust(75)
    startLine = startLine + " --   MAX --   MIN --  AVERAGE\n\n"
    featuresFinalDoc.write(startLine)

    transition = False
    for i in range(numFeatures):
        if( not transition and not featureList[i][3]):
            featuresFinalDoc.write("\n----  NEVER USED ----\n\n")
            transition = True
        featuresFinalDoc.write(finalList[i])
        featuresFinalDoc.write("\n")

    return

def ranklib_kfoldFeatureEval(kfoldNum: int = 1):
    for i in range(kfoldNum):
        modelLocation = FLDR_KF + "f" + str(i+1) + KFLD_SUFFIX
        ranklib_featureEval(modelLocation)
        pass
    pass

# Evaluates the models
def ranklib_eval(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    modelLoc = modelFile + '.txt'
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -load {modelLoc} -train {train} -validate {validate} -test {test} -metric2T {params["metric2T"]}'

    output = os.popen(cmd).read()

    # Save the evaluation output in case we want to read it later
    modelEvalLoc = modelFile + '_EvalOutput.txt'
    f = open(modelEvalLoc, "w")
    f.write(output)
    f.close()

    """
    split_output = output.strip().split('\n')

    def extract_score(line: str) -> float:
        return float(line[line.index(':')+2:])

    return {
        'train': extract_score(split_output[-4]),
        'validation': extract_score(split_output[-3]),
        'test': extract_score(split_output[-1])
    }"""


def ranklib_kfoldEval(kfoldNum, train: str, validate: str, test: str, params: dict = DEFAULT_PARAMS):
    for i in range(kfoldNum):
        fileName = FLDR_KF + "f" + str(i+1) + KFLD_SUFFIX
        ranklib_eval(train, validate, test, fileName, params)
        pass
    pass

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


def run_all_test():
    print("Shuffling training data...")
    ranklib_shuffle('ranklib/adjusted_train.txt')
    print("K-fold training...")
    ranklib_kfold_train('ranklib/adjusted_train.txt.shuffled', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile')
    
    print("kfoldEvaluation...")
    ranklib_kfoldEval(5, 'ranklib/adjusted_train.txt.shuffled', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt')

    print("Evaluating kfold model features...")
    ranklib_kfoldFeatureEval(5)

    print("Processing kfold model features...")
    ranklib_kfoldFeatureProcess("test",5)

    pass


if __name__ == '__main__':
    #print(ranklib_train('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile'))
    #print(ranklib_eval('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile'))
    #ranklib_featureEval('testFile')
    #ranklib_shuffle('ranklib/adjusted_train.txt')
    #ranklib_kfold_train('ranklib/adjusted_train.txt', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', 'testFile')
    #ranklib_featureEval('testFile')
    #ranklib_featureProcess('testFile', 5)
    run_all_test()
    #ranklib_kfoldEvalProcess("test",5)

# %%

# %%
