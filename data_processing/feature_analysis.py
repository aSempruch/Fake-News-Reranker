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
FLDR_TRAIN_OUTPUT = FLDR_OUTPUT + 'trainingOutput/'
FLD_FEATURE_EVAL = FLDR_OUTPUT + 'feature_evals/'
FLD_FEATURE_PROC = FLDR_OUTPUT + 'feature_processed/'
FLDR_FEATURE_FIN = FLDR_OUTPUT + 'feature_fin/'
FLDR_ANALYZE_NORM = FLDR_OUTPUT + 'analyze/unadjusted/'
FLDR_ANALYZE_ADJ = FLDR_OUTPUT + 'analyze/adjusted/'


# Evaluates the features on the model
def ranklib_featureEval(modelFile: str):
    modelLoc = FLDR_OUTPUT + modelFile#'models/' + modelFile + '.txt'
    # NOTE -- the ';' in between MATH3_JAR and 'RANKLIB_JAR' is for windows systems only.  Try using ':' instead of it doesn't work.
    cmd = f'"{JAVA_HOME}/bin/java.exe"   -cp \"{MATH3_JAR}\";\"{RANKLIB_JAR}\" \"ciir.umass.edu.features.FeatureManager\" -feature_stats {modelLoc}'
    output = os.popen(cmd).read()

    # This is important -- not all features are always used!  This will let us know which ones weren't and which aren't!
    outputFile = FLD_FEATURE_EVAL + modelFile + '_FeatureStatsRaw.txt'
    f = open(outputFile, "w")
    f.write(output)
    f.close()

    return

# Processes the features of a single model
def ranklib_singFeatureProcess(modelFile: str):
    # Preset variables used later on in the function 
    # 
    # Regex's used to extract the features used, and their frequences
    featureNumRegex = '(?<=\[)\d+(?=\])'
    featureFreqRegex = '(?<=\s\s)\d+(?=\n)'

    # First the number of features
    featureNames = pd.read_csv("ranklib/feature_info.csv")
    numFeatures = len(featureNames)

    # store feature id and value
    featureList = [[0,0]] * numFeatures

    # Now to set the list so that the feature number is correct for all of them!
    for i in range(numFeatures):
        featureList[i] = [i+1,0]


    # First, we want to get the correct files out.
    featureFile = FLD_FEATURE_EVAL + modelFile + '_FeatureStatsRaw.txt'
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

        # Set feature value
        featureList[curFeature][1] = tuple[1]

    # Now that the stats are properly processed, time to order the array.
    featureList.sort(key = lambda x: x[1], reverse = True)

    # Now that we have that, method to create the string we want given the tuple:
    def featureString(featureVals: list[int,int]) -> str:

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
        toReturnString = toReturnString.ljust(80)
        
        toReturnString += " --  " + (str(featureVals[1])).rjust(3)
        return toReturnString

    finalList = ["ERROR"] * numFeatures

    # First make the full list of features that we currently are using
    for i in range(numFeatures):
        finalList[i] = featureString(featureList[i])
    
    finDocName = FLD_FEATURE_PROC + modelFile + "_FeatureStatsProcessed.txt"
    featuresFinalDoc = open(finDocName, "w")

    # First off, the starting line of the summary document
    startLine = ("ID".ljust(17) + "NAME").ljust(50)
    startLine = (startLine + "TYPE").ljust(65)
    startLine = (startLine + "PART").ljust(80)
    startLine = startLine + " --  COUNT\n\n"
    featuresFinalDoc.write(startLine)

    transition = False
    for i in range(numFeatures):
        if( not transition and not featureList[i][1]):
            featuresFinalDoc.write("\n----  NEVER USED ----\n\n")
            transition = True
        featuresFinalDoc.write(finalList[i])
        featuresFinalDoc.write("\n")

    return

# Evaluates the models
def ranklib_eval(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    modelLoc = modelFile
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -load {modelLoc} -test {test} -metric2T {params["metric2T"]}'
    print()
    print(cmd)
    print()

    output = os.popen(cmd).read()
    
    # Save the evaluation output in case we want to read it later
    modelEvalLoc = modelFile + '_EvalOutput.txt'
    f = open(modelEvalLoc, "w")
    f.write(output)
    f.close()



# Trains the models
def ranklib_train(train: str, validate: str, test: str, modelFile: str, params: dict = DEFAULT_PARAMS):
    if(modelFile):
        modelLoc = FLDR_OUTPUT + modelFile
    cmd = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -train {train} -validate {validate} -test {test} -save {modelLoc} {" ".join([f"-{param} {value}" for param, value in params.items()])}'
    output = os.popen(cmd).read()

    # If we where given a location to save the model, save the normal console output at it's name '_TrainingOutput.txt'.
    if(modelFile):
        modelTrainLoc = FLDR_TRAIN_OUTPUT + modelFile + '_TrainingOutput.txt'
        f = open(modelTrainLoc, "w")
        f.write(output)
        f.close()

    split_output = output.strip().split('\n')

    def extract_score(line: str) -> float:
        return float(line[line.index(':')+2:])


# Basic shuffling
def ranklib_shuffle(train: str):
    cmd = f'"{JAVA_HOME}/bin/java.exe"  -cp \"{RANKLIB_JAR}\" \"ciir.umass.edu.features.FeatureManager\" -input {train} -output ranklib/ -shuffle'
    os.popen(cmd)
    return


# IMPORTANT NOTE:  Whether this was used on the baseline or adjusted training sets was done via manual editing.
# It was late enough when I was doing it that getting it done was more important than making it modular or
# easily expandible.
def run_fin():

    def getMetric(i: str):
        if(i == "m"):
            return "MAP"
        if(i == "n"):
            return "NDCG@10"
        if(i == "d"):
            return "DCG@10"
        if(i == "p"):
            return "P@10"
        if(i == "r"):
            return "RR@10"
        if(i == "e"):
            return "ERR@10"

    def getName(i: str):
        if(i == "m"):
            return "MAP"
        if(i == "n"):
            return "NDCG"
        if(i == "d"):
            return "DCG"
        if(i == "p"):
            return "P"
        if(i == "r"):
            return "RR"
        if(i == "e"):
            return "ERR"

    metrics = ["m","n","d","p","r","e"]
    normalizeOptions = ["no_norm", "sum", "zscore", "linear"]

    for norm in normalizeOptions:
        for train in metrics:
            for test in metrics:

                cur_params = {
                    'ranker': 6,
                    'metric2T': getMetric(test),
                    'metric2t': getMetric(train),
                }

                if(norm != 'no_norm'):
                    cur_params['norm'] = norm

                modelName = "BASELINE_" + norm + "_Tr-" + getName(train) + "_Te-" + getName(test)

                # We now, for one loop, have the training and test metrics, as well as the model name.
                # Now we need to train the models and get and store the feature evaluations.

                #ranklib_train('ranklib/adjusted_train.txt.shuffled', 'ranklib/adjusted_valid.txt', 'ranklib/adjusted_test.txt', modelName, cur_params)
                ranklib_train('ranklib/baseline_train.txt', 'ranklib/baseline_valid.txt', 'ranklib/baseline_test.txt', modelName, cur_params)

                # Now that the model is trained, it's time to get its feature evaluation
                ranklib_featureEval(modelName)
                ranklib_singFeatureProcess(modelName)

                # Now we need to get the evaluation for adjusted and unadjusted
                for i in metrics:
                    output_1 = FLDR_ANALYZE_ADJ + i + '/' + modelName
                    output_2 = FLDR_ANALYZE_NORM + i + '/' +  modelName
                    modelLoc = FLDR_OUTPUT + modelName

                    cmdAdj = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -load {modelLoc} -test ranklib/adjusted_test.txt -metric2T {getMetric(i)} -idv {output_1}'
                    cmdNorm = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -load {modelLoc} -test ranklib/baseline_test.txt -metric2T {getMetric(i)} -idv {output_2}'
                    os.popen(cmdAdj)
                    os.popen(cmdNorm)
                    pass

                print( "Completed model: " + modelName )
                
    # We should now have all of the models trained, saved, evaluated, and with features extracted in both raw
    # and processed formates.  Now, to make the baselines for the analysis.

    for train in metrics:
        outputAdj = FLDR_ANALYZE_ADJ + train + '/baseline'
        outputNorm = FLDR_ANALYZE_NORM + train + '/baseline'
        cmdA = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -test ranklib/adjusted_test.txt -metric2T {getMetric(train)} -idv {outputAdj}'
        cmdB = f'"{JAVA_HOME}/bin/java.exe" -jar \"{RANKLIB_JAR}\" -test ranklib/baseline_test.txt -metric2T {getMetric(train)} -idv {outputNorm}'
        os.popen(cmdA)
        os.popen(cmdB)

    print("Baseline Files Completed")

    for i in metrics:
        locA = FLDR_ANALYZE_ADJ + i + '/'
        locB = FLDR_ANALYZE_NORM + i + '/'
        nameA = FLDR_ANALYZE_ADJ + '/analysis/compilation_' + getName(i) + '.txt'
        nameB = FLDR_ANALYZE_NORM + '/analysis/compilation_' + getName(i) + '.txt'
        cmdA = f'"{JAVA_HOME}/bin/java.exe" -cp \"{RANKLIB_JAR}\" ciir.umass.edu.eval.Analyzer -all {locA} -base baseline > {nameA}'
        cmdB = f'"{JAVA_HOME}/bin/java.exe" -cp \"{RANKLIB_JAR}\" ciir.umass.edu.eval.Analyzer -all {locB} -base baseline > {nameB}'
        os.popen(cmdA)
        os.popen(cmdB)
        pass
    
           
    # Now that all that is done with, it's time to process the variety of features in a more efficient manner.
    # For this, we shall want features by:
    #   Normalization value
    #   Training value
    #   Testing value
    # Thus we shall start by making the folder lists that we will be wanting
    # Then we shall send those lists into a function to make the output
    # This should make a total of 16 files, all told.

    featureFoldersTrain = []
    featureFoldersTest = []
    featureFoldersNorm = []

    for i in metrics:
        trainList = []
        testList = []
        for j in metrics:
            for norm in normalizeOptions:
                # As for the train list the train folders should be the same, and the reverse is true
                # for the norm list, this is how we're doing it.
                trainFld = FLD_FEATURE_PROC + "BASELINE_" + norm + "_Tr-" + getName(i) + "_Te-" + getName(j) + "_FeatureStatsProcessed.txt"
                testFld = FLD_FEATURE_PROC + "BASELINE_" + norm + "_Tr-" + getName(j) + "_Te-" + getName(i) + "_FeatureStatsProcessed.txt"
                trainList.append(trainFld)
                testList.append(testFld)
            pass
        featureFoldersTrain.append(trainList)
        featureFoldersTest.append(testList)
    
    for norm in normalizeOptions:
        normList = []
        for i in metrics:
            for j in metrics:
                fldr = FLD_FEATURE_PROC + "BASELINE_" + norm + "_Tr-" + getName(i) + "_Te-" + getName(j) + "_FeatureStatsProcessed.txt"
                normList.append(fldr)
        featureFoldersNorm.append(normList)

    # Now that we have the various lists, we need the processing functions.  Ho boy.

    def procProccessedFeatures(fileList, fileName):
        # Preset variables used later on in the function 

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
        for i in range(len(fileList)):
            # First, we want to get the correct files out.
            featureFile = fileList[i]
            #print(featureFile)
            f = open(featureFile, "r")
            rawFeatures = f.read()
            f.close()

            # Now I gather all the numbers!
            numbers = re.findall('(?<=\s)\d+', rawFeatures)

            featuresTuples = [(0,0)] * numFeatures
            for i in range(numFeatures):
                curFeature = int(numbers[i*2])
                curFreq = int(numbers[i*2+1])
                featuresTuples[i] = (curFeature,curFreq)
                if(curFeature > numFeatures):
                    print("ERROR")
                if(curFeature <= 0):
                    print("ERROR")
            #print(featuresTuples)
            

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
        for i in range(numFeatures):
            curVal = featureList[i][3]
            curVal = curVal/len(fileList)
            curVal = round(curVal,2)
            featureList[i][3] = curVal

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
            toReturnString = toReturnString.ljust(80)
            
            toReturnString += " -- " + (str(featureVals[1])).rjust(5)
            toReturnString += " -- " + (str(featureVals[2])).rjust(5)
            toReturnString += " -- " + (str(featureVals[3])).rjust(8)
            return toReturnString

        finalList = ["ERROR"] * numFeatures

        # First make the full list of features that we currently are using
        for i in range(numFeatures):
            finalList[i] = featureString(featureList[i])
        
        finDocName = FLDR_FEATURE_FIN + fileName + ".txt"
        featuresFinalDoc = open(finDocName, "w")

        # First off, the starting line of the summary document
        startLine = ("ID".ljust(17) + "NAME").ljust(50)
        startLine = (startLine + "TYPE").ljust(65)
        startLine = (startLine + "PART").ljust(80)
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

    # And now, finally, we call upon this for each list of files!
    # Done manually because EXHAUSTED
    procProccessedFeatures(featureFoldersTest[0], "BASELINE_test_" + getName("m"))
    procProccessedFeatures(featureFoldersTest[1], "BASELINE_test_" + getName("n"))
    procProccessedFeatures(featureFoldersTest[2], "BASELINE_test_" + getName("d"))
    procProccessedFeatures(featureFoldersTest[3], "BASELINE_test_" + getName("p"))
    procProccessedFeatures(featureFoldersTest[4], "BASELINE_test_" + getName("r"))
    procProccessedFeatures(featureFoldersTest[5], "BASELINE_test_" + getName("e"))


    procProccessedFeatures(featureFoldersTrain[0], "BASELINE_train_" + getName("m"))
    procProccessedFeatures(featureFoldersTrain[1], "BASELINE_train_" + getName("n"))
    procProccessedFeatures(featureFoldersTrain[2], "BASELINE_train_" + getName("d"))
    procProccessedFeatures(featureFoldersTrain[3], "BASELINE_train_" + getName("p"))
    procProccessedFeatures(featureFoldersTrain[4], "BASELINE_train_" + getName("r"))
    procProccessedFeatures(featureFoldersTrain[5], "BASELINE_train_" + getName("e"))

    procProccessedFeatures(featureFoldersNorm[0], "BASELINE_norm_none")
    procProccessedFeatures(featureFoldersNorm[1], "BASELINE_norm_sum")
    procProccessedFeatures(featureFoldersNorm[2], "BASELINE_norm_zscore")
    procProccessedFeatures(featureFoldersNorm[3], "BASELINE_norm_linear")


if __name__ == '__main__':

    # NOTE:  run_fin was done roughly and quickly; it ended up using copied code from earlier functions.
    # This is due to the time of night it was coded; it worked, and got us the data we needed, and thus
    # hasn't been improved since.
    run_fin()