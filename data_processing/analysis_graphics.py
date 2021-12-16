from matplotlib import pyplot as plt
import re
import numpy as np

FLDR_BASE = "models/analyze/"
FLDR_ADJ = FLDR_BASE + "adjusted/analysis/"
FLDR_UNADJ = FLDR_BASE + "unadjusted/analysis/"

FILE_NAMES = ["compilation_MAP", "compilation_NDCG", "compilation_DCG", "compilation_P", "compilation_RR", "compilation_ERR"]
FLDR_LOCS = [FLDR_ADJ, FLDR_UNADJ]

def plot_analysis():

    """
        What I need to do is:
            Show comparison between baseline and ajusted
                On baseline and adjusted
            Via training and model type
                as test type is not needed
            There is a total of 2*6*4=48 graphs to compare to themselves and each other
            Thus, I shall split them into 4 graphs of 12 -- splitting them up by normalization
                And obtaining all of the correct normalization into the correct slot
            Thus, there shall be a total of 8 graphs to create.
    """
    nameTrain = [
        "_Tr-MAP",
        "_Tr-NDCG",
        "_Tr-DCG",
        "_Tr-P",
        "_Tr-RR",
        "_Tr-ERR"
    ]
    nameTest = "_Te-NDCG"
    nameModel = {
        "no_norm",
        "sum",
        "zscore",
        "linear"
    }
    nameBase = "BASELINE_"

    namesNone = []
    namesSum = []
    namesZscore = []
    namesLinear = []
    for trainType in nameTrain:
        namesNone.append("no_norm" + trainType + nameTest)
        namesSum.append("sum" + trainType + nameTest)
        namesZscore.append("zscore" + trainType + nameTest)
        namesLinear.append("linear" + trainType + nameTest)

    nameTrain = [
        "MAP",
        "NDCG",
        "DCG",
        "P",
        "RR",
        "ERR"
    ]

    allNames = []
    # Here we are collecting all names

    # For every folder and every file name...
    for fileNameNot in FILE_NAMES:
        fileName = fileNameNot + ".txt"
        dataBaseNone = []
        dataBaseSum = []
        dataBaseZscore = []
        dataBaseLinear = []

        dataAdjNone = []
        dataAdjSum = []
        dataAdjZscore = []
        dataAdjLinear = []


        fileLocBase = FLDR_UNADJ+fileName
        fileLocAdj = FLDR_ADJ+fileName
        base = open(fileLocBase, "r")
        adj = open(fileLocAdj, "r")
        # Here I collect the data

        for line in adj:
            if(line.startswith("Detailed break down")):
                break
            for i in range(6):
                # Here, I will be extracting hte lines in the adjusted file.
                result = (0,"",0)
                name = nameTrain[i]
                score = re.findall("\d+.\d+",line)
                if(not score):
                    continue
                score = float( score[0] )
                result = (0,name,score)

                # Here is getting the trained-from-baseline values
                if line.startswith("BASELINE_" + namesNone[i]):
                    print(line)
                    dataAdjNone.append(result)
                    continue
                if line.startswith("BASELINE_" + namesSum[i]):
                    dataAdjSum.append(result)
                    continue
                if line.startswith("BASELINE_" + namesZscore[i]):
                    dataAdjZscore.append(result)
                    continue
                if line.startswith("BASELINE_" + namesLinear[i]):
                    dataAdjLinear.append(result)
                    continue


                result = (1,name,score)
                # Here is getting the trained-from-adjusted values
                if line.startswith(namesNone[i]):
                    dataAdjNone.append(result)
                    continue
                if line.startswith(namesSum[i]):
                    dataAdjSum.append(result)
                    continue
                if line.startswith(namesZscore[i]):
                    dataAdjZscore.append(result)
                    continue
                if line.startswith(namesLinear[i]):
                    dataAdjLinear.append(result)
                    continue
        
        adj.close()
        
        for line in base:
            if(line.startswith("Detailed break down")):
                break
            for i in range(6):
                # Here, I will be extracting hte lines in the baseline file.
                result = (0,"",0)
                name = nameTrain[i]
                score = re.findall("\d+.\d+",line)
                if(not score):
                    continue
                score = float( score[0] )
                result = (0,name,score)

                # Here is getting the trained-from-baseline values
                if line.startswith("BASELINE_" + namesNone[i]):
                    dataBaseNone.append(result)
                    continue
                if line.startswith("BASELINE_" + namesSum[i]):
                    dataBaseSum.append(result)
                    continue
                if line.startswith("BASELINE_" + namesZscore[i]):
                    dataBaseZscore.append(result)
                    continue
                if line.startswith("BASELINE_" + namesLinear[i]):
                    dataBaseLinear.append(result)
                    continue



                result = (1,name,score)
                # Here is getting the trained-from-adjusted values
                if line.startswith(namesNone[i]):
                    dataBaseNone.append(result)
                    continue
                if line.startswith(namesSum[i]):
                    dataBaseSum.append(result)
                    continue
                if line.startswith(namesZscore[i]):
                    dataBaseZscore.append(result)
                    continue
                if line.startswith(namesLinear[i]):
                    dataBaseLinear.append(result)
                    continue

        base.close()

        # Now we have the appropiate data in the arrays, time to make the plots
        datasets = [
            dataBaseNone, dataBaseSum, dataBaseLinear, dataBaseZscore,
            dataAdjNone, dataAdjSum, dataAdjLinear, dataAdjZscore,
            ]
        dataSetNames = [
            "Tested on Baseline, Norm: None", "Tested on Baseline, Norm: Sum", "Tested on Baseline, Norm: Linear", "Tested on Baseline, Norm: Zscore",
            "Tested on Adjusted, Norm: None", "Tested on Adjusted, Norm: Sum", "Tested on Adjusted, Norm: Linear", "Tested on Adjusted, Norm: Zscore",
            ]
        saveTo = [
            "-test_base-norm_no","-test_base-norm_sum","-test_base-norm_lin","-test_base-norm_zscore",
            "-test_adj-norm_no","-test_adj-norm_sum","-test_base-norm_lin","-test_adj-norm_zscore"
            ]

        for i in range(len(datasets)):
            set = datasets[i]
            names = []
            valsA = []
            valsB = []
            for elem in set:
                if elem[0] == 0:
                    names.append(elem[1])
                    valsB.append(elem[2])
                if elem[0] == 1:
                    valsA.append(elem[2])

            valsBA = [valsB[i]-valsA[i] for i in range(len(valsA))]
            valsAB = [valsA[i]-valsB[i] for i in range(len(valsA))]

            x = np.arange(len(names))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, valsA, width, label='Trained on Adjusted')
            rects2 = ax.bar(x + width/2, valsB, width, label='Trained on Baseline')

            ax.set_ylabel('Scores')
            ax.set_title(dataSetNames[i])
            ax.set_xticks(x, names)
            ax.legend()

            #ax.bar_label(rects1, padding=3)
            #ax.bar_label(rects2, padding=3)

            fig.tight_layout()
            loc = "images/original/" + fileNameNot+saveTo[i]+".jpg"
            #print(loc)
            plt.savefig(loc)
            plt.clf()


            # Now for A-sub-B
            newName = "Adjusted-Baseline performance\n"+dataSetNames[i]
            loc = "images/difference/A_sub_B/" + fileNameNot+saveTo[i]+".jpg"

            plt.bar(names,valsAB)
            plt.title(newName)
            plt.savefig(loc)
            plt.clf()

            plt.close()


            # Now for B-sub-A
            newName = "Baseline-Adjusted performance\n"+dataSetNames[i]
            loc = "images/difference/B_sub_A/" + fileNameNot+saveTo[i]+".jpg"

            plt.bar(names,valsBA)
            plt.title(newName)
            plt.savefig(loc)
            plt.clf()

            plt.close()

        """testNames = [i[0] for i in dataAdjNone]
        testVals = [i[1] for i in dataAdjNone]
        print(testNames)
        print(testVals)
        plt.bar(testNames, testVals)
        plt.title(fileName)
        plt.show()"""

    pass




if __name__ == '__main__':
    plot_analysis()
    pass