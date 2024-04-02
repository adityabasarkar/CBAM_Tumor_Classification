
def evaluate(list):
    
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    tumor_types = ["glioma", "meningioma", "pituitary"]

    for sublist in list:
        if(sublist[0] == sublist[1] and (sublist[0] in tumor_types and sublist[1] in tumor_types)):
            truePositive += 1
        if(sublist[0] == sublist[1] and (sublist[0] not in tumor_types and sublist[1] not in tumor_types)):
            trueNegative += 1
        if(sublist[0] in tumor_types and sublist[1] not in tumor_types):
            falsePositive += 1
        if(sublist[0] not in tumor_types and sublist[1] in tumor_types):
            falseNegative += 1
    
    accuracy = (truePositive + trueNegative) / len(list)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    f1_score = (2 * truePositive) / (2 * truePositive + falsePositive + falseNegative)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

# Testing evaluate()
testList = [["glioma", "glioma"], ["glioma", "no_tumor"], ["no_tumor", "no_tumor"], ["no_tumor", "pituitary"]]
print(evaluate(testList))
