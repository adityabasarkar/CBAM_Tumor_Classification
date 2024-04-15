def evaluate(list):

    # Inputs
    # class_label: actual class label
    # tensor: an array representing prediction by model
    
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    tumor_types = ["glioma", "meningioma", "pituitary"]

    tensor_to_class = {
        torch.tensor([1, 0, 0, 0]): 'glioma',
        torch.tensor([0, 1, 0, 0]): 'meningioma',
        torch.tensor([0, 0, 1, 0]): 'pituitary',
        torch.tensor([0, 0, 0, 1]): 'no'
    }

    for i in range(list.__len__()):
        class_label = list[i][0]
        class_prediction = tensor_to_class.get(list[i][2], "Unknown")



        if (class_label == class_prediction and (class_label in tumor_types and class_prediction in tumor_types)):
            truePositive += 1
        if (class_label == class_prediction and (class_label not in tumor_types and class_prediction not in tumor_types)):
            trueNegative += 1
        if (class_label in tumor_types and target.argmax() not in tumor_types):
            falsePositive += 1
        if (class_label not in tumor_types and target.argmax() in tumor_types):
            falseNegative += 1

        '''
              current_eval = test_set.__getitem__(i)
              inputTensor = current_eval[1]
              inputTensor = inputTensor.to(torch.float32)
              inputTensor = inputTensor.to(device)
              inputTensor = inputTensor.unsqueeze(0)
              target = current_eval[2]
              predicted_output = loaded_model(inputTensor)
        '''

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
