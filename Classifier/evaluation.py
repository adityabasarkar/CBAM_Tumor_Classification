def evaluate(test_set, loaded_model):

    loaded_model.eval()
    
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    tumor_types = ["glioma", "meningioma", "pituitary"]

    for i in range(len(test_set)):
        class_label = test_set[i][0]
        input_tensor_list = test_set[i][1]
        classification_target = test_set[i][2]

        input_tensor_list = input_tensor_list.to(device)
        input_tensor_list = input_tensor_list.unsqueeze(0)

        
        with torch.no_grad():
            predicted_output = loaded_model(input_tensor_list)

        
        class_prediction_index = predicted_output.argmax().item()

    
        class_prediction = tumor_types[class_prediction_index]

        
        if class_label == class_prediction:
            if class_label in tumor_types:
                truePositive += 1
            else:
                trueNegative += 1
        else:
            if class_label in tumor_types:
                falseNegative += 1
            else:
                falsePositive += 1

    # Calculate evaluation metrics
    accuracy = (truePositive + trueNegative) / len(test_set)
    precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) != 0 else 0
    recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) != 0 else 0
    f1_score = (2 * truePositive) / (2 * truePositive + falsePositive + falseNegative) if (2 * truePositive + falsePositive + falseNegative) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

eval_select = input("Which model do you want to evaluate?")
loaded_model = torch.load(f"./content/models/{eval_select}.pt")
loaded_model.to(torch.float32)
loaded_model.to(device)
test_set = torch.load("./content/test/test_datasets/test_dataset1.pt")
train_set = torch.load("./content/train/train_datasets/train_dataset1.pt")

print(evaluate(test_set, loaded_model)["Accuracy"])
print(evaluate(test_set, loaded_model)["Recall"])
print(evaluate(test_set, loaded_model)["Precision"])
print(evaluate(test_set, loaded_model)["F1 Score"])

