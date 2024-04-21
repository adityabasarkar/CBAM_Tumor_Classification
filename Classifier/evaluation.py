from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def evaluate(test_set, loaded_model):

    loaded_model.eval()

    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    tumor_types = ["glioma", "meningioma", "pituitary", "no"]
    roc_y = []
    roc_y_pred = []

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

        
        roc_y.append(classification_target.cpu().numpy())
        roc_y_pred.append(predicted_output.cpu().numpy())


        # Used for accuracy, recall, precision, and f1 score
        if class_label == class_prediction:
            if class_label in ["glioma", "meningioma", "pituitary"]:
                truePositive += 1
            else:
                trueNegative += 1
        else:
            if class_label in ["glioma", "meningioma", "pituitary"]:
                falseNegative += 1
            else:
                falsePositive += 1

    # Calculate evaluation metrics
    accuracy = (truePositive + trueNegative) / len(test_set)
    precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) != 0 else 0
    recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) != 0 else 0
    f1_score = (2 * truePositive) / (2 * truePositive + falsePositive + falseNegative) if (2 * truePositive + falsePositive + falseNegative) != 0 else 0

    

    roc_y = np.array(roc_y)
    roc_y_pred = np.array(roc_y_pred)

    roc_y_pred_flat = np.array([pred[0] for pred in roc_y_pred])

    true_labels_binary = label_binarize(roc_y, classes=[0, 1, 2, 3])
    true_labels = np.argmax(true_labels_binary, axis=1)
    n_classes = true_labels_binary.shape[1] 

    #predicted_scores_binary = label_binarize(roc_y_pred, classes=[0, 1, 2, 3])
    predicted_labels = np.argmax(roc_y_pred_flat, axis=1)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=tumor_types, yticklabels=tumor_types)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

