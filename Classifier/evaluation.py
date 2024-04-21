from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch, cv2, os, random, json, time, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
import math
main_dir = os.path.abspath(__file__)
for i in range(0, 1):
    main_dir = os.path.dirname(main_dir)

class channel_attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(channel_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # We can use Conv2d instead of FCLs to simplify the operation and avoid having to flatten the layers.
        # The operation is essentially the same as in the CBAM paper but applied in a different way.
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, math.ceil(in_channels / reduction_ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(math.ceil(in_channels / reduction_ratio), in_channels, 1, bias=False)
        )

    def forward(self, x):
        # As the network is trained, the channels (feature maps) that should be paid more attention to become more pronounced.
        # Example avg_o and max_o ==> input: [batch_size, 6, 512, 512] -> 2 x [batch_size, 6, 1, 1] -> 2 x [batch_size, 2, 1, 1] -> 2 x [batch_size, 6, 1, 1]
        avg_o = self.fc(self.avg_pool(x))
        max_o = self.fc(self.max_pool(x))
        # Here just add the two channel attentions and put it through a sigmoid function.
        # This will give the attention score for each channel.
        out = torch.sigmoid(avg_o + max_o)
        return out


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):

        # Compress across the channel dimension by getting the average and max of all values across that dimension.
        # input: (batch_size, #channels, height, width) -> output: (batch_size, 1, height, width)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, thr = torch.max(x, dim=1, keepdim=True)

        # Concat the two maps.
        # input: 2 x (batch_size, 1, height, width) -> output: (batch_size, 2, height, width)
        x = torch.cat([avg_map, max_map], dim=1)

        x = self.conv(x)
        out = torch.sigmoid(x)
        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4,sa_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel = channel_attention(in_channels, reduction_ratio)
        self.spatial = spatial_attention(sa_kernel_size)
    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x

class CNN_Attention(nn.Module):
    def __init__(self):
        super(CNN_Attention, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
            nn.BatchNorm2d(6),
            CBAM(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(12),
            CBAM(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            nn.BatchNorm2d(24),
            CBAM(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3),
            nn.BatchNorm2d(48),
            CBAM(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3),
            nn.BatchNorm2d(96),
            CBAM(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3),
            nn.BatchNorm2d(192),
            CBAM(192),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

# CNN without CBAM
class CNN_NoAttention(nn.Module):
    def __init__(self):
        super(CNN_NoAttention, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(18816, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.classify(x)

class BrainTumorClassifier_Attention(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier_Attention, self).__init__()
        self.feature_extraction = CNN_Attention()
        self.classification = DNN()
    def forward(self, x):
        features = self.feature_extraction(x)
        flattened_features = features.view(features.size(0), -1)
        classification = self.classification(flattened_features)
        return classification

class BrainTumorClassifier_NoAttention(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier_NoAttention, self).__init__()
        self.feature_extraction = CNN_NoAttention()
        self.classification = DNN()
    def forward(self, x):
        features = self.feature_extraction(x)
        flattened_features = features.view(features.size(0), -1)
        classification = self.classification(flattened_features)
        return classification

class ImageDataset(Dataset):
    def __init__(self, IO_pairs):
        self.IO_pairs = IO_pairs

    def __len__(self):
        return len(self.IO_pairs)

    def __getitem__(self, index):
        # Get the image
        image_class_name = self.IO_pairs[index][0]
        image_tensor_list = self.IO_pairs[index][1]
        classification_target = self.IO_pairs[index][2]

        return image_class_name, image_tensor_list, classification_target

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

eval_select = input("Which model do you want to evaluate?\n")
attention_select = input("Attention? (Y|N)\n")

if (attention_select == "Y"):
    loaded_model: BrainTumorClassifier_Attention = torch.load(os.path.join(main_dir, "content", "models", f"{eval_select}.pt"))
else:
    loaded_model: BrainTumorClassifier_NoAttention = torch.load(os.path.join(main_dir, "content", "models", f"{eval_select}.pt"))

device = torch.device("cpu")
if torch.cuda.is_available:
    torch.cuda.empty_cache()
    device = torch.device("cuda")

loaded_model.to(torch.float32)
loaded_model.to(device)
loaded_model.eval()

test_set = torch.load(os.path.join(main_dir, "content", "test", "test_datasets", "test_dataset1.pt"))

print(evaluate(test_set, loaded_model))
