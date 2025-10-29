import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from CGNet_Model_MyDecoder__SA import *
from ChangeFormerV4_MyEncoder import *

from My_Trainer_1 import test_loader
import time
from sklearn.metrics import roc_curve, auc

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random
""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class change_detection_dataset(Dataset):
    def __init__(self,root_path) -> None:
        super().__init__()
        self.pre_change_path=os.path.join(root_path,"A")
        self.post_change_path=os.path.join(root_path,"B")
        self.change_label_path=os.path.join(root_path,"label")
        self.fname_list=os.listdir(self.pre_change_path)
    def __getitem__(self, index):
        fname=self.fname_list[index]
        pre_img=Image.open(os.path.join(self.pre_change_path,fname)).convert("RGB")
        post_img=Image.open(os.path.join(self.post_change_path,fname)).convert("RGB")
        change_label=Image.open(os.path.join(self.change_label_path,fname)).convert("1")
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        pre_tensor=transform(pre_img)
        post_tensor=transform(post_img)
        label_tensor=transform(change_label)
        return {'pre':pre_tensor,'post':post_tensor,'label':label_tensor,'fname':fname}
    
    def __len__(self):
        return len(self.fname_list)
    




# Generate predictions and labels
def get_predictions_and_labels(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
            pre_tensor = pre_tensor.to(device)
            post_tensor = post_tensor.to(device)
            label_tensor = label_tensor.to(device)
            probs = model(pre_tensor, post_tensor)
            # print (probs.shape, label_tensor.shape)
            # print (probs.min(), probs.max())
            
            # prediction = torch.where(probs>0.5,1.0,0.0)
            # print (prediction.min(), prediction.max())

            all_labels.append(label_tensor.cpu().numpy())
            all_predictions.append(probs.cpu().numpy())
            # print (len(all_labels))
    return np.concatenate(all_predictions), np.concatenate(all_labels)


test_path="D:\\Datasets\\Levir_croped_256\\LEVIR_CD\\test"
test_loader=DataLoader(change_detection_dataset(root_path=test_path),batch_size=4,shuffle=False,num_workers=0,pin_memory=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
print(device)

model1 = CGNet()
model1=model1.to(device)
model1.load_state_dict(torch.load("E://VS Codes Testing things//test_17-3-2025__ChangeFormer_CGNet - 6//checkpoints//checkpoint2-BCE-50epochs//ResUnet49.pth"))
model1.eval()

# model2 = CGNet()
# model2=model1.to(device)
# model2.load_state_dict(torch.load("E://VS Codes Testing things//test_13-3-2025__ChangeFormer+CGNet - 2//checkpoints//checkpoint4-BCE-100epochs//ResUnet99.pth"))
# model2.eval()


# Example usage
predictions, labels = get_predictions_and_labels(model1, test_loader, device)
predictions = predictions.flatten()
labels = labels.flatten()

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)

print (len(thresholds))

j_scores = tpr - fpr
optimal_idx = j_scores.argmax()
best_threshold = thresholds[optimal_idx]
print (best_threshold)

print(f"Best Threshold: {best_threshold}")
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Diagonal line
plt.xlabel('FPR')#'False Positive Rate')
plt.ylabel('TPR')#True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
