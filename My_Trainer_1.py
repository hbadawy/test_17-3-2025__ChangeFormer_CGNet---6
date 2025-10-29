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
from CGNet.Loss_Function import *
# from Loss_Function import *

import torch.optim.lr_scheduler as lr_scheduler

from skimage import exposure
from skimage.exposure import match_histograms

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random
import time

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

        # print ("first one:  ", pre_tensor.shape)

        ############# MATCHING MODULE ################################
        # pre_tensor = pre_tensor.numpy()
        # pre_tensor =  np.transpose(pre_tensor, (1, 2, 0))

        # # print ("sec one: ", pre_tensor.shape)

        # post_tensor = post_tensor.numpy()
        # post_tensor =  np.transpose(post_tensor, (1, 2, 0))

        # # pre_tensor = match_histograms(pre_tensor, post_tensor)
        # post_tensor = match_histograms(post_tensor, pre_tensor)   #opposite to the previous line

        # # post_tensor = match_histograms(post_tensor, pre_tensor)

        # # print ("third one: ", pre_tensor.shape)
       
        # pre_tensor =  np.transpose(pre_tensor, (2, 0, 1))
        # post_tensor =  np.transpose(post_tensor, (2, 0, 1))
        
        # # print ("forth one: ", pre_tensor.shape)

        # pre_tensor = transform(pre_tensor)
        # post_tensor = transform(post_tensor)

        # # print ("fith one: ", pre_tensor.shape)
        
        # pre_tensor = pre_tensor.permute(1, 0, 2)
        # post_tensor = post_tensor.permute(1, 0, 2)
        # # print ("sixth one: ", pre_tensor.shape)
        #####################################################################

        return {'pre':pre_tensor,'post':post_tensor,'label':label_tensor,'fname':fname}
    def __len__(self):
        return len(self.fname_list)
    

train_path="D:\\Datasets\\Levir_croped_256\\LEVIR_CD\\train"
test_path="D:\\Datasets\\Levir_croped_256\\LEVIR_CD\\test"
val_path="D:\\Datasets\\Levir_croped_256\\LEVIR_CD\\val"


train_loader=DataLoader(change_detection_dataset(root_path=train_path),batch_size=8,shuffle=True,num_workers=0,pin_memory=False)
test_loader=DataLoader(change_detection_dataset(root_path=test_path),batch_size=4,shuffle=False,num_workers=0,pin_memory=False)
val_loader=DataLoader(change_detection_dataset(root_path=val_path),batch_size=4,shuffle=False,num_workers=0,pin_memory=False)


"""
for i, data in enumerate(train_loader):
        pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
        fig=plt.figure(figsize=(15,5))
        print (f'pre_tensor.shape={pre_tensor.shape}')
        print (f'post_tensor.shape={post_tensor.shape}')
        print (f'label_tensor.shape={label_tensor.shape}')
        

        preplot=fig.add_subplot(131)
        preplot.imshow(pre_tensor[1,:,:,:].permute(1,2,0).numpy())
        preplot.set_title("pre-change")

        postplot=fig.add_subplot(132)
        postplot.set_title("post-change")
        postplot.imshow(post_tensor[1,:,:,:].permute(1,2,0).numpy())

        labelplot=fig.add_subplot(133)
        labelplot.set_title("change label")
        labelplot.imshow(label_tensor[1,:,:,:].permute(1,2,0).numpy())
        # transforms.ToPILImage()(pre_tensor[0,:,:,:])
        # transforms.ToPILImage()(post_tensor[0,:,:,:])
        # transforms.ToPILImage()(label_tensor[0,:,:,:])
        print(f'fname={fname[0]}')
        plt.show()
        break
"""

""""
model = BASE_Transformer(input_nc=3, output_nc=1, with_pos='learned')
x = torch.randn(8, 3, 256, 256)
y = torch.randn(8, 3, 256, 256)
out = model(x, y)
print ("out.shape:", out.shape)  # out.shape: torch.Size([8, 1, 256, 256])
"""

def train(model, train_loader, val_loader, optimizer, loss_function, device, num_epochs, save_path, scheduler):
    loss_graph_list=[]  # added new
    loss_graph_list_val=[]   #added new
    best_valid_loss = float("inf")
    # oa_graph_list=[]
    start_training_time = time.time()
    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        loss_list=[]
        model.train()
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
            # print ("pre_tensor shape is: ", pre_tensor.shape)
            # print ("post_tensor shape is: ", post_tensor.shape)
            # print ("label_tensor shape is: ", label_tensor.shape)
            pre_tensor = pre_tensor.to(device)
            post_tensor = post_tensor.to(device)
            label_tensor = label_tensor.to(device)
            prediction = model(pre_tensor, post_tensor)

            total_loss=loss_function(prediction,label_tensor)
            loss_list.append(total_loss.item()) #only append the loss value and ignore the grad to save memory
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        loss_avg=sum(loss_list)/len(loss_list)
        print(f'Epoch {epoch} training completed, the average loss is {loss_avg}')# , Learning Rate: {scheduler.get_last_lr()[0]:.6f}".')

        num_val_epochs=1
        if (epoch+1) % num_val_epochs ==0:
            model.eval()
            OA_list=[]
            val_list=[]  #added new
            for _, data in enumerate(val_loader):
                pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
                pre_tensor = pre_tensor.to(device)
                post_tensor = post_tensor.to(device)
                label_tensor = label_tensor.to(device)
                probs = model(pre_tensor, post_tensor)
                
                val_loss=loss_function(probs,label_tensor)   #added new
                # scheduler.step(val_loss)
                val_list.append(val_loss.item())     #added new

                prediction = torch.where(probs>0.5,1,0)
                true_count=torch.sum(prediction==label_tensor)
                OA=true_count/prediction.numel()
                OA_list.append(OA)
            val_loss_avg=sum(val_list)/len(val_list) #added new
            # print("*"*10)
            end_time_epoch = time.time()
            total_epoch_time = end_time_epoch - start_time_epoch
            print(f'Epoch {epoch} evaluation completed, Epoch Time: {total_epoch_time:.2f} seconds, the average OA is {sum(OA_list)/len(OA_list)}, the val loss is {val_loss_avg}') #added new
            print("*"*10)

            if val_loss_avg < best_valid_loss:
                best_valid_loss = val_loss_avg
                torch.save(model.state_dict(),os.path.join(save_path,"ResUnet"+str(epoch)+".pth"))
            
        loss_graph_list.append(loss_avg)   #added new
        loss_graph_list_val.append(val_loss_avg)   #added new
        # oa_graph_list.append(sum(OA_list)/len(OA_list))
    end_training_time = time.time()
    total_training_time = end_training_time - start_training_time
    print(f'[[Total Training Time: {total_training_time:.2f} seconds]]')
    return loss_graph_list, loss_graph_list_val

############## Train the model ##################
############## Train the model ##################

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"


    print(device)

    model = CGNet()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # class_weights = torch.tensor([2.0]).to(device)
    # loss_function = torch.nn.BCELoss(weight=class_weights)

    # pos_weight = torch.tensor([0.7]).to(device)   #positive class weight
    # loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # loss_function = FocalLoss3()
    # loss_function = FocalTverskyLoss()
    loss_function = nn.BCELoss()
    # loss_function = DiceLoss3()
    # loss_function = FocalLoss4()
    # loss_function = LogCosh3()
    # loss_function = DiceBCELoss()
    # loss_function = WeightedBCELoss()

    num_epochs = 50
    save_path = "E://VS Codes Testing things//test_17-3-2025__ChangeFormer_CGNet - 6//checkpoints//checkpoint2-BCE-50epochs"
    os.makedirs(save_path, exist_ok=True)

    loss_gr_list, loss_gr_list_val = train(model, train_loader=train_loader, val_loader=val_loader, 
                                              optimizer=optimizer, loss_function=loss_function, device=device, 
                                                num_epochs=num_epochs, save_path=save_path, scheduler=scheduler)
    



    ########################### PLOT Training and Validation Losses ###############################
    epoch_graph_list = torch.arange(0, num_epochs)
    numbers_list = epoch_graph_list.tolist()

    # plt.figure(figsize=(8, 6))
    plt.plot(epoch_graph_list, loss_gr_list, linestyle='-', color='b', label='Train Loss')
    plt.plot(epoch_graph_list, loss_gr_list_val, linestyle='-', color='r', label='Val Loss')
    # plt.plot(epoch_graph_list, oa_gr_list, linestyle='-', color='c', label='OA')


    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.legend()
    # Display the plot
    # plt.grid(True)
    plt.show()



############## Test the model ##################
############## Test the model ##################

############## Confusion Matrix ##################

# def confusion(prediction, truth):
#     """ Returns the confusion matrix for the values in the `prediction` and `truth`
#     tensors, i.e. the amount of positions where the values of `prediction`
#     and `truth` are
#     - 1 and 1 (True Positive)
#     - 1 and 0 (False Positive)
#     - 0 and 0 (True Negative)
#     - 0 and 1 (False Negative)
#     """

#     confusion_vector = prediction / truth
#     # Element-wise division of the 2 tensors returns a new tensor which holds a
#     # unique value for each case:
#     #   1     where prediction and truth are 1 (True Positive)
#     #   inf   where prediction is 1 and truth is 0 (False Positive)
#     #   nan   where prediction and truth are 0 (True Negative)
#     #   0     where prediction is 0 and truth is 1 (False Negative)

#     true_positives = torch.sum(confusion_vector == 1).item()
#     false_positives = torch.sum(confusion_vector == float('inf')).item()
#     true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
#     false_negatives = torch.sum(confusion_vector == 0).item()

#     return true_positives, false_positives, true_negatives, false_negatives


# ############## Test the model ##################
# from torchvision.utils import save_image

# model = ChangeFormerV1()
# model=model.to(device)
# model.load_state_dict(torch.load("E://VS Projects//test_20-2-2025_ChangeFormer//ResUnet29.pth"))

# test_results_path="E://VS Projects//test_20-2-2025_ChangeFormer//test_results1"
# os.makedirs(test_results_path,exist_ok=True)
# TP=0
# TN=0
# FP=0
# FN=0
# for _, data in enumerate(test_loader):
#     pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
#     pre_tensor = pre_tensor.to(device)
#     post_tensor = post_tensor.to(device)
#     label_tensor = label_tensor.to(device)
#     probs = model(pre_tensor, post_tensor)
#     prediction = torch.where(probs>0.5,1.0,0.0)
#     true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,label_tensor)
#     TP+=true_positives
#     TN+=true_negatives
#     FP+=false_positives
#     FN+=false_negatives
#     for i in range(prediction.shape[0]):
#         save_image(prediction[i,:,:,:].cpu(), os.path.join(test_results_path, fname[i]))


# ################## Visualize the results ##################
# import matplotlib.pyplot as plt
# import numpy as np

# pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
# fig=plt.figure(figsize=(15,5))

# preplot=fig.add_subplot(141)
# preplot.imshow(pre_tensor[1,:,:,:].permute(1,2,0).numpy())
# preplot.set_title("pre-change")

# postplot=fig.add_subplot(142)
# postplot.set_title("post-change")
# postplot.imshow(post_tensor[1,:,:,:].permute(1,2,0).numpy())

# postplot=fig.add_subplot(143)
# postplot.set_title("label-tensor")
# postplot.imshow(label_tensor[1,:,:,:].permute(1,2,0).numpy())

# labelplot=fig.add_subplot(144)
# labelplot.set_title("prediction")
# labelplot.imshow(prediction[1,:,:,:].permute(1,2,0).cpu().numpy())
# # transforms.ToPILImage()(pre_tensor[0,:,:,:])
# # transforms.ToPILImage()(post_tensor[0,:,:,:])
# # transforms.ToPILImage()(label_tensor[0,:,:,:])
# print(f'fname={fname[0]}')
# plt.show()

# ############# Calculate the metrics ##############
# OA=(TP+TN)/(TP+TN+FP+FN)
# Precision=TP/(TP+FP)
# Recall=TP/(TP+FN)
# F1_score=2*Precision*Recall/(Precision+Recall)
# print(f'OA={OA:.3f}, Precision={Precision:.3f}, Recall={Recall:.3f}, F1-score={F1_score:.3f}')

