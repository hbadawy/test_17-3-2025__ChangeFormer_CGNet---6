
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


############## Test the model ##################
############## Test the model ##################

############## Confusion Matrix ##################

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


############## Test the model ##################
from torchvision.utils import save_image

if __name__ == "__main__":


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"


    print(device)

    model = CGNet()
    model=model.to(device)
    model.load_state_dict(torch.load("E://VS Codes Testing things//test_17-3-2025__ChangeFormer_CGNet - 6//checkpoints//checkpoint2-BCE-50epochs//ResUnet49.pth"))
    model.eval()
    test_results_path="E://VS Codes Testing things//test_17-3-2025__ChangeFormer_CGNet - 6//test_results//test_results2-BCE-50epochs"
    os.makedirs(test_results_path,exist_ok=True)
    TP=0
    TN=0
    FP=0
    FN=0
    start_time = time.time()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
            pre_tensor = pre_tensor.to(device)
            post_tensor = post_tensor.to(device)
            label_tensor = label_tensor.to(device)
            probs = model(pre_tensor, post_tensor)
            prediction = torch.where(probs>0.47,1.0,0.0)
            true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,label_tensor)
            TP+=true_positives
            TN+=true_negatives
            FP+=false_positives
            FN+=false_negatives
            for i in range(prediction.shape[0]):
                save_image(prediction[i,:,:,:].cpu(), os.path.join(test_results_path, fname[i]))
        end_time = time.time()
        inference_time = (end_time - start_time)/len(test_loader)
    ################## Visualize the results ##################
    import matplotlib.pyplot as plt
    import numpy as np

    pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
    fig=plt.figure(figsize=(10,10))

    preplot=fig.add_subplot(441)
    preplot.imshow(pre_tensor[1,:,:,:].permute(1,2,0).numpy())
    preplot.set_title("Pre-change")

    postplot=fig.add_subplot(442)
    postplot.set_title("Post-change")
    postplot.imshow(post_tensor[1,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(443)
    postplot.set_title("Label-tensor")
    postplot.imshow(label_tensor[1,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(444)
    labelplot.set_title("Prediction")
    labelplot.imshow(prediction[1,:,:,:].permute(1,2,0).cpu().numpy(), cmap='gray') 
    # transforms.ToPILImage()(pre_tensor[0,:,:,:])
    # transforms.ToPILImage()(post_tensor[0,:,:,:])
    # transforms.ToPILImage()(label_tensor[0,:,:,:])



    preplot=fig.add_subplot(445)
    preplot.imshow(pre_tensor[0,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(446)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[0,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(447)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[0,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(448)
    # labelplot.set_title("prediction")
    labelplot.imshow(prediction[0,:,:,:].permute(1,2,0).cpu().numpy(), cmap='gray') 
    print(f'fname={fname[0]}')


    preplot=fig.add_subplot(449)
    preplot.imshow(pre_tensor[3,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(4,4,10)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[3,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(4,4,11)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[3,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(4,4,12)
    # labelplot.set_title("prediction")
    labelplot.imshow(prediction[3,:,:,:].permute(1,2,0).cpu().numpy(),cmap='gray') 
    print(f'fname={fname[0]}')

        
    preplot=fig.add_subplot(4,4,13)
    preplot.imshow(pre_tensor[2,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(4,4,14)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[2,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(4,4,15)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[2,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(4,4,16)
    # labelplot.set_title("prediction")
    labelplot.imshow(prediction[2,:,:,:].permute(1,2,0).cpu().numpy(),cmap='gray') 
    print(f'fname={fname[0]}')

    plt.show()

    ############# Calculate Inference time of Model #############
    # dummy_x1 = torch.rand([1,3,256,256]).to(device)
    # dummy_x2 = torch.rand([1,3,256,256]).to(device)
    # model.eval()
    # start_time = time.time()
    # y = model (dummy_x1, dummy_x2)
    # end_time = time.time()
    # inference_time = end_time - start_time
    # print(f'Inference Time: {inference_time:.6f} seconds')

    ############# Calculate the metrics ##############
    OA=(TP+TN)/(TP+TN+FP+FN)   # Observed Accuracy
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1_score=2*Precision*Recall/(Precision+Recall)
    IoU1 =TP/(TP+FP+FN)
    IoU2 =TN/(TN+FP+FN)
    mIoU = (IoU1+IoU2)/2
    Spec = TN/(TN+FP)

    P1 = TP/(TP+FP)
    R1 = TP/(TP+FN)
    P2 = TN /(TN+FN)
    R2 = TN /(TN+FP)
    F1 = 2*P1*R1/(P1+R1)
    F2 = 2*P2*R2/(P2+R2)
    microF = (F1+F2)/2

    total = TP+TN+FP+FN 
    PRE = ((TP+FP)*(TP+FN) + (TN+FN)*(TN+FP)) / (total ** 2)  # Expected accuracy
    Kappa = (OA - PRE) / (1 - PRE)

    print ("                                                             ")
    print("--------------------------Results-----------------------------")
    print(f'Inference Time: {inference_time:.6f} sec, OA={OA:.3f}, Precision={Precision:.3f}, Recall/Sensitivity={Recall:.3f}, Specificity={Spec:.3f}, Kappa={Kappa:.3f}, mIoU={mIoU:.3f} IoU={IoU1:.3f}, mF1-score={microF:.3f}, F1-score={F1_score:.3f}' )
    print(f'Inference Time, OA, Precision, Recall/Sensitivity, Specificity, Kappa, mIoU, IoU, mF1-score, F1-score' )
    print(f'{inference_time:.4f}, {OA:.3f}, {Precision:.3f}, {Recall:.3f}, {Spec:.3f},{Kappa:.3f}, {mIoU:.3f} , {IoU1:.3f}, {microF:.3f}, {F1_score:.3f}' )
    print ("-------------------------------------------------------------")


    ################## ROC curve ######################
