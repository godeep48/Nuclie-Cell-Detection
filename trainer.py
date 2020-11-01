from utils.data_preprocess import LoadDataSet, get_train_transform
from utils.image_process import format_image, format_mask
from utils.metrices import DiceLoss, IoU
from utils.checkpoint import save_ckp, load_ckp
from core.unet import UNet

import torch

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm as tqdm

import torch.nn.functional as F
from PIL import Image
from torch import nn
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


config_dict = {
	"train_dir": './data/',
	"split_ratio":0.3,
	"batch_size":10,
	"learning_rate": 1e-3,
	"checkpoint_path": '/model/chkpoint_',
	"bestmodel_path": '/model/bestmodel.pt',
	"epochs": 100
}



#Directory of dataset contain images and masks
TRAIN_PATH = config_dict["train_dir"]

#Load dataset from the train directory
train_dataset = LoadDataSet(TRAIN_PATH, transform=get_train_transform())

## Split train and validation set.
split_ratio = config_dict["split_ratio"]
train_size=int(np.round(train_dataset.__len__()*(1 - split_ratio),0))
valid_size=int(np.round(train_dataset.__len__()*split_ratio,0))
print(train_size, valid_size)
train_data, valid_data = random_split(train_dataset, [train_size, valid_size]) #2491, 200

#DataLoader for train dataset
batch_size = config_dict["batch_size"]
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
#DataLoader for valid dataset
val_loader = DataLoader(dataset=valid_data, batch_size=10)

#Initialize model
model = UNet(3,1).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = config_dict["learning_rate"])

if not os.path.exists("/model"):
	os.makedirs("model")


def main():

	#from engine import evaluate
	criterion = DiceLoss()
	accuracy_metric = IoU()
	num_epochs=config_dict["epochs"]
	valid_loss_min = np.Inf
	
	checkpoint_path = config_dict["checkpoint_path"]
	best_model_path = config_dict["bestmodel_path"]
	
	total_train_loss = []
	total_train_score = []
	total_valid_loss = []
	total_valid_score = []
	
	losses_value = 0
	for epoch in range(num_epochs):
	  
	    train_loss = []
	    train_score = []
	    valid_loss = []
	    valid_score = []
	    #<-----------Training Loop---------------------------->
	    pbar = tqdm(train_loader, desc = 'description')
	    for x_train, y_train in pbar:
	      x_train = torch.autograd.Variable(x_train).cuda()
	      y_train = torch.autograd.Variable(y_train).cuda()
	      optimizer.zero_grad()
	      output = model(x_train)
	      #Loss
	      loss = criterion(output, y_train)
	      losses_value = loss.item()
	      #Score
	      score = accuracy_metric(output,y_train)
	      loss.backward()
	      optimizer.step()
	      train_loss.append(losses_value)
	      train_score.append(score.item())
	      #train_score.append(score)
	      pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")
	
	    #<---------------Validation Loop---------------------->
	    with torch.no_grad():
	      for image,mask in val_loader:
	        image = torch.autograd.Variable(image).cuda()
	        mask = torch.autograd.Variable(mask).cuda()
	        output = model(image)
	        ## Compute Loss Value.
	        loss = criterion(output, mask)
	        losses_value = loss.item()
	        ## Compute Accuracy Score
	        score = accuracy_metric(output,mask)
	        valid_loss.append(losses_value)
	        valid_score.append(score.item())
	
	    total_train_loss.append(np.mean(train_loss))
	    total_train_score.append(np.mean(train_score))
	    total_valid_loss.append(np.mean(valid_loss))
	    total_valid_score.append(np.mean(valid_score))
	    print(f"\n###############Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}###############")
	    print(f"###############Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}###############")
	
	    #Save best model Checkpoint
	    # create checkpoint variable and add important data
	    checkpoint = {
	        'epoch': epoch + 1,
	        'valid_loss_min': total_valid_loss[-1],
	        'state_dict': model.state_dict(),
	        'optimizer': optimizer.state_dict(),
	    }
	    
	    # save checkpoint
	    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
	    
	    ## TODO: save the model if validation loss has decreased
	    if total_valid_loss[-1] <= valid_loss_min:
	        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
	        # save checkpoint as best model
	        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
	        valid_loss_min = total_valid_loss[-1]

if __name__ == '__main__':
	main()