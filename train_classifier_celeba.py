#!/usr/bin/env python
# coding: utf-8

# # Let's train a classifier

# In[1]:


import os
# os.system("pip install pandas")
# os.system("pip install torchvision")
os.system("CUDA_LAUNCH_BLOCKING=1")
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

import time

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

torch.manual_seed(1)

# In[2]:


# choose experiment
size_train = 10000 #86744
experiments = ["FairFace", "CelebA", "CelebA only white", "CelebA augmented"]
exp = 1

# set celeb paths
celeb_attr_path = "datasets/celeba/list_attr_celeba.txt"
celeb_partitions_path = 'datasets/celeba/list_eval_partition.txt'
celeb_race_path = "CelebA/races/races_ff.csv"
celeb_label_dir = "CelebA/labels_split/"
celeb_img_dir = "CelebA/cropped/"
celeb_img_aug_dir = "CelebA/augmented/"
celeb_train_csv = f"train_{size_train}_samples_random.csv" # "train_total.csv"
celeb_train_only_white_csv = f"train_{size_train}_samples_random_white.csv"
celeb_train_aug_csv = f"train_aug_{size_train}_samples.csv"
celeb_val_csv = "val_total.csv"
celeb_test_csv = "test_total.csv"


# set fairface paths
ff_img_dir = "fairface/dataset/fairface-img-margin125-trainval"
ff_label_dir = "fairface/dataset/"
ff_train_csv = "fairface_label_train.csv"
ff_val_csv = "fairface_label_val.csv"


# set hyperparameters
learning_rates = [5e-5, 5e-5, 5e-5, 2e-5]
lr = learning_rates[exp]
num_epochs = 10

# Architecture
feat_size = (256, 256)
bs_train = 128
bs_val = 128
bs_test = 128
device = 'cuda:3'


races = ["Black", "Indian", "Latino", "Middle Eastern", "Southeast Asian", "East Asian", "White"]
ignored_attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Pale_Skin"]

# In[3]:


# define datasets
class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None, ignored_attributes=[]):
    
        df = pd.read_csv(csv_path, index_col=None)
        # print(df.head())
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df["Image_Name"].values
        self.races = df["Race"].values
        drop_cols = ["Image_Name", "Race"] + ignored_attributes
        self.y = np.expand_dims(np.array(df["Male"].values), axis=1) #df.drop(drop_cols, axis=1).values #
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        gt_race = self.races[index]
        return img, label, gt_race

    def __len__(self):
        return self.y.shape[0]
    


class FairFaceDataset(Dataset):
    """Custom Dataset for loading FairFace images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=None)
        # print(df.head())
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df["file"].values
        self.races = df["race"].replace("Latino_Hispanic", "Latino").values
        gender = df["gender"].replace("Male", 1).replace("Female", 0)
        self.y = np.expand_dims(np.array(gender.values), axis=1)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        gt_race = self.races[index]
        return img, label, gt_race

    def __len__(self):
        return self.y.shape[0]

# In[4]:


# create datasets based on current experiment
num_workers = 6
custom_transform = transforms.Compose([transforms.Resize(feat_size),
                                       transforms.ToTensor()])

# training dataset
if experiments[exp].startswith("CelebA"):

    if experiments[exp].endswith("augmented"):
        train_csv = celeb_train_aug_csv
        train_img_dir = celeb_img_aug_dir
    
    else:
        if "only white" in experiments[exp]:
            train_csv = celeb_train_only_white_csv
        else:
            train_csv = celeb_train_csv
        train_img_dir = celeb_img_dir

    train_dataset = CelebaDataset(csv_path=celeb_label_dir + train_csv,
                                img_dir=train_img_dir,
                                transform=custom_transform,
                                ignored_attributes=ignored_attributes)

if experiments[exp].startswith("FairFace"):
    train_dataset = FairFaceDataset(csv_path=ff_label_dir + ff_train_csv,
                                    img_dir=ff_img_dir,
                                    transform=custom_transform)


# validation dataset
val_dataset = FairFaceDataset(csv_path=ff_label_dir + ff_val_csv,
                                img_dir=ff_img_dir,
                                transform=custom_transform)

# val_dataset = CelebaDataset(csv_path=celeb_label_dir + celeb_val_csv,
#                             img_dir=celeb_img_dir,
#                             transform=custom_transform,
#                             ignored_attributes=ignored_attributes)


# test datasets
test_dataset_celeb = CelebaDataset(csv_path=celeb_label_dir + celeb_test_csv,
                            img_dir=celeb_img_dir,
                            transform=custom_transform,
                            ignored_attributes=ignored_attributes)

test_dataset_ff = val_dataset


# create dataloaders on these datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=bs_train,
                          shuffle=True,
                          num_workers=num_workers)

val_loader = DataLoader(dataset=val_dataset,
                          batch_size=bs_val,
                          shuffle=False,
                          num_workers=num_workers)

test_loader_celeb = DataLoader(dataset=test_dataset_celeb,
                          batch_size=bs_test,
                          shuffle=False,
                          num_workers=num_workers)

test_loader_ff = DataLoader(dataset=test_dataset_ff,
                          batch_size=bs_test,
                          shuffle=False,
                          num_workers=num_workers)

# In[5]:


# build model, define loss and create optimizer
model = resnet34(weights=ResNet34_Weights.DEFAULT)
model.to(device)
num_attr_predicted = train_dataset.y.shape[1]
fc_layer = nn.Linear(1000, num_attr_predicted, device=device)
sigmoid = nn.Sigmoid()
bin_ce = nn.BCELoss()
params = list(model.parameters()) + list(fc_layer.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

# In[6]:


# define evaluation procedure
def evaluate_metrics(model, data_loader, device, show_tqdm=False):

    correct_predictions = np.zeros(len(races))
    true_pos = np.zeros(len(races))
    true_neg = np.zeros(len(races))
    positive_preds = np.zeros(len(races))
    positive_targets = np.zeros(len(races))
    num_examples = np.zeros(len(races))
    total_examples = len(data_loader.dataset) 

    # total_it = int(np.ceil(total_examples / data_loader.batch_size))
    for _, (features, targets, gt_races) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating", disable=not show_tqdm):

        features = features.to(device)
        probas = sigmoid(fc_layer(model(features)))
        prediction = (probas >= 0.5).cpu().numpy()
        targets = targets.numpy()

        # prepape annotated races for metric split afterwards
        gt_races = np.array([races.index(race) for race in gt_races])
        gt_races = np.expand_dims(gt_races, axis=1)
        gt_races = np.broadcast_to(gt_races, prediction.shape)

        # collect the necessary data split by annotated race
        for j in range(len(races)):
            correct_preds = (gt_races == j) & (prediction == targets)
            true_pos[j] += (correct_preds & (prediction == 1)).sum()
            true_neg[j] += (correct_preds & (prediction == 0)).sum()
            correct_predictions[j] += correct_preds.sum()
            positive_targets[j] += ((gt_races == j) & (targets == 1)).sum()
            positive_preds[j] += np.where(gt_races == j, prediction, 0).sum()
            num_examples[j] += (gt_races == j).sum()

    # calculate and return metrics    
    zero = 1e-10
    print("Race distribution:", num_examples/targets.shape[1], "Total:", total_examples)

    total_accuracy = correct_predictions.sum() / num_examples.sum()
    accuracies = correct_predictions / (num_examples + zero)
    accs_out = [f"{a:.2%}" for a in accuracies]
    max_acc_disparity = np.log(max(accuracies)/min(accuracies))

    total_precision = true_pos.sum() / (positive_preds.sum() + zero)
    precisions = [f"{p:.2%}" for p in true_pos / (positive_preds + zero)]

    total_recall = true_pos.sum() / (positive_targets.sum() + zero)
    recalls = [f"{r:.2%}" for r in true_pos / (positive_targets + zero)]
    return total_accuracy, accs_out, max_acc_disparity, total_precision, precisions, total_recall, recalls


def get_elapsed_time(start_time):
    elapsed = int(time.time() - start_time)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"

# In[7]:


# Training loop
start_time = time.time()

print(f"Initiating experiment '{experiments[exp]}' with a lr of {lr} and {size_train} samples on device {device}")


for epoch in range(num_epochs):
    
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {(epoch+1):02d}/{num_epochs:02d}")
    for _, (features, targets, _) in pbar:
        
        features = features.to(device)
        targets = targets.float().to(device)
            
        # forward and backward pass
        model_output = model(features)
        logits = sigmoid(fc_layer(model_output))
        loss = bin_ce(logits, targets)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        optimizer.zero_grad()
        
        loss.backward()
        
        # update model params 
        optimizer.step()
        
        # if batch_idx == 0:
        #     break 

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        acc_total, accs, max_acc_disp, prec_total, precs, rec_total, recs = evaluate_metrics(model, val_loader, device)
        print(f"Evaluation epoch {(epoch+1):02d}/{num_epochs:02d}:")
        print(f"Total accuracy: {acc_total:.2%}\t| Accuracies:\t{accs} | Max disparity: {max_acc_disp:.4f}")
        print(f"Total precision: {prec_total:.2%}\t| Precisions:\t{precs}")
        print(f"Total recall: {rec_total:.2%}\t| Recalls:\t{recs}\n")
    
print(f"Total Training Time: {get_elapsed_time(start_time)}")

# In[8]:


# evaluate experiment on test sets
with torch.set_grad_enabled(False): # save memory during inference
    print(f"\nEvaluation of experiment: '{experiments[exp]}'\n")

    # evaluation CelebA
    acc_total, accs, max_acc_disp, prec_total, precs, rec_total, recs = evaluate_metrics(model, test_loader_celeb, device, show_tqdm=True)
    print("\nEvaluation CelebA test set:")
    print(f"Total accuracy: {acc_total:.2%}\t| Accuracies:\t{accs}")
    print(f"Maximum accuracy disparity: {max_acc_disp:.4f}")
    print(f"Total precision: {prec_total:.2%}\t| Precisions:\t{precs}")
    print(f"Total recall: {rec_total:.2%}\t| Recalls:\t{recs}\n")

    # evaluation FairFace
    acc_total, accs, max_acc_disp, prec_total, precs, rec_total, recs = evaluate_metrics(model, test_loader_ff, device, show_tqdm=True)
    print("\nEvaluation FairFace test set:")
    print(f"Total accuracy: {acc_total:.2%}\t| Accuracies:\t{accs}")
    print(f"Maximum accuracy disparity: {max_acc_disp:.4f}")
    print(f"Total precision: {prec_total:.2%}\t| Precisions:\t{precs}")
    print(f"Total recall: {rec_total:.2%}\t| Recalls:\t{recs}\n")

