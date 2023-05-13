#!/usr/bin/env python
# coding: utf-8

# # Let's train a classifier

# In[3]:


import os
# os.system("pip install foolbox")
# os.system("pip install torchmetrics")
# os.system("pip install neptune")
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

import neptune
from rtpt import RTPT
import foolbox

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

torch.manual_seed(1)

# In[4]:


# choose experiment
size_train = 10000 #86744
experiments = ["FairFace", "CelebA", "CelebA only white", "CelebA augmented"]
exp = 3

# set celeb paths
celeb_attr_path = "datasets/celeba/list_attr_celeba.txt"
celeb_partitions_path = 'datasets/celeba/list_eval_partition.txt'
celeb_race_path = "CelebA/races/races_ff.csv"
celeb_label_dir = "CelebA/labels_split/"
celeb_img_dir = "CelebA/cropped/"
celeb_img_aug_dir = "CelebA/augmentations/aug_precise_prompts_strong" #"CelebA/augmentations/aug_0_5__0/"
celeb_train_csv = f"train_{size_train}_samples_random.csv" # "train_total.csv"
celeb_train_only_white_csv = f"train_{size_train}_samples_random_white.csv"
celeb_train_aug_csv = "train_aug_10k_total_fid_harm_precise_prompts.csv" #"train_aug_10k_all_samples.csv"#"train_aug_10k_distinct_samples_harm.csv" #f"train_aug_{size_train}_distinct_samples_correct_gender.csv"
celeb_val_csv = "val_total.csv"
celeb_test_csv = "test_total.csv"


# set fairface paths
ff_img_dir = "fairface/dataset/fairface-img-margin125-trainval"
ff_label_dir = "fairface/dataset/"
ff_train_csv = "fairface_label_train.csv"
ff_val_csv = "fairface_label_val.csv"


# set hyperparameters
lr = 1e-4
num_epochs = 100

# Architecture
pretrained = False
feat_size = (256, 256)
bs_train = 128
bs_val = 128
bs_test = 128
device = 'cuda:1'
scheduler_step_size = 3
scheduler_gamma = 0.9


races = ["Black", "Indian", "Latino", "Middle Eastern", "Southeast Asian", "East Asian", "White"]
# ignored_attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Pale_Skin"]

# In[5]:


# define datasets
class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=None)
        # print(df.head())
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df["Image_Name"].values
        self.races = df["Race"].values
        # drop_cols = ["Image_Name", "Race"] + ignored_attributes
        # male = df["Male"].replace("Male", 1).replace("Female", 0)
        # if "Original_Gender" in df.columns:
            # df = df.drop("Male", axis=1).rename(columns={"Original_Gender": "Male"})
            # print(df.columns)
        self.y = np.expand_dims(np.array(df["Male"].values), axis=1) #df.drop(drop_cols, axis=1).values #
        # self.y = np.expand_dims(np.array(df["Male"].values), axis=1) #df.drop(drop_cols, axis=1).values #
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

# In[10]:


# create datasets based on current experiment
num_workers = 6
custom_transform = transforms.Compose([transforms.Resize(feat_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                       ])

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
                                transform=custom_transform)

if experiments[exp].startswith("FairFace"):
    train_dataset = FairFaceDataset(csv_path=ff_label_dir + ff_train_csv,
                                    img_dir=ff_img_dir,
                                    transform=custom_transform)


# validation dataset
val_dataset_celeb = CelebaDataset(csv_path=celeb_label_dir + celeb_val_csv,
                            img_dir=celeb_img_dir,
                            transform=custom_transform)

val_dataset_ff = FairFaceDataset(csv_path=ff_label_dir + ff_val_csv,
                                img_dir=ff_img_dir,
                                transform=custom_transform)



# test datasets
test_dataset_celeb = CelebaDataset(csv_path=celeb_label_dir + celeb_test_csv,
                            img_dir=celeb_img_dir,
                            transform=custom_transform)

test_dataset_ff = val_dataset_ff


# create dataloaders on these datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=bs_train,
                          shuffle=True,
                          num_workers=num_workers)

val_loader = DataLoader(dataset=val_dataset_ff,
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

# In[11]:


# build model, define loss and create optimizer
model = resnet34(weights=(ResNet34_Weights.DEFAULT if pretrained else None))

model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                         nn.ReLU(inplace=True),
                         nn.Linear(256, train_dataset.y.shape[1]))
model.to(device)
bin_ce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

# initialize experiment tracking
run = neptune.init_run(
    project="danielritter0508/aug-precise-pre-trained",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9uZXctdWkubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDI4OGY4ZC05YmUzLTRjNzAtOTE2NS0xNzU0NTkyMTJiYmUifQ==",
)
run["params"] = {"experiment": experiments[exp], "learning_rate": lr, "num_epochs": num_epochs, 
                 "scheduler_step": scheduler_step_size, "gamma": scheduler_gamma, "pretrained": pretrained}

# In[8]:


# calculate the root of the mean of the squared differences between the values and their average
def get_RMSD(values):
    # calculate the average of the values
    avg = sum(values) / len(values)

    # calculate the sum of the squared deviations
    squared_deviations_sum = sum((x - avg) ** 2 for x in values)
    
    # calculate the mean squared deviation and take the root of it
    rmsd = (squared_deviations_sum / len(values)) ** 0.5
    return rmsd


# calculate all necessary metrics
def evaluate_metrics(model, data_loader, device, log_description, show_tqdm=False, decimals=4):

    # initialize all parameters with zeros
    correct_predictions, true_pos, true_neg, pos_preds, pos_targets, num_examples = np.zeros((6, len(races)))

    # iterate through validation/test data and collect necessary basic data
    for _, (features, targets, gt_races) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating", disable=not show_tqdm):

        predictions = (model(features.to(device)) >= 0).cpu().numpy()
        targets = targets.numpy()

        # prepape annotated races for race-wise split afterwards
        gt_races = np.array([races.index(race) for race in gt_races])
        gt_races = np.expand_dims(gt_races, axis=1)

        # collect the necessary data split by annotated race
        for j in range(len(races)):
            correct_preds = (gt_races == j) & (predictions == targets)
            true_pos[j] += (correct_preds & (predictions == 1)).sum()
            true_neg[j] += (correct_preds & (predictions == 0)).sum()
            correct_predictions[j] += correct_preds.sum()
            pos_targets[j] += ((gt_races == j) & (targets == 1)).sum()
            pos_preds[j] += np.where(gt_races == j, predictions, 0).sum()
            num_examples[j] += (gt_races == j).sum()

    # calculate the metrics    
    zero = 1e-10

    # accuracy
    total_acc = correct_predictions.sum() / num_examples.sum()
    accs = correct_predictions / (num_examples + zero)

    # bias (inter-race accuracy variation)
    mad = np.log(max(accs) / min(accs))
    rmsd_acc = get_RMSD(accs)

    # precision
    total_prec = true_pos.sum() / (pos_preds.sum() + zero)
    precs = true_pos / (pos_preds + zero)

    # recall
    total_rec = true_pos.sum() / (pos_targets.sum() + zero)
    recs = true_pos / (pos_targets + zero)

    # F1 score
    total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec)
    f1_scores = 2 * precs * recs / (precs + recs)


    # round and track general results
    total_acc, mad, rmsd_acc, total_prec, total_rec, total_f1 = np.round((total_acc, mad, rmsd_acc, total_prec, total_rec, total_f1), decimals)
    metrics = {"total_acc": total_acc, "MAD": mad, "RMSD": rmsd_acc, "total_prec": total_prec, "total_rec": total_rec, "total_f1": total_f1}
    for (name, value) in metrics.items():
        run[log_description + "/" + name].append(value)


    # round and track race-specific results
    accs, precs, recs, f1_scores = np.round((accs, precs, recs, f1_scores), decimals)
    for race, acc, prec, rec, f1  in zip(races, accs, precs, recs, f1_scores):
        run[log_description + "/acc/" + race].append(acc)
        run[log_description + "/prec/" + race].append(prec)
        run[log_description + "/rec/" + race].append(rec)
        run[log_description + "/f1/" + race].append(f1)
    
    return total_acc, accs, mad, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs



def get_elapsed_time(start_time):
    elapsed = int(time.time() - start_time)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"

# In[12]:


# Training loop
start_time = time.time()

rtpt = RTPT('DR', 'Train_Gender_Classifier', num_epochs)
rtpt.start()

print(f"Initiating experiment '{experiments[exp]}' with a lr of {lr} and {size_train} samples on device {device}")

for epoch in range(num_epochs):
    
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {(epoch+1):03d}/{num_epochs:03d}")
    for batch_idx, (features, targets, _) in pbar:
        
        features = features.to(device)
        targets = targets.float().to(device)
            
        # forward and backward pass
        logits = model(features)
        loss = bin_ce(logits, targets)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        run["train/loss"].append(loss.item())

        optimizer.zero_grad()
        
        loss.backward()
        
        # update model params 
        optimizer.step()
        
        
        # if batch_idx == 3:
        #     break
        #     first = features[0].cpu().numpy()
        #     first = np.moveaxis(first, 0, -1) * 255
        #     first = first.astype(np.uint8)
        #     image = Image.fromarray(first)
        #     print("Woman" if int(targets[0].item()) == 0 else "Man")
        #     image.show()
       
            # print("Gender:", "Man" if targets)

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        # total_acc, accs, max_acc_disp, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs = 
        evaluate_metrics(model, val_loader, device, "valid")
        # print(f"Evaluation epoch {(epoch+1):02d}/{num_epochs:02d}:")
        # print(f"Total accuracy: {total_acc:.2%}\t| Accuracies:\t{accs} | Max disparity: {max_acc_disp}")
        # print(f"Total precision: {total_prec:.2%}\t| Precisions:\t{precs}")
        # print(f"Total recall: {total_rec:.2%}\t| Recalls:\t{recs}\n")
        
    scheduler.step()
    rtpt.step()
    
print(f"Total Training Time: {get_elapsed_time(start_time)}")

# In[ ]:


# evaluate experiment on test sets
with torch.set_grad_enabled(False): # save memory during inference
    print(f"\nEvaluation of experiment: '{experiments[exp]}'\n")

    # evaluation CelebA
    total_acc, accs, MAD, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs = evaluate_metrics(model, test_loader_celeb, device, "eval_celeb", show_tqdm=True)
    print("\nEvaluation CelebA test set:")
    print(f"Total accuracy: {total_acc:.2%}\t| Accuracies:\t{accs}")
    print(f"Maximum accuracy disparity: {MAD}")
    print(f"Total precision: {total_prec:.2%}\t| Precisions:\t{precs}")
    print(f"Total recall: {total_rec:.2%}\t| Recalls:\t{recs}\n")

    # evaluation FairFace
    total_acc, accs, MAD, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs = evaluate_metrics(model, test_loader_ff, device, "eval_ff", show_tqdm=True)
    print("\nEvaluation FairFace test set:")
    print(f"Total accuracy: {total_acc:.2%}\t| Accuracies:\t{accs}")
    print(f"Maximum accuracy disparity: {MAD}")
    print(f"Total precision: {total_prec:.2%}\t| Precisions:\t{precs}")
    print(f"Total recall: {total_rec:.2%}\t| Recalls:\t{recs}\n")

