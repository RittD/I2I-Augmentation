#!/usr/bin/env python
# coding: utf-8

# # Let's train a classifier

# In[10]:


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
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, densenet121, densenet169, mobilenet_v3_small, mobilenet_v3_large, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, DenseNet121_Weights, DenseNet169_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

import time

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import neptune
from rtpt import RTPT
import foolbox as fb
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import L2AdditiveGaussianNoiseAttack, LinfFastGradientAttack, LinfDeepFoolAttack
from foolbox.criteria import Misclassification
import eagerpy as ep

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

torch.manual_seed(1)


# In[11]:


# choose experiment
size_train = 10000 #86744 #
exp_setups = ["only white", "augmented", "original"]
is_celeba = False
exp =  2

# set hyperparameters
model_desc = "resnet34" #densenet169  resnet34
smallest_eps = 0
biggest_eps = 2e-3
eps_steps = 25
eps_attack = np.linspace(start=smallest_eps, stop=biggest_eps, num=eps_steps)#[] #[2e-4, 1e-4, 5e-5] #[1e-5, 2e-5, 5e-5]# #[5e-5, 1e-4, 5e-4] #[2e-4, 1e-4, 5e-5] #Gaussian: approx. 20, FGSM: approx. 5e-5
device = 'cuda:7'
lr = 1e-4
num_epochs = 100
pretrained = True
feat_size = (256, 256)
bs_train = 128
bs_val = 128
bs_test = 128
scheduler_step_size = 3
scheduler_gamma = 0.9

# save/load checkpoints
ckpt_dir = "ckpts/"
load_ckpt = True
exp_num = 22 + exp # 25 20 21
after_epoch = 100
# ckpt_load_path = f"{ckpt_dir}AUGPRE-{exp_num}_{after_epoch}_epochs.pt"
ckpt_load_path = f"{ckpt_dir}EX-{exp_num}_{after_epoch}_epochs.pt"
save_ckpt_after_epochs = [50, 100]#92, 94, 96, 98, 


# set celeb paths
celeb_dir = "../CelebA/"
celeb_label_dir = celeb_dir + "labels/"
celeb_img_dir = celeb_dir + "cropped/"
celeb_img_aug_dir = celeb_dir + "augmentations/aug_precise_prompts_strong" #"CelebA/augmentations/aug_0_5__0/"
celeb_train_csv = f"train_{size_train}_samples_random.csv" # "train_total.csv"
celeb_train_only_white_csv = f"train_{size_train}_samples_random_white.csv"
celeb_train_aug_csv = "train_aug_10k_total_fid_harm_precise_prompts.csv" #"train_aug_10k_all_samples.csv"#"train_aug_10k_distinct_samples_harm.csv" #f"train_aug_{size_train}_distinct_samples_correct_gender.csv"
celeb_val_csv = "val_total.csv"
celeb_test_csv = "test_total.csv"


# set fairface paths
ff_dir = "../fairface/dataset/"
ff_label_dir = ff_dir + "labels/"
ff_img_dir = ff_dir + "fairface-img-margin125-trainval"
ff_img_aug_dir = ff_dir + "augmentations/mixed_caps_total_84k/"
ff_train_csv = "permutations/train_1.csv" #"fairface_label_train.csv" #"fairface_label_train_random_10k.csv" #
ff_train_only_white_csv = "ff_train_only_white_10k.csv"
ff_train_aug_csv = "ff_mixed_caps_10k_balanced.csv"
ff_val_csv = "val.csv"
ff_test_csv = "test.csv"


races = ["Black", "Indian", "Latino", "Middle Eastern", "Southeast Asian", "East Asian", "White"]
# ignored_attributes = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Pale_Skin"]


# In[12]:


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
        self.y = df["Male"].values #df.drop(drop_cols, axis=1).values #
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
        self.img_names = df["Image_Name"].values
        self.races = df["Race"].replace("Latino_Hispanic", "Latino").values
        self.y = df["Gender"].replace("Male", 1).replace("Female", 0).values
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


# In[13]:


# create datasets based on current experiment
num_workers = 6

# do normalization here when not performing attack
# if do_attack:
# custom_transform = transforms.Compose([transforms.Resize(feat_size),
#                                     transforms.ToTensor()])
# else:
custom_transform = transforms.Compose([transforms.Resize(feat_size),
                                    transforms.ToTensor()])#,
                                    # transforms.RandomHorizontalFlip()]) #,
                                    # transforms.Normalize(mean=(0.421, 0.337, 0.301), std=([0.290, 0.269, 0.264]))])
                                    # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

# training dataset
if is_celeba:

    if exp_setups[exp] == "augmented":
        train_csv = celeb_train_aug_csv
        train_img_dir = celeb_img_aug_dir
    
    else:
        if exp_setups[exp] == "only white":
            train_csv = celeb_train_only_white_csv
        else:
            train_csv = celeb_train_csv
        train_img_dir = celeb_img_dir

    train_dataset = CelebaDataset(csv_path=celeb_label_dir + train_csv,
                                img_dir=train_img_dir,
                                transform=custom_transform)

else:
    if exp_setups[exp] == "augmented":
        train_csv = ff_train_aug_csv
        train_img_dir = ff_img_aug_dir
    
    else:
        if exp_setups[exp] == "only white":
            train_csv = ff_train_only_white_csv
        else:
            train_csv = ff_train_csv
        train_img_dir = ff_img_dir
    train_dataset = FairFaceDataset(csv_path=ff_label_dir + train_csv,
                                    img_dir=train_img_dir,
                                    transform=custom_transform)


# validation dataset
# val_dataset_celeb = CelebaDataset(csv_path=celeb_label_dir + celeb_val_csv,
#                             img_dir=celeb_img_dir,
#                             transform=custom_transform)

val_dataset_ff = FairFaceDataset(csv_path=ff_label_dir + ff_val_csv,
                                img_dir=ff_img_dir,
                                transform=custom_transform)



# test datasets
# test_dataset_celeb = CelebaDataset(csv_path=celeb_label_dir + celeb_test_csv,
#                             img_dir=celeb_img_dir,
#                             transform=custom_transform)

test_dataset_ff = FairFaceDataset(csv_path=ff_label_dir + ff_test_csv,
                                img_dir=ff_img_dir,
                                transform=custom_transform)


# create dataloaders on these datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=bs_train,
                          shuffle=True,
                          num_workers=num_workers)

val_loader = DataLoader(dataset=val_dataset_ff,
                          batch_size=bs_val,
                          shuffle=False,
                          num_workers=num_workers)

# test_loader_celeb = DataLoader(dataset=test_dataset_celeb,
#                           batch_size=bs_test,
#                           shuffle=False,
#                           num_workers=num_workers)

test_loader_ff = DataLoader(dataset=test_dataset_ff,
                          batch_size=bs_test,
                          shuffle=False,
                          num_workers=num_workers)


# In[14]:


# # Calculate the mean and standard deviation per channel
# channel_means = torch.zeros(3)
# channel_stds = torch.zeros(3)
# total_samples = 0

# for batch in train_loader:
#     images, _, _ = batch
#     batch_size = images.size(0)
#     total_samples += batch_size
#     channel_sums = torch.sum(torch.sum(images, dim=2), dim=2)
#     channel_means += torch.sum(channel_sums, dim=0)
#     channel_stds += torch.sum(torch.sum(torch.sum(images ** 2, dim=2), dim=2), dim=0)

# channel_means /= total_samples * images.size(2) * images.size(3)
# channel_stds /= total_samples * images.size(2) * images.size(3)
# # channel_stds = torch.sqrt(channel_stds - channel_means ** 2)

# print("Mean per channel:", channel_means.tolist())
# # print("Standard Deviation per channel:", channel_stds.tolist())


# In[15]:


# # Define the known means per channel
# channel_means = [0,0,0]

# # Calculate the standard deviation per channel
# channel_stds = torch.zeros(3)
# total_samples = 0

# for batch in train_loader:
#     images, _, _ = batch
#     batch_size = images.size(0)
#     total_samples += batch_size
#     channel_sums = torch.sum(torch.sum(images, dim=2), dim=2)
#     channel_stds += torch.sum(torch.sum(torch.sum((images - torch.Tensor(channel_means).reshape(1, 3, 1, 1)) ** 2, dim=2), dim=2), dim=0)

# channel_stds /= total_samples * images.size(2) * images.size(3)
# channel_stds = torch.sqrt(channel_stds)

# print("Standard Deviation per channel:", channel_stds.tolist())


# In[16]:


# pick model
if model_desc == "resnet18":
    model = resnet18(weights=(ResNet18_Weights.DEFAULT if pretrained else None))
elif model_desc == "resnet34":
    model = resnet34(weights=(ResNet34_Weights.DEFAULT if pretrained else None))
elif model_desc == "resnet50":
    model = resnet50(weights=(ResNet50_Weights.DEFAULT if pretrained else None))
elif model_desc == "densenet121":
    model = densenet121(weights=(DenseNet121_Weights.DEFAULT if pretrained else None))
elif model_desc == "densenet169":
    model = densenet169(weights=(DenseNet169_Weights.DEFAULT if pretrained else None))
elif model_desc == "mobilenetV3S":
    model = mobilenet_v3_small(weights=(MobileNet_V3_Small_Weights.DEFAULT if pretrained else None))
elif model_desc == "mobilenetV3L":
    model = mobilenet_v3_large(weights=(MobileNet_V3_Large_Weights.DEFAULT if pretrained else None))
else:
    raise Exception("Model not found!")


# adapt model output
if model_desc.startswith("resnet"):
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 2))
    
elif model_desc.startswith("densenet"):
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 2))
elif model_desc.startswith("mobilenet"):
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 2))   
    
else:
    raise Exception("Model not found!")

model.to(device)

# load checkpoint
if load_ckpt:
    ckpt = torch.load(ckpt_load_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt = None
    print(f"Loaded checkpoint: {ckpt_load_path}")

# define loss and create optimizer
bin_ce = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable params:", total_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

do_attack = len(eps_attack) > 0
dataset_desc = "CelebA" if is_celeba else "FairFace"

# initialize experiment tracking
# run = neptune.init_run(
#     project="danielritter0508/aug-precise-pre-trained",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9uZXctdWkubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDI4OGY4ZC05YmUzLTRjNzAtOTE2NS0xNzU0NTkyMTJiYmUifQ==",
# )
run = neptune.init_run(
    project="MyMasterThesis/experiments",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDI4OGY4ZC05YmUzLTRjNzAtOTE2NS0xNzU0NTkyMTJiYmUifQ==",
)

run["params"] = {"dataset": dataset_desc, 
                "experiment": exp_setups[exp], 
                "model_desc": model_desc,
                "do_attack": do_attack, 
                "eps_min": smallest_eps,
                "eps_max": biggest_eps,
                "eps_steps": eps_steps,
                "learning_rate": lr, 
                "num_epochs": num_epochs, 
                "scheduler_step": scheduler_step_size, 
                "gamma": scheduler_gamma, 
                "pretrained": pretrained}

# init foolbox model if necessary
if do_attack:
    # batch_iter = iter(train_loader)
    # for _ in range(10):
    #     batch = next(batch_iter)
    #     # ds = da
    #     # print(ds)
    #     print(torch.mean(batch[0][:,0,:,:]), torch.mean(batch[0][:,1,:,:]), torch.mean(batch[0][:,2,:,:]))
    # normalization = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fool_model = PyTorchModel(model, bounds=(0, 1), device=device)#, preprocessing=normalization
    # attack = L2AdditiveGaussianNoiseAttack()
    attack = LinfFastGradientAttack()
    # attack = LinfDeepFoolAttack()
else:
    fool_model = None
    attack = None


# In[17]:


# visualize foolbox attack
def visualize_attack(images, imgs_mod, epsilons, img_index=0):
    
    # show original image
    print("\noriginal image:")
    image = images[img_index].cpu().numpy()
    image = np.moveaxis(image, 0, -1) * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()
    
    # show all noise steps
    for i, eps in enumerate(epsilons):
        print(f"noisy image (eps = {eps}):")
        image = imgs_mod[i][img_index].cpu().numpy()
        image = np.moveaxis(image, 0, -1) * 255
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.show()


# calculate the root of the mean of the squared differences between the values and their average
def get_RMSD(values):
    # calculate the average of the values
    avg = sum(values) / len(values)

    # calculate the sum of the squared deviations
    squared_deviations_sum = sum(((x - avg) / avg) ** 2 for x in values)
    
    # calculate the mean squared deviation and take the root of it
    rmsd = (squared_deviations_sum / len(values)) ** 0.5
    return rmsd


# calculate all necessary metrics
def evaluate_metrics(model, fool_model, data_loader, device, log_description, show_tqdm=False, decimals=4):
    
    total_robs = None
    robs = None

    # initialize all parameters with zeros
    correct_predictions, true_pos, true_neg, pos_preds, pos_targets, num_examples = np.zeros((6, len(races)))

    # for adversarial attacks
    failed_per_race = [torch.tensor([]).reshape((len(eps_attack), 0)) for _ in races]
    failed_total = torch.tensor([]).reshape((len(eps_attack), 0))

    # iterate through validation/test data and collect necessary basic data
    for _, (features, targets, gt_races) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating", disable=not show_tqdm):#

        images = features.to(device)
        labels = targets.to(device)
        
        # prepape annotated races for race-wise split afterwards
        gt_races = np.array([races.index(race) for race in gt_races])
        
        # do adversarial attack and collect results
        if fool_model is not None:
            # raw, clipped, is_adv
            _, _, fail = attack(model=fool_model, inputs=images, criterion=Misclassification(labels), epsilons=eps_attack)
            fail = fail.cpu()
            
            # append results race-wise and total
            for i in range(len(races)):
                fail_race = fail[:, gt_races==i]
                failed_per_race[i] = torch.cat([failed_per_race[i], fail_race], dim=-1)
            failed_total = torch.cat([failed_total, fail], dim=-1)
            # visualize current attack
            # visualize_attack(images, images_mod, eps_attack, img_index=0)
            # return
  
        # do forward pass without gradients
        with torch.set_grad_enabled(False):
            predictions = torch.argmax(model(features.to(device)), -1).cpu().numpy()

        targets = targets.numpy()
        

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

    # noise robustness (if attack was done)
    if fool_model is not None:
        attack_acc_total = 1 - failed_total.float().mean(axis=-1).numpy()
        attack_acc_races = np.array([1 - failed_per_race[i].float().mean(axis=-1).numpy() for i in range(len(races))])
        # total_robs = (attack_acc_total - 0.5) / (total_acc - 0.5)
        # total_robs[total_robs < 0] = 0
        # robs = np.array([(attack_acc_races[i] - 0.5) / (accs[i] - 0.5) for i in range(len(races))])
        # robs[robs < 0] = 0
        predef_total_accs_dn169 = [0.746, 0.752, 0.771]
        predef_accs_dn169 = [[0.641, 0.749, 0.799, 0.792, 0.706, 0.737, 0.789], 
                             [0.666, 0.794, 0.782, 0.805, 0.731, 0.753, 0.743], 
                             [0.721, 0.811, 0.798, 0.802, 0.757, 0.754, 0.760]] # EX 25 20 21 
        
        predef_total_accs_rn34 = [0.742, 0.726, 0.750]
        predef_accs_rn34 = [[0.601, 0.722, 0.774, 0.813, 0.703, 0.771, 0.800], 
                            [0.637, 0.747, 0.757, 0.764, 0.703, 0.738, 0.735], 
                            [0.670, 0.774, 0.776, 0.818, 0.704, 0.752, 0.763]] # AUG-PRE 545 547 548
        
        predef_total_accs_rn34_pretr = [0.904, 0.909, 0.919]
        predef_accs_rn34_pretr = [[0.828, 0.920, 0.929, 0.945, 0.889, 0.883, 0.934], 
                                  [0.853, 0.936, 0.923, 0.940, 0.909, 0.884, 0.921], 
                                  [0.894, 0.945, 0.935, 0.945, 0.900, 0.901, 0.918]] # EX 22 23 24
        total_robs = attack_acc_total / attack_acc_total[0]#total_acc # predef_total_accs_rn34_pretr[exp]#
        robs = attack_acc_races / np.expand_dims(attack_acc_races[:, 0], axis=1) # accs # predef_accs_rn34_pretr[exp]


    # accuracy
    total_acc = correct_predictions.sum() / num_examples.sum()
    accs = correct_predictions / (num_examples + zero)
    # print(total_robs.shape, total_robs)
    # print(robs.shape, robs)
    # total_acc = total_robs[0]
    # accs = robs[:, 0]
    
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


    # round and track race-independent results
    total_acc, mad, rmsd_acc, total_prec, total_rec, total_f1 = np.round((total_acc, mad, rmsd_acc, total_prec, total_rec, total_f1), decimals)
    metrics = {"total_acc": total_acc, "MAD": mad, "RMSD": rmsd_acc, "total_prec": total_prec, 
               "total_rec": total_rec, "total_f1": total_f1}
    
    for (name, value) in metrics.items():
        run[log_description + "/" + name].append(value)
            
    if fool_model is not None:
        total_robs = np.round(total_robs, decimals)
        for eps, rob in zip(eps_attack, total_robs):
            run[f"{log_description}/total_rob"].append(rob) #/{eps}


    # round and track race-specific results
    accs, precs, recs, f1_scores = np.round((accs, precs, recs, f1_scores), decimals)
    
    for race, acc, prec, rec, f1  in zip(races, accs, precs, recs, f1_scores):
        run[log_description + "/acc/" + race].append(acc)
        run[log_description + "/prec/" + race].append(prec)
        run[log_description + "/rec/" + race].append(rec)
        run[log_description + "/f1/" + race].append(f1)
    
    if fool_model is not None:
        robs = np.round(robs, decimals)
        for robs_race, race in zip(robs, races):
            for eps, rob in zip(eps_attack, robs_race):
                run[f"{log_description}/rob/{race}"].append(rob) #{eps}/
            
            
    return total_acc, accs, mad, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs, total_robs, robs



def get_elapsed_time(start_time):
    elapsed = int(time.time() - start_time)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


# In[18]:


# Training loop
if not load_ckpt:
    start_time = time.time()

    rtpt = RTPT('DR', 'Train_Gender_Classifier', num_epochs)
    rtpt.start()

    print("\nInitiating experiment...")
    print(f"setup:\t\t{exp_setups[exp]}")
    print(f"model:\t\t{model_desc}")
    print(f"dataset:\t{dataset_desc}")
    print(f"lr:\t\t{lr}")
    print(f"trainset size:\t{size_train}")
    print(f"pretraining:\t{pretrained}")
    if do_attack:
        print(f"attacks:\t{eps_attack}")
    print(f"device:\t\t{device}\n")


    for epoch in range(num_epochs):
        
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {(epoch+1):03d}/{num_epochs:03d}")
        for batch_idx, (features, targets, _) in pbar:
            
            features = features.to(device)
            targets = targets.to(device)
            
                
            # forward and backward pass
            logits = model(features)
            targets_exp = F.one_hot(targets).float()
            loss = bin_ce(logits, targets_exp)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            run["train/loss"].append(loss.item())

            optimizer.zero_grad()
            
            loss.backward()
            
            # update model params 
            optimizer.step()
            
            
            # if batch_idx == 3:
            #     break
            # first = features[0].cpu().numpy()
            # first = np.moveaxis(first, 0, -1) * 255
            # first = first.astype(np.uint8)
            # image = Image.fromarray(first)
            # print("Woman" if int(targets[0].item()) == 0 else "Man")
            # image.show()
        
                # print("Gender:", "Man" if targets)

        if (epoch+1) in save_ckpt_after_epochs:
            ckpt_save_path = ckpt_dir + run["sys/id"].fetch() + "_" + str(epoch+1) + "_epochs.pt"
            
            torch.save({
                "model_state_dict": model.state_dict()
            }, ckpt_save_path)
            print(f"Saved checkpoint at: {ckpt_save_path}")
            
        model.eval()
        # with torch.set_grad_enabled(False): # save memory during inference
        evaluate_metrics(model, fool_model, val_loader, device, "valid", True)

        scheduler.step()
        rtpt.step()
        
    print(f"Total Training Time: {get_elapsed_time(start_time)}")   


# In[19]:


# evaluate experiment on test sets
# with torch.set_grad_enabled(False): # save memory during inference
print(f"\nEvaluation of experiment: '{exp_setups[exp]}'\n")

# evaluation CelebA
# total_acc, accs, MAD, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs = evaluate_metrics(model, fool_model, test_loader_celeb, device, "eval_celeb", show_tqdm=True)
# print("\nEvaluation CelebA test set:")
# print(f"Total accuracy: {total_acc:.2%}\t| Accuracies:\t{accs}")
# print(f"Maximum accuracy disparity: {MAD}")
# print(f"Total precision: {total_prec:.2%}\t| Precisions:\t{precs}")
# print(f"Total recall: {total_rec:.2%}\t| Recalls:\t{recs}\n")

# evaluation FairFace
model.eval()
total_acc, accs, MAD, rmsd_acc, total_f1, f1_scores, total_prec, precs, total_rec, recs, total_robs, robs = evaluate_metrics(model, fool_model, test_loader_ff, device, "valid", show_tqdm=True)
print("\nEvaluation FairFace test set:")
print(f"Total accuracy: {total_acc:.2%}\t| Accuracies:\t{accs}")
print(f"Maximum accuracy disparity: {MAD}")
print(f"Total precision: {total_prec:.2%}\t| Precisions:\t{precs}")
print(f"Total recall: {total_rec:.2%}\t| Recalls:\t{recs}\n")

# print(f"Total robustnesses:{total_robs}")
# print(f"Robustnesses:{robs}")


# In[ ]:


# # print(robs)
# robustness_dir = "robustness_np_arrays"
# file_path_total = f"{robustness_dir}/total_rob_{exp_num}_{smallest_eps}_{biggest_eps:.1e}_{eps_steps}_steps.npy"
# file_path_races = f"{robustness_dir}/robs_{exp_num}_{smallest_eps}_{biggest_eps:.1e}_{eps_steps}_steps.npy"
# np.save(file_path_total, total_robs)
# np.save(file_path_races, robs)
# # print(np.load(file_path))

