#!/usr/bin/env python
# coding: utf-8

# ## Copyright 2022 Google LLC. Double-click for license information.

# In[1]:


# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Editing with Prompt-to-Prompt

# In[1]:


import os
# os.system("pip install pandas")
from typing import Optional, Union, Tuple, List, Dict
from tqdm import tqdm
import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from PIL import Image
from huggingface_hub import login
from matplotlib import pyplot as plt
from PIL import Image
import re
from rtpt import RTPT
import pandas as pd

from prompt_engineering import get_ff_prompts


# In[2]:


def get_image_numbers_from_inv_dir(directory, size):

    file_list = os.listdir(directory)  # Get a list of all filenames in the directory
    image_numbers = []

    for i, filename in enumerate(file_list):
        if i == size:
            break
        
        image_number = os.path.splitext(filename)[0]  # Extract the part before the dot
        image_number = int(image_number)  # Parse it to an integer
        image_numbers.append(image_number)

    return image_numbers


# In[3]:


# pick dataset
use_celeba = False #True

if use_celeba:
    input_dir = "CelebA/cropped"
    inversion_dir = "CelebA/latents_precise_prompts"
    solo_saving_dir = "CelebA/aug_precise_prompts_strong/notatonce" 
else:
    input_dir = "fairface/dataset/fairface-img-margin025-trainval/train"
    inversion_dir = "fairface/dataset/latents/only_white" 
    solo_saving_dir = "fairface/dataset/augmentations/mixed_caps_total_70k"

use_random_caps = True
all_at_once = False

first_image = 8000
last_image = 10000
# image_numbers = [256, 298, 300, 326, 332]
# image_numbers = [336, 522, 525, 531, 537]
image_numbers = get_image_numbers_from_inv_dir(inversion_dir, 10000) [first_image : last_image]
# image_numbers = [10092, 10293, 10300, 10436, 10520, 10186, 10665, 10860, 10881, 10904, 10962]#[i for i in range(first_image, last_image+1)]
# image_numbers = []
# print(len(image_numbers), image_numbers)

# show_race =  [True,      True,       True,       True,              False, False, False   ]
show_race =  [True,      True,       True,       True,              True,               True,           True]
# show_race =  [False, False, True, False,   False,      False,       False]

# show_race =  [True, True, True, False,   False,      False,       False]
# show_race =  [False, False, True, True,   False,      False,       False]
# show_race =  [False, False, False, True,   True,      True,       True]
# show_race =  [False, False, False, True,   True,      True,       True]

races =      ["black",   "indian",   "latino",   "middle eastern",   "southeast asian",  "east asian",   "white" ]

debug_mode = False

save_whole = False
whole_saving_dir = None #"outputs/CelebA_cropped"

save_solo = True

device = "cuda:12"


# For loading the Stable Diffusion using Diffusers, follow the instuctions https://huggingface.co/blog/stable_diffusion and update MY_TOKEN with your token.

# In[5]:


scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# get your account token from https://huggingface.co/settings/tokens
# MY_TOKEN = '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' # read
# login(token=MY_TOKEN)

LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

model_name = "/workspaces/PromptToPrompt/StableDiffusion/src/models/stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler).to(device) 
# stabilityai/stable-diffusion # runwayml/stable-diffusion-v1-5 # CompVis/stable-diffusion-v1-4, use_auth_token=MY_TOKEN

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


# ## Prompt-to-Prompt code

# In[ ]:


class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


# ## Inference Code

# In[ ]:


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], disable=not debug_mode)):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t


# In[ ]:


# auxillary methods:

def save_images(img_number, images, races, prompts, self_replace_steps, 
                save_whole, whole_saving_dir, save_solo, solo_saving_dir, debug_mode=True, img_size=None):
    # prompts_lengths = [len(prompt.split(" ")) for prompt in prompts]
    prompts_are_of_equal_length = False # max(prompts_lengths) == min(prompts_lengths)
    is_equal = "_eq" if prompts_are_of_equal_length else ""
    
    if save_whole:
        # save all images as a whole
        num_images = len(prompts)
        fig = plt.figure(figsize=(4 * num_images, 4))
        columns = num_images
        rows = 1
        for i in range(1, columns*rows+1):
            img = images[i-1]
            ax = fig.add_subplot(rows, columns, i)
            plt.imshow(img)

            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(5)
                if i == 1:
                    ax.spines[axis].set_color("red")
                else:
                    ax.spines[axis].set_color("white")

            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft = False) 

        plt.tight_layout()

        srs_str = str(self_replace_steps).replace(".", "_")

        whole_save_name = f'{whole_saving_dir}/img_{img_number}_{srs_str}{is_equal}.png'
        fig.savefig(whole_save_name)
        if debug_mode:
            print(f"Saved images as a whole at '{whole_save_name}'")
        plt.close()


    if save_solo:
        # save images separately
        if not os.path.exists(solo_saving_dir):
            os.makedirs(solo_saving_dir)

        for i, image_array in enumerate(images):
            if i == 0:
                continue
            img = Image.fromarray(image_array)
            if img_size is not None:
                img.thumbnail(img_size, Image.LANCZOS)
            race, race_index = races[i-1]
            race_desc = "_gt" if i == 0 else f"_{race_index}_{race}"
            img.save(f"{solo_saving_dir}/{img_number}{race_desc}{is_equal}.jpg")
        
        if debug_mode:
            print(f"Saved images separately at '{solo_saving_dir}'")
    
    if debug_mode and (save_whole or save_solo):
        print("\n")


# In[ ]:


print("Starting augmentation...")
print(f"image interval: {(image_numbers[0], image_numbers[-1])}")
print(f"races used: {list(np.delete(races, ~np.array(show_race)))}")
print(f"device in use: {device}\n")

rtpt = RTPT('DR', 'P2P_Dataset_Augmentation', len(image_numbers))
rtpt.start()

# set parameters for all images 
picked_races = [(race, i)  for i, race in enumerate(races) if show_race[i]]
words = []
for (race, _) in picked_races:
    words = words + race.split(" ")
words = list(dict.fromkeys(words))

eq_params = None #{"words": (words), "values": (5,)} # amplify attention to the word "tiger" by *2 
cross_replace_steps = {'default_': .8,}
self_replace_steps = 0.5
blend_word = None #((('black',), ("black",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
prompts_are_of_equal_length = False


# iterate of all image numbers
for img_nmb in tqdm(image_numbers):
    img_number = f"{img_nmb:06}" if use_celeba else str(img_nmb)
    inv_file = img_number + ".pt"

    # load latent vectors, embeddings and other information
    data = torch.load(os.path.join(inversion_dir, inv_file), map_location=device)
    
    
    # create prompts for image creation
    original_prompt = data["prompt"]
    # print()
    # print(original_prompt)    
    

    new_prompts = []
    
    for (race, index) in picked_races:
        if use_celeba:
            new_prompt = original_prompt.replace("famous", f"famous, {race}")
        
        else:
            new_prompt = get_ff_prompts([img_nmb], race_index=index, random_race_desc=use_random_caps)[img_nmb]
        
        # print(new_prompt)
        new_prompts.append(new_prompt)
    prompts = [original_prompt] + new_prompts


    # create images
   
    # if debug_mode:
    #     print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
    #     images, latent = run_and_display(prompts, controller, run_baseline=False, latent=data["latents"], uncond_embeddings=data["uncond_embeddings"], verbose=debug_mode)

    #     show_cross_attention(controller, 16, ["up", "down"])
    #     print("\n")

    # create all images at the same time
    if all_at_once:
        controller = make_controller(prompts, prompts_are_of_equal_length, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        images, _ = run_and_display(prompts, 
                                    controller, 
                                    run_baseline=False, 
                                    latent=data["latents"], 
                                    uncond_embeddings=data["uncond_embeddings"],
                                    verbose=debug_mode)
        
    # create the images one by one
    else:
        images = []
        for i, new_p in enumerate(new_prompts):
            prompts_tmp = [original_prompt, new_p]
            controller = make_controller(prompts_tmp, prompts_are_of_equal_length, cross_replace_steps, self_replace_steps, blend_word, eq_params)
            image, _ = run_and_display(prompts_tmp, 
                                    controller, 
                                    run_baseline=False, 
                                    latent=data["latents"], 
                                    uncond_embeddings=data["uncond_embeddings"],
                                    verbose=debug_mode)
            # add inverted image once
            if i == 0:
                images.append(image[0])

            # add augmented images
            images.append(image[1])



    # save images as a whole in one folder and separately in another
    img_size = (448,448)
    save_images(img_number, images, picked_races, prompts, self_replace_steps, 
                save_whole, whole_saving_dir, save_solo, solo_saving_dir, debug_mode=debug_mode, img_size=img_size)
    rtpt.step()

