"""
------------------------------------------------------------
                        CELEBA
------------------------------------------------------------

"""

"""

PROMPT SCHEME:
"A [blurry] photo of a(n) 

[[attractive], [bald], [smiling], [young]]

famous man|woman 

[with [bags under his|her eyes], [eyeglasses], [a goatee*], [gray hair], [heavy makeup], [his|her mouth slightly open], [a mustache*], 
[a beard*], [a receding hairline] (and) [rosy cheeks]]

[wearing [earrings], [a hat], [lipstick], [a necklace*] (and) [a necktie*]]"


* mind mutual exclusion
"""


import pandas as pd


# build a prompt from a pandas row (see scheme above)
def build_celeba_prompt(current):

    # build prefix
    is_blurry = " blurry" if current["Blurry"] == 1 else ""
    prefix = f"a{is_blurry} photo of a"


    # build list of adjectives
    attractive = "n attractive," if current["Attractive"] == 1 else ""
    bald = " bald," if current["Bald"] == 1 else ""
    smiling = " smiling," if current["Smiling"] == 1 else ""
    young = " young," if current["Young"] == 1 else ""
    adjectives = attractive + bald + smiling + young


    # get gender
    gender = " man" if current["Male"] == 1 else " woman"


    # build list of features
    pronoun = "his" if current["Male"] == 1 else "her"
    features = []
    if current["Bags_Under_Eyes"] == 1:
        features.append(f"bags under {pronoun} eyes")

    if current["Eyeglasses"] == 1:
        features.append("glasses")

    if current["Gray_Hair"] == 1:
        features.append("gray hair")

    if current["Heavy_Makeup"] == 1:
        features.append("heavy makeup")

    if current["Mouth_Slightly_Open"] == 1:
        features.append(f"{pronoun} mouth slightly open")

    if current["Receding_Hairline"] == 1:
        features.append("a receding hairline")

    if current["Rosy_Cheeks"] == 1:
        features.append("rosy cheeks")

    # don't allow multiple versions of beards
    if current["No_Beard"] == -1:
        if current["Goatee"] == 1:
            features.append("a goatee")
        elif current["Mustache"] == 1:
            features.append("a mustache")
        else:
            features.append("a beard")

    # connect all features to one string
    feat_str = ""
    for i, wear in enumerate(features):
        if i < len(features) - 2:
            feat_str += wear + ", "
        elif i == len(features) - 2:
            feat_str += wear + " and "
        else:
            feat_str += wear

    # add 'with' in front of the feature string
    if len(features) > 0:
        feat_str = " with " + feat_str
        

    # build list of wearables
    wearables = []
    if current["Wearing_Earrings"] == 1:
        wearables.append(f"earrings")

    if current["Wearing_Hat"] == 1:
        wearables.append("a hat")

    if current["Wearing_Lipstick"] == 1:
        wearables.append("lipstick")
    
    if current["Wearing_Necktie"] == 1:
        wearables.append("a necktie")

    elif current["Wearing_Necklace"] == 1:
        wearables.append("a necklace")
    
    # connect all wearables to one string
    wear_str = ""
    for i, wear in enumerate(wearables):
        if i < len(wearables) - 2:
            wear_str += wear + ", "
        elif i == len(wearables) - 2:
            wear_str += wear + " and "
        else:
            wear_str += wear

    # add 'wearing' in front of the features
    if len(wearables) > 0:
        wear_str = " wearing " + wear_str

    # connect all parts of the prompt to the final prompt
    prompt = prefix + adjectives + " famous" + gender + feat_str + wear_str
    
    return prompt


# build very precise prompts for the inserted image_numbers based on the attributes available for CelebA images
# returns dict: (image_number: prompt)
def get_precise_celeba_prompts(image_numbers):
    attr_path = "datasets/celeba/list_attr_celeba.txt"
    attributes = pd.read_csv(attr_path, sep="\s+", skiprows=1)[[
    "Attractive", "Bags_Under_Eyes", "Bald", "Blurry", "Eyeglasses",
    "Goatee", "Gray_Hair", "Heavy_Makeup", "Male", "Mouth_Slightly_Open",
    "Mustache", "No_Beard", "Receding_Hairline", "Rosy_Cheeks", "Smiling",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", 
    "Wearing_Necktie", "Young"
    ]]
    
    prompts = {}
    for img_nmb in image_numbers:
        current = attributes.iloc[img_nmb-1]
        prompts[img_nmb] = build_celeba_prompt(current)
    
    return prompts







"""
------------------------------------------------------------
                        FAIRFACE
------------------------------------------------------------

"""


""" 
    PROMPT SCHEME:
    "A photo of a(n) [RACE:1] [GENDER & AGE] [RACE:2]

    
    GENDER & AGE:
    baby girl/boy
    girl/boy
    teenage girl/boy
    young woman/man
    woman/man
    woman/man in her/his fourties/fifties/sixties
    elderly woman/man

    
    RACE:
    either a description standing before or after GENDER & AGE (concret description: see below)

"""

import pandas as pd
import numpy as np
from numpy.random import randint

races = ["Black", "Indian", "Latino_Hispanic", "Middle Eastern", "Southeast Asian", "East Asian", "White"]

black_desc = ["Black", "African American", "Northern African", "Eastern African", 
              "from Southern Africa", "from Western Africa", "from Central Africa"]

indian_desc = ["Indian", "Pakistani", "Northern Indian", "Eastern Indian",
               "from Southern India", "from Western India", "from Central India"]

lat_desc = ["Latino", "Hispanic", "Mexican", "Latin American",
            "from Brazil", "from South America", "from Central America"]

me_desc = ["Middle Eastern", "Arab", "Iranian", "Turkish",
           "from the Middle East", "from the Arabian Peninsula", "from Egypt"]

sea_desc = ["Southeast Asian", "Indonesian", "Vietnamese", "Filipino",
            "from Mainland Southeast Asia", "from Maritime Southeast Asia", "from Indonesia"]

easian_desc = ["East Asian", "Chinese", "Korean", "Han Chinese",
               "from East Asia", "from China", "from Japan"]

white_desc = ["White", "European American", "European Australian", "Northern European", 
              "from Eastern Europe", "from Southern Europe", "from Western Europe"]


race_descs = [black_desc, indian_desc, lat_desc, me_desc, sea_desc, easian_desc, white_desc]
race_desc_pos_sep = 4


age_steps = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
male_desc = ["baby boy", "boy", "teenage boy", "young man", "man", "man in his fourties", "man in his fifties", "man in his sixties", "elderly man"]
female_desc = ["baby girl", "girl", "teenage girl", "young woman", "woman", "woman in her fourties", "woman in her fifties", "woman in her sixties", "elderly woman"]


# build a FairFace prompt from age, gender and race index
def build_ff_prompt(age, gender, race, random_race_desc):

    # build prefix
    prefix = f"a photo of a"


    # get gender & age descriptor
    if gender == "Male":
        gender_age_desc = male_desc[age_steps.index(age)]
    else:
        gender_age_desc = female_desc[age_steps.index(age)]

    race_index = races.index(race)

    # pick race descriptor
    if random_race_desc:
        rand = randint(len(race_descs[0]))
    else:
        rand = 0

    race_desc = race_descs[race_index][rand]

    # put race descriptor at the correct position in the sentence and finish the prompt
    if rand < race_desc_pos_sep:
        filler = "n " if race_desc[0] in 'aeiouAEIOU' else " "
        prompt = prefix + filler + race_desc + " " + gender_age_desc
    else:
        filler = "n " if gender_age_desc[0] in 'aeiouAEIOU' else " "
        prompt = prefix + filler + gender_age_desc + " " + race_desc 
    
    return prompt


# build diverse prompts for the inserted image_numbers based on the attributes available for FairFace images
# returns dict: (image_number: prompt)
def get_ff_prompts(image_numbers, race_index=None, random_race_desc=True):
    attr_path = "fairface/dataset/labels/fairface_label_train.csv"
    attributes = pd.read_csv(attr_path)
    prompts = {}
    for img_nmb in image_numbers:
        current = attributes.iloc[img_nmb-1]
        current_race = races[race_index] if race_index is not None else current["Race"]
        prompts[img_nmb] = build_ff_prompt(current["Age"], current["Gender"], current_race, random_race_desc)
    return prompts