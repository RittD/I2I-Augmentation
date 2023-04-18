# PROMPT SCHEME:
# "A [blurry] photo of a(n) 

# [[attractive], [bald], [smiling], [young]]

# famous man|woman 

# [with [bags under his|her eyes], [eyeglasses], [a goatee*], [gray hair], [heavy makeup], [his|her mouth slightly open], [a mustache*], 
# [a beard*], [a receding hairline] (and) [rosy cheeks]]

# [wearing [earrings], [a hat], [lipstick], [a necklace*] (and) [a necktie*]]"


# * mind mutual exclusion



import pandas as pd


# build a prompt from a pandas row (see scheme above)
def build_prompt(current):

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
        prompts[img_nmb] = build_prompt(current)
    
    return prompts