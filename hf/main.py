from datasets import load_dataset
from transformers import pipeline

from transformers import pipeline
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Load the dataset
dataset = load_dataset("biglam/european_art", "coco")
# dataset = load_dataset("laion/laion-art")

# Access the image and label data
# images = dataset['train']['image']
# labels = dataset['train']['label']

# https://www.sothebys.com/en/buy/auction/2019/style-silver-ceramics-furniture/8bb898a8-0e71-4b4e-93ea-419fc06007b4

# image_to_classify = "../samples/balon-dog.jpg"
# image_to_classify = "../samples/guitare-pablo-picasso.jpg"
# image_to_classify = "../samples/NU_003.jpg"
image_to_classify = "../samples/ours-Pompon.jpg"
# image_to_classify = "../samples/chair-antique-3.jpeg"
# image_to_classify = "../samples/sconce-1.jpeg"
# image_to_classify = "../samples/table-1.png"
# image_to_classify = "../samples/table-2.png"
# image_to_classify = "../samples/table-3.png"

# Other
# image_to_classify = "../samples/cat.jpeg"
# image_to_classify = "../samples/raccoon_6.jpg"



labels_for_classification =  [
    "Painting", 
    "Picture", 
    "Sculpture",
    "Chair",
    "Mirror",
    "Table",
    "Chandelier",
    "Sconce",
    "Other",
]
# labels_for_classification =  [
#     "Glass or crystal or mirror", 
#     "Marble or other stone", 
#     "Ceramic or terracotta",
#     "Concrete or plaster",
#     "My item does not contain any of the materials listed above"
# ]

##
# image-classification
##
model_name = "microsoft/resnet-50"
classifier = pipeline("image-classification", model = model_name)
scores = classifier(image_to_classify)
# print(("res", res))


##
# Zero-shot-image-classification
##

# Items typologies model
# model_name = "openai/clip-vit-large-patch14-336"
# model_name = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
# classifier = pipeline("zero-shot-image-classification", model = model_name)

# materials model
# KO
# extractor = AutoFeatureExtractor.from_pretrained("jasmine009/materials")
# model = AutoModelForImageClassification.from_pretrained("jasmine009/materials")
# classifier = pipeline("zero-shot-image-classification", model = model)

# scores = classifier(image_to_classify, 
#                     candidate_labels = labels_for_classification)
print(f"The highest score is {scores[0]['score']:.3f} for the label {scores[0]['label']}")
