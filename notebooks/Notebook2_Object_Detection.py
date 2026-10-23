# %% [markdown]
# Source notebook: Notebook2_Object_Detection.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 3: Object Detection

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %% [markdown]
# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
#     ```
#     !pip install -q comet_ml transformers ultralytics torch
#     ```

# %% [markdown]
# ### Set up Comet

# %%
# Notebook cell 5
import comet_ml

# %% [markdown]
# Info about ['Comet'](https://www.comet.com/site/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L3)

# %%
# Notebook cell 7
comet_ml.login(anonymous=True, project_name="3: OWL-ViT + SAM")

# %%
# Notebook cell 8
exp = comet_ml.Experiment()

# %% [markdown]
# ### Load the image

# %%
# Notebook cell 10
# To display the image
from PIL import Image

# %%
# Notebook cell 11
logged_artifact = exp.get_artifact("L3-data", "anmorgan24")

# %% [markdown]
# >Note: the images referenced in this notebook have already been uploaded to the Jupyter directory, in this classroom, for your convenience. For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 13
local_artifact = logged_artifact.download("./")

# %%
# Notebook cell 14
# Display the images
raw_image = Image.open("L3_data/dogs.jpg")
raw_image

# %% [markdown]
# ### Get bounding boxes with OWL-ViT object detection model

# %% [markdown]
# >Note: `pipeline` is already installed for you in this classroom.

# %%
# Notebook cell 17
from transformers import pipeline

# %%
# Notebook cell 18
OWL_checkpoint = "./models/google/owlvit-base-patch32"

# %% [markdown]
# Info about ['google/owlvit-base-patch32'](https://huggingface.co/google/owlvit-base-patch32)

# %% [markdown]
# * Build the pipeline for the detector model.

# %%
# Notebook cell 21
# Load the model
detector = pipeline(
    model= OWL_checkpoint,
    task="zero-shot-object-detection"
)

# %%
# Notebook cell 22
# What you want to identify in the image
text_prompt = "dog"

# %%
# Notebook cell 23
output = detector(
    raw_image,
    candidate_labels = [text_prompt]
)

# %%
# Notebook cell 24
# Print the output to identify the bounding boxes detected
output

# %% [markdown]
# * Use the **util**'s function to prompt boxes in top of the image.

# %% [markdown]
# >Note: ```utils``` is an additional file containing the methods that have been already developed for you to be used in this classroom. 
# For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 27
from utils import preprocess_outputs

# %%
# Notebook cell 28
input_scores, input_labels, input_boxes = preprocess_outputs(output)

# %%
# Notebook cell 29
from utils import show_boxes_and_labels_on_image

# %%
# Notebook cell 30
# Show the image with the bounding boxes
show_boxes_and_labels_on_image(
    raw_image,
    input_boxes[0],
    input_labels,
    input_scores
)

# %% [markdown]
# ### Get segmentation masks using Mobile SAM

# %%
# Notebook cell 32
# Load the SAM model from the imported ultralytics library
from ultralytics import SAM

# %%
# Notebook cell 33
SAM_version = "mobile_sam.pt"

# %% [markdown]
# Info about [mobile_sam.pt](https://docs.ultralytics.com/models/mobile-sam/)

# %%
# Notebook cell 35
model = SAM(SAM_version)

# %% [markdown]
# * Generate an array using numpy.

# %%
# Notebook cell 37
import numpy as np

# %%
# Notebook cell 38
# Create a list of positive labels of same length as the number of predictions generated above
labels = np.repeat(1, len(output))

# %%
# Notebook cell 39
# Print the labels
labels

# %%
# Notebook cell 40
result = model.predict(
    raw_image,
    bboxes=input_boxes[0],
    labels=labels
)

# %%
# Notebook cell 41
result

# %%
# Notebook cell 42
masks = result[0].masks.data
masks

# %%
# Notebook cell 43
from utils import show_masks_on_image

# %%
# Notebook cell 44
# Visualize the masks
show_masks_on_image(
    raw_image,
    masks
)

# %% [markdown]
# >Note: Please note that the results obtained from running this notebook may vary slightly from those demonstrated by the instructor in the video. 

# %% [markdown]
# ### Image Editing: blur out faces

# %% [markdown]
# * Load the image.

# %%
# Notebook cell 48
from PIL import Image

# %%
# Notebook cell 49
image_path = "L3_data/people.jpg"

# %% [markdown]
# >Note: the images referenced in this notebook have already been uploaded to the Jupyter directory, in this classroom, for your convenience. For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 51
raw_image = Image.open(image_path)
raw_image

# %%
# Notebook cell 52
raw_image.size

# %% [markdown]
# * Resize the image.

# %%
# Notebook cell 54
# Calculate width percent to maintain aspect ratio in resize transformation
mywidth = 600
wpercent = mywidth / float(raw_image.size[0])
wpercent

# %%
# Notebook cell 55
# Calculate height percent to maintain aspect ratio in resize transformation
hsize = int( float(raw_image.size[1]) * wpercent )
hsize

# %%
# Notebook cell 56
# Resize the image
raw_image = raw_image.resize([mywidth, hsize])
raw_image

# %%
# Notebook cell 57
raw_image.size

# %%
# Notebook cell 58
# Save the resized image
image_path_resized = "people_resized.jpg"
raw_image.save(image_path_resized)

# %% [markdown]
# ### Detect faces

# %%
# Notebook cell 60
candidate_labels = ["human face"]

# %%
# Notebook cell 61
# Define a new Comet experiment for this new pipeline
exp = comet_ml.Experiment()

# %%
# Notebook cell 62
# Log raw image to the experiment
_ = exp.log_image(
    raw_image,
    name = "Raw image"
)

# %% [markdown]
# * Create bounding boxes with OWL-ViT.

# %%
# Notebook cell 64
# Apply detector model to the raw image
output = detector(
    raw_image,
    candidate_labels=candidate_labels
)

# %%
# Notebook cell 65
input_scores, input_labels, input_boxes = preprocess_outputs(output)

# %%
# Notebook cell 66
# Print values of the bounding box coordinates identified
input_boxes

# %% [markdown]
# #### Log the images and bounding boxes.

# %%
# Notebook cell 68
metadata = {
    "OWL prompt": candidate_labels,
    "SAM version": SAM_version,
    "OWL Version": OWL_checkpoint
}

# %%
# Notebook cell 69
from utils import make_bbox_annots

# %%
# Notebook cell 70
annotations = make_bbox_annots(
    input_scores,
    input_labels,
    input_boxes,
    metadata
)

# %%
# Notebook cell 71
_ = exp.log_image(
    raw_image,
    annotations= annotations,
    metadata=metadata,
    name= "OWL output"
)

# %% [markdown]
# ### Segmentation masks using SAM

# %%
# Notebook cell 73
result = model.predict(
    image_path_resized,
    bboxes=input_boxes[0],
    labels=np.repeat(1, len(input_boxes[0]))
)

# %% [markdown]
# ### Blur entire image first

# %%
# Notebook cell 75
from PIL.ImageFilter import GaussianBlur

# %%
# Notebook cell 76
blurred_img = raw_image.filter(GaussianBlur(radius=5))

# %%
# Notebook cell 77
blurred_img

# %%
# Notebook cell 78
masks = result[0].masks.data.cpu().numpy()

# %%
# Notebook cell 79
# Create an array of zeroes of the same shape as our image mask
total_mask = np.zeros(masks[0].shape)

# %%
# Notebook cell 80
# Add each output mask to the total_mask
for mask in masks:
    total_mask = np.add(total_mask,mask)

# %%
# Notebook cell 81
# Where there is any value other than zero (where any masks exist), show the blurred image
# Else, show the original (unblurred) image
output = np.where(
    np.expand_dims(total_mask != 0, axis=2),
    blurred_img,
    raw_image
)

# %%
# Notebook cell 82
import matplotlib.pyplot as plt

# %%
# Notebook cell 83
# Print image with faces blurred
plt.imshow(output)

# %% [markdown]
# * Log this image in the **Comet** platform.

# %%
# Notebook cell 85
metadata = {
    "OWL prompt": candidate_labels,
    "SAM version": SAM_version,
    "OWL version": OWL_checkpoint
}

# %%
# Notebook cell 86
_ = exp.log_image(
    output,
    name="Blurred masks",
    metadata = metadata,
    annotations=None
)

# %% [markdown]
# ### Blur just faces of those not wearing sunglasses

# %%
# Notebook cell 88
# New label
candidate_labels = ["a person without sunglasses"]

# %% [markdown]
# * Re-run the pipeline.

# %%
# Notebook cell 90
exp = comet_ml.Experiment()

# %%
# Notebook cell 91
_ = exp.log_image(raw_image, name="Raw image")

# %%
# Notebook cell 92
output = detector(raw_image, candidate_labels=candidate_labels)

# %%
# Notebook cell 93
input_scores, input_labels, input_boxes = preprocess_outputs(output)

# %%
# Notebook cell 94
# Print the bounding box coordinates
input_boxes

# %% [markdown]
# * Explore what is happening in the **Comet** platform.

# %%
# Notebook cell 96
from utils import make_bbox_annots

# %%
# Notebook cell 97
metadata = {
    "OWL prompt": candidate_labels,
    "SAM version": SAM_version,
    "OWL version": OWL_checkpoint,
}

# %%
# Notebook cell 98
annotations = make_bbox_annots(
    input_scores,
    input_labels,
    input_boxes,
    metadata
)

# %%
# Notebook cell 99
_ = exp.log_image(
    raw_image,
    annotations=annotations,
    metadata=metadata,
    name="OWL output no sunglasses"
)

# %%
# Notebook cell 100
result = model.predict(
    image_path_resized,
    bboxes=input_boxes[0],
    labels=np.repeat(1, len(input_boxes[0]))
)

# %%
# Notebook cell 101
from PIL.ImageFilter import GaussianBlur
blurred_img = raw_image.filter(GaussianBlur(radius=5))

# %%
# Notebook cell 102
masks = result[0].masks.data.cpu().numpy()

total_mask = np.zeros(masks[0].shape)
for mask in masks:
    total_mask = np.add(total_mask, mask)

# %%
# Notebook cell 103
# Print the result
output = np.where(
    np.expand_dims(total_mask != 0, axis=2),
    blurred_img,
    raw_image
)
plt.imshow(output)

# %% [markdown]
# * Analyze results in the **Comet** platform.

# %%
# Notebook cell 105
metadata = {
    "OWL prompt": candidate_labels,
    "SAM version": SAM_version,
    "OWL version": OWL_checkpoint,
}

# %%
# Notebook cell 106
_ = exp.log_image(
    output,
    name="Blurred masks no sunglasses",
    metadata=metadata,
    annotations=None
)

# %% [markdown]
# ### Try yourself! 
# Try the image editing with the following images.

# %% [markdown]
# >Note: the images referenced in this notebook have already been uploaded to the Jupyter directory, in this classroom, for your convenience. For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 109
cafe_img = Image.open("L3_data/cafe.jpg")
cafe_img

# %%
# Notebook cell 110
crosswalk_img = Image.open("L3_data/crosswalk.jpg")
crosswalk_img

# %%
# Notebook cell 111
metro_img = Image.open("L3_data/metro.jpg")
metro_img

# %%
# Notebook cell 112
friends_img = Image.open("L3_data/friends.jpg")
friends_img

# %% [markdown]
# ### Additional Resources
# 
# * For more on how to use [Comet](https://www.comet.com/site/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L3) for experiment tracking, check out this [Quickstart Guide](https://colab.research.google.com/drive/1jj9BgsFApkqnpPMLCHSDH-5MoL_bjvYq?usp=sharing) and the [Comet Docs](https://www.comet.com/docs/v2/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L3).
# * This course was based off a set of two blog articles from Comet. Explore them here for more on how to use newer versions of Stable Diffusion in this pipeline, additional tricks to improve your inpainting results, and a breakdown of the pipeline architecture:
#   * [SAM + Stable Diffusion for Text-to-Image Inpainting](https://www.comet.com/site/blog/sam-stable-diffusion-for-text-to-image-inpainting/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L3)
#   * [Image Inpainting for SDXL 1.0 Base Model + Refiner](https://www.comet.com/site/blog/image-inpainting-for-sdxl-1-0-base-refiner/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L3)
