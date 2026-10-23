# %% [markdown]
# Source notebook: Notebook1_Image_Segmentation.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 2: Image Segmentation

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %% [markdown]
# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
#     ```
#     !pip install ultralytics torch
#     ```

# %% [markdown]
# ### Load the sample image

# %%
# Notebook cell 5
from PIL import Image
raw_image = Image.open("dogs.jpg")
raw_image

# %% [markdown]
# >Note: the images referenced in this notebook have already been uploaded to the Jupyter directory, in this classroom, for your convenience. For further details, please refer to the **Appendix** section located at the end of the lessons.

# %% [markdown]
# * Resize the image.

# %%
# Notebook cell 8
from utils import resize_image
resized_image = resize_image(raw_image, input_size=1024)
resized_image

# %% [markdown]
# ### Import and prepare the model

# %%
# Notebook cell 10
import torch

# %%
# Notebook cell 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# Info about [torch](https://pytorch.org/).

# %%
# Notebook cell 13
from ultralytics import YOLO
model = YOLO('./models/comet-c1/FastSAM.pt')

# %% [markdown]
# Info about ['FastSAM'](https://docs.ultralytics.com/models/fast-sam/)

# %% [markdown]
# ### Use the model

# %% [markdown]
# >Note: ```utils``` is an additional file containing the methods that have been already developed for you to be used in this classroom. 
# For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 17
from utils import show_points_on_image

# %%
# Notebook cell 18
# Define the coordinates for the point in the image
# [x_axis, y_axis]
input_points = [ [350, 450 ] ]

# %%
# Notebook cell 19
input_labels = [1] # positive point

# %%
# Notebook cell 20
# Function written in the utils file
show_points_on_image(resized_image, input_points)

# %%
# Notebook cell 21
# Run the model
results = model(resized_image, device=device, retina_masks=True)

# %% [markdown]
# * Filter the mask based on the point defined before.

# %%
# Notebook cell 23
from utils import format_results, point_prompt

# %%
# Notebook cell 24
results = format_results(results[0], 0)

# %%
# Notebook cell 25
# Generate the masks
masks, _ = point_prompt(results, input_points, input_labels)

# %%
# Notebook cell 26
from utils import show_masks_on_image

# %%
# Notebook cell 27
# Visualize the generated masks
show_masks_on_image(resized_image, [masks])

# %% [markdown]
# * Define 'semantic masks' - two points to be masked.

# %%
# Notebook cell 29
# Specify two points in the same image
# [x_axis, y_axis]
input_points = [ [350, 450], [620, 450] ]

# %%
# Notebook cell 30
# Specify both points as "positive prompt"
input_labels = [1 , 1] # both positive points

# %%
# Notebook cell 31
# Visualize the points defined before
show_points_on_image(resized_image, input_points)

# %%
# Notebook cell 32
# Run the model
results = model(resized_image, device=device, retina_masks=True)

# %%
# Notebook cell 33


# %%
# Notebook cell 34
results = format_results(results[0], 0)

# %%
# Notebook cell 35
# Generate the masks
masks, _ = point_prompt(results, input_points, input_labels)

# %%
# Notebook cell 36
# Visualize the generated masks
show_masks_on_image(resized_image, [masks])

# %% [markdown]
# >Note: Please note that the results obtained from running this notebook may vary slightly from those demonstrated by the instructor in the video. 

# %% [markdown]
# * Identify subsections of the image by adding a **negative prompt**.

# %%
# Notebook cell 39
# Define the coordinates for the regions to be masked
# [x_axis, y_axis]
input_points = [ [350, 450], [400, 300]  ]

# %%
# Notebook cell 40
input_labels = [1, 0] # positive prompt, negative prompt

# %%
# Notebook cell 41
# Visualize the points defined above
show_points_on_image(resized_image, input_points, input_labels)

# %% [markdown]
# >Note: From the image above, the red star indicates the negative prompt and the green star the positive prompt.

# %%
# Notebook cell 43
# Run the model
results = model(resized_image, device=device, retina_masks=True)

# %%
# Notebook cell 44
results = format_results(results[0], 0)

# %%
# Notebook cell 45
# Generate the masks
masks, _ = point_prompt(results, input_points, input_labels)

# %%
# Notebook cell 46
# Visualize the generated masks
show_masks_on_image(resized_image, [masks])

# %% [markdown]
# >Note: From the image above, only the jacket, from the dog in the left, was segmented, so, it is following the command given by the positive prompt!

# %% [markdown]
# ### Prompting with bounding boxes

# %%
# Notebook cell 49
from utils import box_prompt

# %%
# Notebook cell 50
# Set the bounding box coordinates
# [xmin, ymin, xmax, ymax]
input_boxes = [530, 180, 780, 600]

# %%
# Notebook cell 51
from utils import show_boxes_on_image

# %%
# Notebook cell 52
# Visualize the bounding box defined with the coordinates above
show_boxes_on_image(resized_image, [input_boxes])

# %% [markdown]
# * Now, try to isolate the mask from the total output of the model.

# %%
# Notebook cell 54
from utils import box_prompt

# %%
# Notebook cell 55
results = model(resized_image, device=device, retina_masks=True)

# %%
# Notebook cell 56
# Generate the masks
masks = results[0].masks.data

# %%
# Notebook cell 57
masks

# %%
# Notebook cell 58
# Convert to True/False boolean mask
masks = masks > 0

# %%
# Notebook cell 59
masks

# %%
# Notebook cell 60
masks, _ = box_prompt(masks, input_boxes)

# %%
# Notebook cell 61
# Visualize the masks
show_masks_on_image(resized_image, [masks])

# %%
# Notebook cell 62
# Print the segmentation mask, but in its raw format
masks

# %%
# Notebook cell 63
# To visualize, import matplotlib
from matplotlib import pyplot as plt

# %%
# Notebook cell 64
# Plot the binary mask as an image
plt.imshow(masks, cmap='gray')

# %% [markdown]
# ### Try yourself! 
# Try the image segmentation explained before with your own images.

# %%
# Notebook cell 66
# To start opening images, upload your own or use the sample images we've uploaded, for example: younes.png
# The image younes.png is already uploaded in this classroom
raw_image = Image.open('younes.png')
raw_image

# %%
# Notebook cell 67
# Resize image
from utils import resize_image
resized_image = resize_image(raw_image, input_size=1024)
resized_image

# %%
# Notebook cell 68
# Define the coordinates for the point: [x_axis, y_axis]
input_points = [ [700, 500 ] ]

# %%
# Notebook cell 69
show_points_on_image(resized_image, input_points)

# %%
# Notebook cell 70
# Define the positive or negative prompt
# Run the model
results = model(resized_image, device=device, retina_masks=True)
results = format_results(results[0], 0)
# Generate the masks
masks, _ = point_prompt(results, input_points, input_labels)

# %%
# Notebook cell 71
# show_points_on_image(resized_image, input_points)
show_masks_on_image(resized_image, [masks])

# %% [markdown]
# ### Additional Resources
# 
# * For more on how to use [Comet](https://www.comet.com/site/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L2) for experiment tracking, check out this [Quickstart Guide](https://colab.research.google.com/drive/1jj9BgsFApkqnpPMLCHSDH-5MoL_bjvYq?usp=sharing) and the [Comet Docs](https://www.comet.com/docs/v2/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L2).
# * This course was based off a set of two blog articles from Comet. Explore them here for more on how to use newer versions of Stable Diffusion in this pipeline, additional tricks to improve your inpainting results, and a breakdown of the pipeline architecture:
#   * [SAM + Stable Diffusion for Text-to-Image Inpainting](https://www.comet.com/site/blog/sam-stable-diffusion-for-text-to-image-inpainting/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L2)
#   * [Image Inpainting for SDXL 1.0 Base Model + Refiner](https://www.comet.com/site/blog/image-inpainting-for-sdxl-1-0-base-refiner/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L2)
