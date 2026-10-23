# %% [markdown]
# Source notebook: Notebook3_Image_Generation.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 4: Image Generation

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %% [markdown]
# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
#     ```
#     !pip install -q accelerate torch diffusers transformers comet_ml
#     ```

# %% [markdown]
# ### Set up Comet

# %%
# Notebook cell 5
import comet_ml

# %%
# Notebook cell 6
comet_ml.login(anonymous=True, project_name="4: Diffusion Prompting")

# %%
# Notebook cell 7
# Create the Comet Experiment for logging
exp = comet_ml.Experiment()

logged_artifact = exp.get_artifact("L4-data", "anmorgan24")
local_artifact = logged_artifact.download("./")

# %% [markdown]
# ### Load images

# %%
# Notebook cell 9
from PIL import Image

# %%
# Notebook cell 10
image=Image.open("L4_data/boy-with-kitten.jpg").resize((256, 256))
image_mask=Image.open("L4_data/cat_binary_mask.png").resize((256, 256))

# %% [markdown]
# >Note: the images referenced in this notebook have already been uploaded to the Jupyter directory, in this classroom, for your convenience. For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 12
import matplotlib.pyplot as plt

# %%
# Notebook cell 13
# Print the image
plt.imshow(image)

# %%
# Notebook cell 14
# Print the mask
plt.imshow(image_mask)

# %% [markdown]
# ### Import and prepare the model

# %% [markdown]
# #### Import [torch](https://pytorch.org/).

# %%
# Notebook cell 17
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# * Initialize the Stable Diffusion inpainting pipeline.
#   -- Note, if you'd like to learn more about the `float16` versus `bfloat16` data type and when you would use one or the other, please check out the short course ["Quantization Fundamentals" Lesson "Data Types and Sizes](https://learn.deeplearning.ai/courses/quantization-fundamentals/lesson/3/data-types-and-sizes)

# %%
# Notebook cell 19
from diffusers import StableDiffusionInpaintPipeline
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "./models/stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
    low_cpu_mem_usage=False if torch.cuda.is_available() else True,
).to(device)

# %%
# Notebook cell 20
# Set the value of seed manually for reproducibility of the results
seed = 66733
generator = torch.Generator(device).manual_seed(seed)

# %%
# Notebook cell 21
prompt = "a realistic green dragon"

# %% [markdown]
# ### Note
# - Starting from this point, the code that generates the images will  take too long to run in the given classroom environment. 
#   - So the code is left as markdown in the classroom.
# - In the classroom, and regardless of whether you have access to GPUs, you can still run the code that retrieves these image generation results using the experiment tracking tool.
# > - Thank you for your understanding as we try to make these courses free and accessible to as many people as possible. 💕 💫
# 
# - **Hardware requirements:** 
# To generate the images taught in this lab, a machine with at least 8 GB of CPU should suffice, and you can expect results within approximately 3 minutes for 3 inference steps. However, for tasks involving 10 steps, using a CPU may extend the execution time to about 10 minutes. Tasks involving 100 steps will significantly prolong the execution time on a CPU.
# Alternatively, consider utilizing a GPU for faster processing. With a GPU, such as a local one or through platforms like [Colab GPU](https://colab.research.google.com/), all three steps can be completed in under 1 second.

# %% [markdown]
# #### Define new **Comet** experiment.
# - The following image generation code will takes hours without a GPU.
# - Its results are saved with an experiment tracking tool (Comet), so that you can retrieve them in this classroom environment (and on any computer, regardless of GPU access).

# %% [markdown]
# ```Python
# exp = comet_ml.Experiment()
# 
# output = sd_pipe(
#   image=image,
#   mask_image=image_mask,
#   prompt=prompt,
#   generator=generator,
#   num_inference_steps=3,
# )
# ```

# %% [markdown]
# 
# ```Python
# generated_image = output.images[0]
# 
# exp.log_image(
#     generated_image,
#     name=f"{prompt}",
#     metadata={
#         "prompt": prompt,
#         "seed": seed,
#         "num_inference_steps": 3
#     }
# )
# 
# exp.end()
# ```

# %% [markdown]
# #### Retrieve the experiment results
# - Regardless of the environment that you are running in, you can retrieve the results of the experiment using the experiment tracking tool (Comet).

# %%
# Notebook cell 27
import io

reference_experiment = comet_ml.APIExperiment(
    workspace="ckaiser",
    project_name="4-diffusion-prompting",
    previous_experiment="b1b9e80bb0054b52a8914beec97d36a6"
)

reference_image = reference_experiment.get_asset_by_name(f"{prompt}")

# %% [markdown]
# -  Print the reference_image

# %%
# Notebook cell 29
plt.imshow(Image.open(io.BytesIO(reference_image)))

# %% [markdown]
# ### Note: 
# - We'll now explore different hyperparameters.
# - As before, running the image generation code would take hours in the classroom environment (or in any environment with GPUs).

# %% [markdown]
# * Set up a different 'number of inference steps'.

# %% [markdown]
# 
# ```Python
# exp = comet_ml.Experiment()
# 
# prompt = "a realistic green dragon"
# 
# exp.log_parameters({
#     "seed": seed,
#     "num_inference_steps": 100
# })
# ```

# %% [markdown]
# ```Python
# output = sd_pipe(
#   image=image,
#   mask_image=image_mask,
#   prompt=prompt,
#   generator=generator,
#   num_inference_steps=100,
# )
# 
# generated_image = output.images[0]
# 
# exp.log_image(
#     generated_image,
#     name=f"{prompt}",
#     metadata={
#         "prompt": prompt,
#         "seed": seed,
#         "num_inference_steps": 100
#     }
# )
# 
# exp.end()
# ```

# %% [markdown]
# #### Retrieve the experiment results
# - In the classroom or in any environment, you can retrieve the results of the image generation run by accessing the logs.

# %%
# Notebook cell 35
reference_experiment = comet_ml.APIExperiment(
    workspace="ckaiser",
    project_name="4-diffusion-prompting",
    previous_experiment="948c8e6cfd23420c86a0de5f65719955"
)

reference_image = reference_experiment.get_asset_by_name(f"{prompt}")

# %%
# Notebook cell 36
plt.imshow(Image.open(io.BytesIO(reference_image)))

# %% [markdown]
# #### Set up the different 'Guidance Scale' values.

# %% [markdown]
# - This code is best run on a GPU.  It's left as markdown in the classroom.

# %% [markdown]
# ```Python
# import numpy as np
# guidance_scale_values = [x for x in np.arange(0, 21, 10)]
# ```

# %% [markdown]
# ```Python
# exp = comet_ml.Experiment()
# 
# prompt = "a realistic green dragon"
# 
# num_inference_steps = 100 #if torch.cuda.is_available() else 10
# 
# exp.log_parameters({
#     "seed": seed,
# })
# ```

# %% [markdown]
# - Pass the guidance_scale to this pipeline

# %% [markdown]
# ```Python
# for guidance_scale in guidance_scale_values:
# 
#     output = sd_pipe(
#       image=image,
#       mask_image=image_mask,
#       prompt=prompt,
#       generator=generator,
#       num_inference_steps=num_inference_steps,
#       guidance_scale=guidance_scale
#     )
# 
#     generated_image = output.images[0]
# 
#     exp.log_image(
#         generated_image,
#         name=f"{prompt}",
#         metadata={
#             "prompt": prompt,
#             "seed": seed,
#             "num_inference_steps": num_inference_steps,
#             "guidance_scale": guidance_scale
#         }
#     )
# 
# exp.end()
# ```

# %% [markdown]
# #### Retrieve the experiment results
# - As before, regardless of whether you have access to GPUs or not, you can retrieve the results of the image generation code from the logs.

# %%
# Notebook cell 44
reference_experiment = comet_ml.APIExperiment(
    workspace="ckaiser",
    project_name="4-diffusion-prompting",
    previous_experiment="b34b94f94c594802b7090b6f2f1224f2"
)

reference_experiment.display(tab="images")

# %% [markdown]
# #### Set up another hyperparameter: 'strength'.

# %% [markdown]
# - Add the strength hyperparameter
# - This code is best run on a GPU.  It's left as markdown in the classroom.

# %% [markdown]
# ```Python
# strength_values = [x for x in np.arange(0.1, 1.1, 0.2)]
# ```

# %% [markdown]
# ```Python
# exp = comet_ml.Experiment()
# 
# prompt = "a realistic green dragon"
# 
# num_inference_steps = 200 if torch.cuda.is_available() else 10
# 
# exp.log_parameters({
#     "seed": seed,
# })
# 
# ```

# %% [markdown]
# ```Python
# for strength in strength_values:
# 
#     output = sd_pipe(
#       image=image,
#       mask_image=image_mask,
#       prompt=prompt,
#       generator=generator,
#       num_inference_steps=num_inference_steps,
#       strength=strength
#     )
# 
#     generated_image = output.images[0]
# 
#     exp.log_image(
#         generated_image,
#         name=f"{prompt}",
#         metadata={
#             "prompt": prompt,
#             "seed": seed,
#             "num_inference_steps": num_inference_steps,
#             "strength": strength
#         }
#     )
# 
# exp.end()
# ```

# %% [markdown]
# #### Retrieve the experiment results
# - With experiment tracking, you can compare the most recent run with the earlier ones.

# %%
# Notebook cell 51
reference_experiment = comet_ml.APIExperiment(
    workspace="ckaiser",
    project_name="4-diffusion-prompting",
    previous_experiment="2964615a382d46f09c3a36c50c74deef"
)

reference_experiment.display(tab="images")

# %% [markdown]
# ### Try adding a Negative Prompt.
# - If you set the negative prompt to "cartoon", this is asking the image generation model to not generate an image that looks like a cartoon.
# - Again, the image generation code is best run on a GPU.

# %% [markdown]
# ```Python
# exp = comet_ml.Experiment()
# 
# prompt = "a realistic green dragon"
# negative_prompt = "cartoon"
# 
# num_inference_steps = 100 if torch.cuda.is_available() else 10
# 
# exp.log_parameters({
#     "seed": seed,
# })
# 
# ```

# %% [markdown]
# ```Python
# output = sd_pipe(
#   image=image,
#   mask_image=image_mask,
#   prompt=prompt,
#   negative_prompt=negative_prompt,
#   generator=generator,
#   num_inference_steps=num_inference_steps,
#   guidance_scale=10
# )
# 
# generated_image = output.images[0]
# 
# exp.log_image(
#     generated_image,
#     name=f"{prompt}",
#     metadata={
#         "prompt": prompt,
#         "seed": seed,
#         "num_inference_steps": num_inference_steps,
#         "guidance_scale": 10
#     }
# )
# 
# exp.end()
# ```

# %% [markdown]
# #### Retrieve the experiment results

# %%
# Notebook cell 56
reference_experiment = comet_ml.APIExperiment(
    workspace="ckaiser",
    project_name="4-diffusion-prompting",
    previous_experiment="f05b04ac203a4f9aa606ea6cf9417fa3"
)

reference_image = reference_experiment.get_asset_by_name(f"{prompt}")

plt.imshow(Image.open(io.BytesIO(reference_image)))

# %% [markdown]
# ### Additional Resources
# * For more on how to use [Comet](https://www.comet.com/site/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L4) for experiment tracking, check out this [Quickstart Guide](https://colab.research.google.com/drive/1jj9BgsFApkqnpPMLCHSDH-5MoL_bjvYq?usp=sharing) and the [Comet Docs](https://www.comet.com/docs/v2/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L4).
# * This course was based off a set of two blog articles from Comet. Explore them here for more on how to use newer versions of Stable Diffusion in this pipeline, additional tricks to improve your inpainting results, and a breakdown of the pipeline architecture:
#   * [SAM + Stable Diffusion for Text-to-Image Inpainting](https://www.comet.com/site/blog/sam-stable-diffusion-for-text-to-image-inpainting/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L4)
#   * [Image Inpainting for SDXL 1.0 Base Model + Refiner](https://www.comet.com/site/blog/image-inpainting-for-sdxl-1-0-base-refiner/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L4)
