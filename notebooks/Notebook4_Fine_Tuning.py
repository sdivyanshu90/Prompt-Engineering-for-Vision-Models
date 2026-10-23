# %% [markdown]
# Source notebook: Notebook4_Fine_Tuning.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 5: Fine-Tuning

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %% [markdown]
# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
#     ```
#     !pip install -q comet_ml accelerate torch diffusers transformers peft bitsandbytes xformers triton
#     ```

# %% [markdown]
# ### Set up Comet

# %% [markdown]
# * Here you will use the [HuggingFace DreamBooth](https://huggingface.co/docs/diffusers/en/training/dreambooth) training.

# %%
# Notebook cell 6
import comet_ml

# %%
# Notebook cell 7
comet_ml.login(anonymous=True)

# %% [markdown]
# ### Import and prepare the model

# %%
# Notebook cell 9
import torch

if torch.cuda.is_available():
    model_name = 'stabilityai/stable-diffusion-xl-base-1.0'
else:
    model_name = './models/runwayml/stable-diffusion-v1-5'

# %%
# Notebook cell 10
# Define hyperparameters

hyperparameters = {
    "instance_prompt": "a photo of a [V] man",
    "class_prompt": "a photo of a man",
    "seed": 4329,
    "pretrained_model_name_or_path": model_name,
    "resolution": 1024 if torch.cuda.is_available() else 512,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "num_class_images": 200,
    "prior_loss_weight": 1.0
}

# %% [markdown]
# * Set new **Comet** experiment

# %%
# Notebook cell 12
experiment = comet_ml.Experiment()

# %% [markdown]
# ### Load images

# %%
# Notebook cell 14
from utils import DreamBoothTrainer

# %%
# Notebook cell 15
trainer = DreamBoothTrainer(hyperparameters)

# %% [markdown]
# #### Note
# - The code that generates images requires a GPU to run.
# - The code is left here in markdown, but if you have access to GPUs outside of the classroom, you can run it there.
# - In the classroom, you'll still be able to follow along by retrieving the generated images from the experiment tracking tool (Comet).

# %% [markdown]
# ```Python
# # To run the training pipeline
# trainer.generate_class_images()
# ```

# %% [markdown]
# ```Python
# # To see the content of generate_class_image
# ??trainer.generate_class_images
# ```

# %% [markdown]
# #### Get class images (using artifacts).

# %%
# Notebook cell 20
import shutil

# %%
# Notebook cell 21
# Get images
class_artifact = experiment.get_artifact('ckaiser/class-images-15')
class_artifact.download('./')

# %%
# Notebook cell 22
shutil.unpack_archive('./class.zip', './class')

# %% [markdown]
# >Note: the images referenced in this notebook have already been uploaded to the Jupyter directory, in this classroom, for your convenience. For further details, please refer to the **Appendix** section located at the end of the lessons.

# %%
# Notebook cell 24
# Print some images
trainer.display_images("class")

# %% [markdown]
# * Get the instance dataset (images of Andrew)

# %%
# Notebook cell 26
andrew_artifact = experiment.get_artifact('ckaiser/andrew-dataset')
andrew_artifact.download('./')

shutil.unpack_archive('./andrew-dataset.zip', './instance')

# %%
# Notebook cell 27
# Print some images
trainer.display_images("instance")

# %% [markdown]
# ### Initialize the model
# - It will take some time (several minutes) to initialize the model.

# %%
# Notebook cell 29
tokenizer, text_encoder, vae, unet = trainer.initialize_models()

# %% [markdown]
# > Note: see the video lesson for the LoRA explanation.

# %%
# Notebook cell 31
# Add noise to generate images in Stable Diffusion
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler.from_pretrained(
    trainer.hyperparameters.pretrained_model_name_or_path,
    subfolder="scheduler"
)

# %%
# Notebook cell 32
unet = trainer.initialize_lora(unet)

# %%
# Notebook cell 33
optimizer, params_to_optimize = trainer.initialize_optimizer(unet)

# %%
# Notebook cell 34
# Initialize the datasets
train_dataset, train_dataloader = trainer.prepare_dataset(tokenizer, text_encoder)
lr_scheduler = trainer.initialize_scheduler(train_dataloader, optimizer)

# %%
# Notebook cell 35
unet, optimizer, train_dataloader, lr_scheduler = trainer.accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler)

# %%
# Notebook cell 36
total_batch_size = \
    trainer.hyperparameters.train_batch_size * \
    trainer.hyperparameters.gradient_accumulation_steps

# %% [markdown]
# #### Note
# - Starting from this point, the code demonstrated by the instructor will not execute in this notebook due to computational resource constraints. However, we provide the code here for you to run if you have access to a GPU or similar resources.
# - Thank you for your understanding as we work to provide free and accessible courses.

# %% [markdown]
# ```Python
# from tqdm import tqdm
# 
# global_step = 0
# epoch = 0
# 
# progress_bar = tqdm(
#     range(0, trainer.hyperparameters.max_train_steps),
#     desc="Steps"
# )
# ```

# %% [markdown]
# ```Python
# for epoch in range(0, trainer.hyperparameters.num_train_epochs):
#     unet.train()
# 
#     for step, batch in enumerate(train_dataloader):
#         with trainer.accelerator.accumulate(unet):
#             pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
#             model_input = vae.encode(pixel_values).latent_dist.sample()
#             model_input = model_input * vae.config.scaling_factor
# 
#             noise = torch.randn_like(model_input)
#             bsz, channels, height, width = model_input.shape
# 
#             timesteps = torch.randint(
#                 0,
#                 noise_scheduler.config.num_train_timesteps,
#                 (bsz,),
#                 device=model_input.device
#             )
# 
#             timesteps = timesteps.long()
#             noisy_model_input = noise_scheduler.add_noise(
#                 model_input,
#                 noise,
#                 timesteps
#             )
# 
#             encoder_hidden_states = batch["input_ids"]
# 
#             model_predict = unet(
#                 noisy_model_input,
#                 timesteps,
#                 encoder_hidden_states,
#                 return_dict=False,
#             )[0]
# 
#             target = noise
# 
#             model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
#             target, target_prior = torch.chunk(target, 2, dim=0)
# 
#             instance_loss = \
#                 F.mse_loss(
#                     model_pred.float(),
#                     target.float(),
#                     reduction="mean"
#                 )
#             
#             prior_loss = \
#                 F.mse_loss(
#                     model_pred_prior.float(),
#                     target_prior.float(),
#                     reduction="mean"
#                 )
#             
#             loss = \
#                 instance_loss + \
#                 trainer.hyperparameters.prior_loss_weight * \
#                 prior_loss
#             
#             trainer.accelerator.backward(loss)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             global_step +=1
# 
#         loss_metrics = {
#             "loss": loss.detach().item,
#             "prior_loss": prior_loss.detach().item,
#             "lr": lr_scheduler.get_last_lr()[0],
#         }
# 
#         experiment.log_metrics(loss_metrics, step=global_step)
# 
#         progress_bar.set_postfix(**loss_metrics)
#         progress_bar.update(1)
# 
# 
#         if global_step >= trainer.hyperparameters.max_train_steps:
#             break
# 
#     trainer.save_lora_weights(unet)
# experiment.add_tag(f"dreambooth-training")
# experiment.log_parameteres(trainer.hyperparameters)
# trainer.accelerator.end_training()
# ```

# %% [markdown]
# #### Retrieve the training results
# - You can get the training results using the experiment tracking tool, Comet.

# %%
# Notebook cell 41
training_experiment = \
    comet_ml.APIExperiment(
        previous_experiment="d92519b1f657497e8569a2c8e989b457"
    )

# %%
# Notebook cell 42
# See the experiment
training_experiment.display()

# %% [markdown]
# * Prompts to generate images of Andrew.

# %%
# Notebook cell 44
prompts = [
    "a photo of a [V] man playing basketball",
    "a photo of a [V] man riding a horse",
    "a photo of a [V] man at the summit of a mountain",
    "a photo of a [V] man driving a convertible",
    "a photo of a [V] man riding a skateboard on a huge halfpipe",
    "a mural of a [V] man, painted by graffiti artists"
]

validation_prompts = [
    "a photo of a man playing basketball",
    "a photo of a man riding a horse",
    "a photo of a man at the summit of a mountain",
    "a photo of a man driving a convertible",
    "a photo of a man riding a skateboard on a huge halfpipe",
    "a mural of a man, painted by graffiti artists"
]

# %% [markdown]
# #### Note
# - The folowing code requires GPUs.

# %% [markdown]
# ```Python
# from diffusers import DiffusionPipeline
# 
# pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipeline.load_lora_weights("./andrew-model")
# 
# for prompt in prompts:
#     with torch.no_grad():
#         images = pipeline(
#             prompt = prompt,
#         ).images
# 
#         experiment.log_image(images[0], metadata={
#             "prompt": prompt,
#             "model": hyperparameters.pretrained_model_name_or_path,
#         })
# 
# for prompt in validation_prompts:
#     with torch.no_grad():
#         images = pipeline(
#             prompt=prompt,
#         ).images
# 
#     experiment.log_image(images[0], metadata={
#             "prompt": prompt,
#             "model": hyperparameters.pretrained_model_name_or_path,
#         })
# ```

# %% [markdown]
# #### Retrieve the image generation results
# - You can view the results of image generation regardless of whether you have access to GPUs, using the experiment tracking tool.

# %%
# Notebook cell 48
inference_experiment = comet_ml.APIExperiment(
        previous_experiment="0eb292126ab5476ab0c863061a400bdc"
    )

# %%
# Notebook cell 49
# See the experiment
inference_experiment.display(tab="images")

# %% [markdown]
# ### Additional Resources
# * For more on how to use [Comet](https://www.comet.com/site/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L5) for experiment tracking, check out this [Quickstart Guide](https://colab.research.google.com/drive/1jj9BgsFApkqnpPMLCHSDH-5MoL_bjvYq?usp=sharing) and the [Comet Docs](https://www.comet.com/docs/v2/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L5).
# * This course was based off a set of two blog articles from Comet. Explore them here for more on how to use newer versions of Stable Diffusion in this pipeline, additional tricks to improve your inpainting results, and a breakdown of the pipeline architecture:
#   * [SAM + Stable Diffusion for Text-to-Image Inpainting](https://www.comet.com/site/blog/sam-stable-diffusion-for-text-to-image-inpainting/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L5)
#   * [Image Inpainting for SDXL 1.0 Base Model + Refiner](https://www.comet.com/site/blog/image-inpainting-for-sdxl-1-0-base-refiner/?utm_source=dlai&utm_medium=course&utm_campaign=prompt_engineering_for_vision_models&utm_content=dlai_L5)

# %% [markdown]
# ## Did you like this course?
# 
# - If you liked this course, could you consider giving a rating and share what you liked? 💕
# - If you did not like this course, could you also please share what you think could have made it better? 🙏
# 
# #### A note about the "Course Review" page.
# The rating options are from 0 to 10, and used to calculate the "Net Promoter Score"
# - A score of 9 or 10 means you like the course.💫 💕
# - A score of 7 or 8 means you feel neutral about the course (neither like nor dislike). 🙄
# - A score of 0,1,2,3,4,5 or 6 all mean that you do not like the course. 😭 
