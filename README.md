<div align="center">
  <h1 align="center">Prompt Engineering for Vision Models</h1>
  <p align="center">
    A DeepLearning.AI short course in collaboration with Comet
  </p>
  <p align="center">
    This repository contains my personal notes, code, and project files from the "Prompt Engineering for Vision Models" course.
    <br />
    <a href="https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/"><strong>Explore the Course »</strong></a>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Provider-DeepLearning.AI-blue?style=for-the-badge&logo=deeplearning.ai" alt="DeepLearning.AI">
  <img src="https://img.shields.io/badge/In%20Collaboration%20With-Comet-orange?style=for-the-badge&logo=comet" alt="Comet">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
</div>

---

## Introduction

Prompt engineering is a critical skill that has revolutionized the way we interact with AI, extending far beyond the realm of text-based models. This course, **"Prompt Engineering for Vision Models,"** provides a comprehensive exploration of how to effectively communicate with and control sophisticated computer vision models. It demonstrates that a "prompt" in the visual domain can be a text description, a set of pixel coordinates, a bounding box, or even a segmentation mask.

Throughout this hands-on course, we dive into the practical application of prompting cutting-edge vision models, including:

* **Meta's Segment Anything Model (SAM):** A powerful, universal model for image segmentation.
* **OWL-ViT:** A zero-shot object detection model capable of identifying objects based on natural language descriptions.
* **Stable Diffusion 2.0:** A state-of-the-art diffusion model for high-quality image generation.

The curriculum is designed to build a robust skill set, starting from fundamental image generation and progressing to advanced techniques like in-painting and model personalization. A key focus of the course is the iterative nature of visual prompt engineering. By leveraging **Comet**, an MLOps platform, we learn to meticulously track experiments, compare results, and optimize hyperparameters to achieve the desired visual outcomes. This systematic approach transforms prompting from a trial-and-error process into a disciplined engineering practice.

By the end of this course, you will not only understand the theoretical underpinnings of these models but also possess the practical ability to guide them to perform complex visual tasks, from precise object isolation to creating entirely new, personalized imagery using fine-tuning techniques like DreamBooth.

---

## Course Topics

This course is structured around four key pillars of modern computer vision. Each module provides both theoretical knowledge and practical, hands-on experience.

<details>
<summary><strong>1. Image Segmentation</strong></summary>

Image Segmentation is a fundamental task in computer vision that involves partitioning a digital image into multiple segments or sets of pixels, often referred to as superpixels. The primary goal of segmentation is to simplify or change the representation of an image into something that is more meaningful and easier to analyze. Unlike image classification, which assigns a single label to an entire image, or object detection, which places bounding boxes around objects, segmentation provides a much more granular understanding by assigning a class label to every single pixel in the image. This pixel-level precision makes it invaluable for applications requiring a detailed understanding of image content, such as medical image analysis (e.g., identifying tumor boundaries), autonomous driving (e.g., distinguishing road, vehicles, pedestrians, and obstacles), and satellite imagery analysis (e.g., mapping land use).

This course module demystifies image segmentation by introducing **Meta’s Segment Anything Model (SAM)**, a revolutionary foundation model for this task. What makes SAM exceptional is its "zero-shot" generalization capability, meaning it can segment objects and images that it was not explicitly trained on, without the need for additional training. This universality is achieved by training it on a massive and diverse dataset of 11 million images and 1.1 billion masks. The core innovation of SAM lies in its promptable design. Instead of being a fully automated system, it is designed to work interactively with a user, making segmentation more flexible and powerful.

In this course, we learn to prompt SAM using various methods to achieve precise segmentation:
* **Coordinate Prompts:** We can provide SAM with a single foreground point (a positive coordinate) to indicate the object we want to segment. The model intelligently identifies the object associated with that point and generates a corresponding mask. Conversely, we can provide background points (negative coordinates) to exclude specific regions from the final mask, allowing for the refinement of ambiguous boundaries. This interactive point-and-click approach is intuitive and highly effective for isolating complex objects.
* **Bounding Box Prompts:** For more explicit guidance, we can draw a bounding box around the target object. SAM then uses this box as a strong prior to generate a high-quality segmentation mask for the object contained within it. This is particularly useful when an object is well-defined and can be easily enclosed in a rectangle, serving as a faster alternative to placing multiple points.

By combining these prompting techniques, we can perform complex segmentation tasks with remarkable accuracy. For instance, we can first use a bounding box for an initial mask and then use positive and negative points to correct any errors or ambiguities, such as separating an object from its shadow or from other overlapping objects. The course emphasizes how this human-in-the-loop approach, powered by a sophisticated model like SAM, transforms image segmentation from a complex, often manual process into an efficient and interactive workflow. This skill is critical for data annotation, creative image editing, and scientific analysis.
</details>

<details>
<summary><strong>2. Object Detection</strong></summary>

Object Detection is a core computer vision technology that deals with identifying and locating objects within an image or video. It goes a step beyond simple image classification by not only answering "what" is in the image but also "where" it is. The output of an object detection model is typically a set of bounding boxes that tightly enclose each detected object, along with a class label (e.g., "car," "person," "dog") and a confidence score for each prediction. This capability is the backbone of countless real-world applications, including pedestrian detection for autonomous vehicles, inventory management in retail through shelf monitoring, quality control in manufacturing, and content moderation on social media platforms by identifying inappropriate items in images.

Traditionally, object detection models required extensive training on large, manually annotated datasets. If you wanted to detect a new object class, you would need to collect thousands of examples of that object, draw bounding boxes around them, and retrain the model. This process is time-consuming, expensive, and limits the model's utility to a predefined set of classes. The "Prompt Engineering for Vision Models" course introduces a paradigm shift with **OWL-ViT (Open-World Localization with Vision Transformers)**, a zero-shot object detection model. The term "zero-shot" signifies the model's remarkable ability to detect objects based on arbitrary text descriptions, even if it has never seen an example of that object class during its training.

The magic of OWL-ViT lies in its architecture, which is based on Vision Transformers (ViTs) and its training on vast datasets of paired images and text. This allows it to learn a rich, shared representation space where visual concepts and linguistic concepts are aligned. Consequently, the model can understand natural language prompts and use that understanding to search for corresponding objects in an image. Instead of being limited to a fixed list of categories, we can simply prompt the model with a text query like "a red backpack" or "two birds on a branch," and it will produce bounding boxes around the objects that match that description.

This module provides hands-on experience with this powerful prompting mechanism. We learn to:
* **Use Natural Language Prompts:** We explore how to formulate effective text prompts to detect specific and nuanced objects. This includes prompting for single objects, multiple distinct objects in the same image (e.g., "a cat and a dog"), and even objects described by their attributes ("the car parked on the left").
* **Iterate and Refine:** The course emphasizes that prompt engineering is an iterative process. The initial prompt might not always yield perfect results. We learn how to refine our text queries to improve detection accuracy, for example, by being more specific or by rephrasing the description to better align with the model's learned knowledge.

By mastering OWL-ViT, we gain the ability to build flexible and scalable object detection systems without the need for traditional model training cycles. This opens up new possibilities for applications where object categories are dynamic or not known in advance, such as building a personal photo organizer that can find images based on user queries like "find all pictures with my brown leather jacket."
</details>

<details>
<summary><strong>3. Image Generation</strong></summary>

Image Generation stands at the forefront of the generative AI revolution, enabling the creation of novel, high-quality images from textual descriptions. This technology has unlocked unprecedented potential in creative arts, design, advertising, entertainment, and rapid prototyping. At the heart of modern image generation are diffusion models, a class of deep generative models that have demonstrated state-of-the-art results. These models work by starting with a canvas of random noise and gradually refining it over a series of steps, guided by a text prompt, to produce a coherent image. This course provides a deep dive into this process using **Stable Diffusion 2.0**, one of the most powerful and widely used open-source diffusion models.

While a simple text prompt is the starting point, achieving a specific artistic style or desired composition requires a more nuanced approach. This is where the engineering aspect of "prompt engineering" becomes crucial. The course teaches that controlling the output of a diffusion model involves a careful interplay between the text prompt and a set of key hyperparameters. We learn to master these controls to steer the generation process with precision.

The key hyperparameters and techniques covered include:
* **Text Prompts:** We learn to craft detailed and descriptive prompts, often including keywords related to style (e.g., "impressionistic," "photorealistic," "cyberpunk"), composition ("wide-angle shot," "close-up"), and lighting ("cinematic lighting," "soft volumetric light"). We also explore the use of "negative prompts," which instruct the model on what to avoid, helping to eliminate common artifacts or unwanted elements (e.g., "blurry," "disfigured," "watermark").
* **Guidance Scale (CFG Scale):** This parameter controls how strictly the model adheres to the text prompt. A low guidance scale gives the model more creative freedom, which can lead to more varied but potentially less relevant results. A high guidance scale forces the model to follow the prompt more closely, which can improve coherence but may reduce creativity or lead to artifacts if pushed too high.
* **Number of Inference Steps:** This determines how many denoising steps the model takes to generate the final image. More steps generally lead to a higher-quality, more detailed image, but at the cost of longer generation times. Fewer steps are faster but may result in a less refined or noisy output. We learn to find the right balance for our specific needs.
* **Strength (for Image-to-Image):** When providing an initial image as a starting point (img2img), the strength parameter controls how much the model alters that original image. A low strength value will make subtle changes, preserving the overall structure, while a high strength value will allow the model to make more drastic transformations, using the initial image only as a loose guide.

Furthermore, the course combines these techniques to explore **in-painting**, a powerful application where we can replace specific objects or regions within an existing image. This workflow typically involves first using an image segmentation model like SAM to create a precise mask of the area to be modified, and then using Stable Diffusion with that mask to generate new content that seamlessly fills the selected region based on a new text prompt. This synergy between different vision models showcases the power of a multi-tool approach to creative image editing.
</details>

<details>
<summary><strong>4. Fine-tuning</strong></summary>

While prompting large, pre-trained models can achieve incredible results, there are times when we need to teach a model a completely new, highly specific concept it has never seen before. This could be a particular person's face, a unique product, a pet, or a specific artistic style. This process of adapting a general-purpose model for a specialized task is known as **fine-tuning**. Fine-tuning goes beyond prompting; it involves updating the model's internal parameters (weights) to incorporate new knowledge. This course introduces a highly effective and efficient fine-tuning technique called **DreamBooth**.

DreamBooth is designed specifically for diffusion models like Stable Diffusion and allows for the "personalization" of the model with just a handful of images (typically 3-5) of the new subject. The core idea is to create a strong association between a unique, rare text token (e.g., "a photo of a sks dog," where "sks" is the unique identifier) and the visual features of the subject in the provided images. By training the model on these image-text pairs, DreamBooth carefully adjusts the model's weights so that it "learns" the new concept and links it to the unique identifier.

This module provides a step-by-step guide through the DreamBooth fine-tuning process:
1.  **Data Curation:** The first step is to collect a small set of high-quality images of the target subject from various angles and in different lighting conditions. The quality and diversity of these input images are crucial for a successful fine-tuning outcome.
2.  **Training Process:** We learn how to set up and run the DreamBooth training process. This involves pairing our images with a carefully crafted prompt that includes the unique identifier. The training also uses a set of "class-prior" images generated by the model itself (e.g., generic photos of a "dog") to prevent the model from forgetting the general concept while learning the specific one. This technique, known as prior-preservation loss, helps maintain the model's overall generative quality and flexibility.
3.  **Inference and Generation:** Once the model is fine-tuned, the new concept becomes a part of its "vocabulary." We can then use the unique identifier in completely new prompts to generate novel images of our subject in any context or style we can imagine. For example, after fine-tuning on a specific pet dog with the identifier "sks dog," we could use prompts like "a photo of a sks dog wearing sunglasses on a beach" or "a renaissance oil painting of a sks dog as a king."

The course highlights the profound difference between prompting and fine-tuning. Prompting is about skillfully communicating with the model using its existing knowledge. Fine-tuning is about permanently extending that knowledge. Mastering DreamBooth unlocks the ability to create truly personalized and custom AI-generated art and imagery, transforming the generative model into a creative partner that understands our unique visual world. The practical knowledge gained in this module is invaluable for artists, designers, and developers looking to push the boundaries of creative AI.
</details>

---

## Acknowledgement

I would like to express my sincere gratitude to **[DeepLearning.AI](https://www.deeplearning.ai/)** and **[Comet](https://www.comet.com/site/)** for creating and offering this insightful and highly practical course. The clear explanations and hands-on labs provided an excellent learning experience.

Special thanks to the instructors for their expertise and to the developers and research communities behind the powerful open-source models that made this course possible:

* **Meta AI** for the Segment Anything Model (SAM)
* **Google Research** for OWL-ViT
* **Stability AI and LMU Munich** for Stable Diffusion

This repository is a testament to the knowledge and skills I have gained, and I hope it serves as a useful resource for others taking the course.
