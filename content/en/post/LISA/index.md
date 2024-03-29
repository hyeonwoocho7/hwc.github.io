---
title: LISA:Reasoning Segmentation Via Large Language Model
subtitle: This is review of LISA paper. 

# Summary for listings and search engines
summary: This is review of LISA paper.

# Link this post with a project
projects: []

# Date published
date: '2024-01-16T00:00:00Z'

# Date updated
lastmod: '2024-01-17T00:00:00Z'

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'New semgmentaion capability for current multi-modal LLM'
  focal_point: ''
  placement: 2
  preview_only: true

authors:
  - admin

tags:
  - Academic

categories:
  - Multi-Modal
---

## Introduction
In everyday situations, users tend to instruct robots with commands like "Change the TV channel" rather than explicitly stating, "First, go to the table, find the TV remote, and press the button to change the channel." However, most perception systems depend on target objects or categories explicitly defined by humans. These systems often lack the ability to actively infer and understand user intentions from implicit instructions.

This paper introduces a new segmentation task called "reasoning segmentation." This task is designed to output segmentations when given complex and implicit query text. Interestingly, the query text can encompass explicit elements like "the orange" and more complex descriptions involving reasoning or world knowledge, such as "the food with high Vitamin C." To achieve this task, the model needs two key abilities: 1) reasoning complex and implicit text queries jointly with the image, and 2) producing segmentation masks.

 Leveraging the excellent understanding and inference abilities of recent Large Language Models (LLMs), the authors of this paper propose utilizing LLMs for these tasks. While many studies have applied multi-modal LLMs to accommodate visual input, these models have primarily excelled in text generation tasks and shown insufficient performance on tasks requiring fine-grained outputs like segmentation masks.

The paper introduces LISA (Large Language Instructed Segmentation Assistant), which includes an additional token, "< SEG >" in the existing text to grant segmentation abilities to the multi-modal model. Upon generating the < SEG > token, its hidden embedding is decoded into the corresponding segmentation mask. Remarkably, LISA demonstrates not only zero-shot capabilities but also effective performance on complex reasoning, even when trained solely on segmentation and referring segmentation datasets. Furthermore, fine-tuning with just 239 image-instruction reasoning segmentation pairs significantly enhances LISA's performance.

In summary, LISA can handle four scenarios: 1) complex reasoning, 2) world knowledge, 3) explanatory answers, and 4) multi-turn conversations. The accompanying figure provides examples of these scenarios.

<img src="senarios.png" alt="senario" width="500"/>


## Method
 <img src="Method.png" alt="method" width="500"/>

You can understand the approach of LISA by looking at the pipeline in the above figure. Firstly, the multi-modal LLM $\mathcal{F}$ receives $x_{img}$ and $x_{txt}$ as inputs and outputs $\hat{y}_ {txt}$. To generate a segmentation mask, the output $\hat{y}_ {txt}$ must include < SEG >. The embedding $\hat{h}_ {seg}$ of the last layer corresponding to the < SEG > token is extracted, and an MLP projection layer $\gamma$ is applied to obtain $h_ {seg}$. Simultaneously, the vision backbone $\mathcal{F}_ {enc}$ extracts the visual embedding $f$ from the input $x_{img}$. Finally, $h_{seg}$ and $f$ are input into the decoder $\mathcal{F}_ {dec}$ to produce the final segmentation mask $\hat{M}$.

**Training Objectives.**
The model is trained end-to-end with text generation loss $L_{txt}$ and segmentation mask loss $L_{mask}$. $L_{txt}$ is defined as auto-regressive cross-entropy, and $L_{mask}$ is defined as per-pixel binary cross-entropy loss and DICE loss.

**Training Data Formulation.**
As shown in the figure below, a suitable template for visual question answering is provided for training on three different types of public datasets (Semantic Segmentation, Referring Segmentation, Visual Question Answering).

 <img src="train_data_formulation.png" alt="method" width="500"/>



**Trainable Parameters**
To effectively fine-tune while maintaining the generalization ability of the pretrained multi-modal LLM $F$, [LoRA](https://arxiv.org/abs/2106.09685) is applied.

## Experiments
**Network Architecture**
The author employed LLaVA as a multi-modal LLM $F$ model with the ViT-H SAM backbone as the vision backbone $\mathcal{F}_ {enc}$. The projection layer $\gamma$ used an MLP with channels [256, 4096, 4096].

**Evaluation Metrics**
Evaluation of segmentation is done through gIoU and cIoU. 
For a detailed explanation of these metrics, refer to the [Link](https://www.youtube.com/watch?v=4wXXNQ4Ylrk)


**Reasoning Segmentation Results**

 <img src="Reasoning_seg_table.png" alt="Reason_Seg" width="500"/>

The table above presents the evaluation results for the reasoning segmentation task. 'ft' represents the results fine-tuned with 239 reasoning image-instruction pairs. Compared to other methods, it showed an approximately 20% performance improvement. Particularly, LISA-13B outperformed LISA-7B, especially for long sentences. Existing methods were found lacking in reasoning ability or understanding of world knowledge required for reasoning segmentation, as they were designed for referring segmentation tasks.

**Qualitative Results**

<img src="visual_result.png" alt="vis" width="500"/>

Observing the results above, the proposed method demonstrated higher reasoning ability compared to other referring segmentation methods.




## Conclusion
This study introduces a new segmentation task called reasoning segmentation. This task is more challenging than the traditional referring segmentation task and requires understanding and inference for implicit user instructions. For this task, LISA, by applying segmentation capability to a multi-modal LLM, outperformed reasoning segmentation tasks with reasoning-free datasets.

## My opinions or thinking
- The role of the vision backbone $\mathcal{F}{enc}$ seems to be segmenting all objects in the image, while the Multi-modal LLM $F$ appears to extract embeddings for the desired instances from $x{txt}$. I wonder about the potential outcomes if $\mathcal{F}_{enc}$ fails to correctly identify all objects.
- I'm curious about whether the correlation between visual feature $f$ and $h_{seg}$ will be orthogonal or similar. The reason is that $F$ also takes visual input $x_{img}$ to extract features, raising doubts about the necessity of extracting features from $\mathcal{F}_{enc}$.

## Reference
- Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., & Jia, J. (2023). Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.