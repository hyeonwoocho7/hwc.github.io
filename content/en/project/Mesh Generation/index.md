---
title: Mesh Generation
summary: In this project, we build AI that generates a crown mesh from neighborhood tooth mesh.
  - Deep Learning
date: '2024-10-18T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption: 'Overview'
  focal_point: Smart

# links:
#   - icon: twitter
#     icon_pack: fab
#     name: Follow
#     url: https://twitter.com/georgecushen
url_code: ''
url_pdf: ''
url_slides: ''
url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example
---


## Goal⛳️: Generate a crown mesh from adjacent tooth meshes
* It is crucial for the predicted target tooth to have proper occlusion without colliding with adjacent teeth, so the key point is for the model to understand the position of the adjacent teeth.
* Since the requirement for the patient's crown design is to output as a mesh, it is important to learn the connectivity between point clouds.



## Motivation 📚
 * Reduce time and manpower resources in dental labs by automating crown design.
 * Use the generation of various meshes for data augmentation.


## Data 🏦

### Data composition
* Format: Mesh
* Number of data points: 700
* Pairs of adjacent and target teeth


### Data preprocessing
* Rigid ICP is used to align the positions between all tooth datasets.
* Teeth are segmented by tooth number.
* Pairs are formed between the adjacent and target teeth for each segmented tooth.

## Approach 💡
* Using the PoinTr model to solve the point completion problem, it is designed to understand the global positional information of adjacent teeth.
* Additionally, MLP is attached to predict the normal vectors from the point cloud to generate the mesh surface.
* To train the normal vector prediction, it is designed to solve the Poisson equation using Differentiable Poisson Surface Reconstruction.


## Workflow 👓
<img src="workflow.png" width="900px" height="300px" title="workflow" alt="workflow"></img><br/>
1. Extract the point cloud from the adjacent tooth mesh.
2. Use the PoinTr model to generate the point cloud for the target tooth.
3. Pass the point cloud through an MLP to predict the offset and normal vector for the point cloud.
4. Perform Differentiable Poisson Surface Reconstruction using the point cloud and normal vector to create a surface grid (Prediction grid).
5. Sample point clouds and normal vectors from the GT mesh to create a GT grid.
6. Train using Chamfer Distance Loss between the GT point cloud and the predicted point cloud, and MSE Loss between the GT grid and the prediction grid.

### Differentiable Poisson Surface Reconstruction
An approach to solving the Poisson equation to reconstruct an implicit surface from a point cloud and normal vectors:
1. Convert the point cloud and normal vectors into a grid format.
2. Transform the rasterized data into the frequency domain (Fourier Transform).
3. Compute the divergence of the normal vectors.
4. Calculate the Laplacian operator.
5. Solve the Poisson equation (solution in the frequency domain).
6. Transform the solution of the Poisson equation back into the spatial domain.
7. Return the implicit surface field in the form of a 3D grid (represented as positive inside the surface, negative outside, and zero on the surface).

## Results 👨‍🔬
<img src="result.png" width="900px" height="700px" title="result" alt="result"></img><br/>
The table shows the generated tooth mesh, the actual tooth mesh, the collision map between the two meshes, and the final result occluded with the adjacent teeth. When comparing the top surface of the generated tooth to the actual tooth, the curvatures and occlusion within the tooth appear to be well-reproduced. This indicates that global localization information from adjacent teeth was effectively incorporated through PoinTr, while local features of the mesh, such as surface details, were well-captured by training with Differentiable Poisson Surface Reconstruction.