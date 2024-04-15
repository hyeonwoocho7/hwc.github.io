---
title: Diffusion
subtitle: This is a diffusion tutorial. 

# Summary for listings and search engines
summary: This is a diffusion tutorial for undestanding.

# Link this post with a project
projects: []

# Date published
date: '2024-04-14T00:00:00Z'

# Date updated
lastmod: '2024-04-14T00:00:00Z'

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'What is Diffusion? and How is it work?'
  focal_point: ''
  placement: 2
  preview_only: true

authors:
  - admin

tags:
  - Academic

categories:
  - Diffusion
---

## Physical Intuition
Diffusion 아래의 그림과 같이 잉크를 물에 첨가하였을 때 퍼지게 되는 현상입니다.
잉크는 시간이 지나면 결국 고르게 분포하게 되고 Uniform하게 됩니다. 이때, 처음 잉크가 첨가되었을 때의 잉크의 밀도를 구하는 것은 어렵습니다. 여기서, Uniform한 잉크를 다시 처음 상태로 돌릴 수 있으면 어떨까요?? 딥러닝을 통해 해결해 보자!!

<img src="Physical-Intuition.png" alt="physical intuition" width="300"/>

잉크를 분자단위로 보게 되면, 작은 Sequence단위로 보게 되면 가우시안 분포내에서 shifting이 일어나게 됩니다.

## Diffusion 

Diffusion은 아래의 그림과 같이 forward, reverse process로 구성됩니다. 
Forward process는 원본 이미지에서 노이즈가 첨가되어 가는 과정, Reverse process는 노이즈에서 원본 이미지로 생성하는 과정입니다.

<img src="Diffusion.png" alt="Diffusion" width="300"/>

### Forward Diffusion Process

Foward process는 아래의 그림처럼, 원본 이미지 $x_{0}$로부터 가이시안 노이즈를 점진적으로 첨가하면서 복수의 time step를 거쳐서 $x_{T}$로 도달하게 됩니다. 

<img src="Forward.png" alt="Diffusion" width="300"/>

위의 과정에서 각 step은 아래의 식으로 표현할 수 있습니다.
$$ q(x_{t}|x_{t-1}) = \mathcal{N}(x_{t};\sqrt{1-\beta}x_{t-1}, \beta \text{I})$$
일반적으로 $\beta$는 0에 가까운 작은 값을 가지기에, 위의 식을 해석해보면 이전 step의 값을 감소시키면서 $\beta$ 노이즈를 더해주는것으로 foward process를 정의할 수 있습니다.

전체 과정은 아래의 식처럼,  $x_{0}$라는 이미지가 주어지게 되면 그 이미지에 노이즈를 조금씩 추가하는 $x_{1}$에서 $x_{T}$까지 과정을 아래의 joint distribution으로 표현 가능합니다. 
$$ q(x_{1:T}|x_{0}) = \prod_{t=1}^{T}q(x_{t}|x_{t-1})$$

DDPM 논문에서는 아래와 같이 각 time step의 가우시안 커널들이 연속적이기에 어떤 time step $t$에서 만들어지는 이미지를 정의할 수 있게 됩니다.

<img src="Diffusion-kernel.png" alt="Diffusion" width="300"/>

임의의 $x_{t}$를 reparamerization trick를 사용하면 아래와 같이 샘플링 할 수 있습니다.


$$\text{Let} \ \bar{\alpha}=\prod_{s=1}^{t}(1-\beta_{s})$$
$$ q(x_{t}|x_{0}) = \mathcal{N}(x_{t};\sqrt{\bar{\alpha}}x_{0}, (1-\bar{\alpha})\text{I})$$
$$\text{For sampling:} \ x_{t}=\bar{\alpha}x_{0}+\sqrt{1-\bar{\alpha}}\epsilon$$
$$ \text{where} \ \epsilon \sim \mathcal{N}(0, \text{I})$$

### Reverse Diffusion Process

처음에 언급한대로, Diffusion 현상은 Reverse Process가 Forward Process와 동일하게 작은 sequence내에서는 가우시안 분포를 따른다는 물리적인 intuition이 있습니다. 따라서, reverse 과정도 가우시안 분포를 따른다고 가정할 수 있습니다. 그럼, 이러한 가우시안을 딥러닝을 통해 모델링 할 수 있게 됩니다.

<img src="Reverse.png" alt="Diffusion" width="300"/>

$$ p(x_{T})=\mathcal{N}(x_{T};\text{0},\text{I})$$
$$ p_{\theta}(x_{t-1}|x_{t})=\mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t), {\sigma_{t}}^{2} \text{I})$$
즉, 네트워크는 위의 평균과 분산을 가지는 가우시안 분포를 학습하게 됩니다.

전체적인 Reverse 과정을 설명하는 식은 아래와 같습니다.
$$ p_{\theta}(x_{0:T})=p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})$$

