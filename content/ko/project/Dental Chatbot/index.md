---
title: Dental Chatbot
summary: A RAG system for answering questions about dental-related question. 
tags:
  - Deep Learning
date: '2024-05-03T00:00:00Z'

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

## Goal⛳️: RAG를 활용한 사내 웹사이트 챗봇 개발

## Motivation 📚

* 사이트내의 Q&A 질문들을 자동으로 답변하여 노동력 절감.
* 충분한 양의 데이터 보유 (Q&A 질문 건수: 약 70,000건).
* LLM 학습에는 많은 비용 (GPU 및 전력)이 요구되지만, RAG을 활용하게 되면 학습이 필요 없음.


## Data 🏦
<img src="website.png" width="450px" height="300px" title="website example" alt="website"></img><br/>

* 데이터 구성
  * 질문 번호
  * 질문 카테고리
  * 질문 제목 
  * 질문 내용
  * 답변 내용

* 데이터 전처리
  * 질문 혹은 답변의 길이가 매우 짧은 경우 사용 X.
  * 질문과 답변이 연속적으로 이루어지는 경우, 답변에 질문이 포함되어 해당 항목을 제거.


## Workflow 👓
<img src="workflow.png" width="1000px" height="300px" title="website example" alt="website"></img><br/>