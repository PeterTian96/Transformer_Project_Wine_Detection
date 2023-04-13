# Wine Variety Recommendation Using mT5

DS 5899 Final Project

Peter Tian, Enya Tan, Kelly Chen

## Overview

The goal of this project is to build a wine recommendation system using transformers. Wine recommendation is an important application of data science and machine learning in the wine industry. By using machine learning techniques to build a recommendation system, we can provide personalized wine recommendations to consumers based on their preferences and past wine ratings. This can help consumers discover new wines that they may not have otherwise tried, while also increasing customer satisfaction and loyalty. Furthermore, wine recommendation systems can be used by wine retailers to improve sales and marketing strategies. 

We preprocess the data by cleaning and tokenizing the text data, as well as converting the rating scores to categorical labels. Once the data is preprocessed, it is split into training and validation sets. Next, the transformers library is used to build a recommendation model. The wine recommendation task can be framed as a sequence-to-sequence problem, where the input is a wine description and the output is a recommended wine. Once the model is trained on the training set, it is evaluated on the validation set using two metrics: the Rouge score and the validation loss. 

Wine recommendation is a valuable application of machine learning in the wine industry, with potential benefits for both consumers and businesses.

![image](https://user-images.githubusercontent.com/68664277/231605168-372ee93c-1a9f-44ad-9998-8182869ac468.png)

## Data

The dataset used for this project is the [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) dataset available on Kaggle, which contains over 130k wine reviews along with their descriptions and ratings. Each wine review in the dataset contains the following information:

**Country**: The country where the wine was produced.

**Description**: A textual description of the wine's flavor, aroma, and other characteristics.

**Designation**: The vineyard or specific plot where the grapes that made the wine were grown.

**Points**: A rating score between 0 and 100 that reflects the wine's quality.

**Price**: The cost of a bottle of wine.

**Province**: The specific region within a country where the wine was produced.

**Region_1**: A broader geographical region within the province where the wine was produced.

**Region_2**: A more specific sub-region within the region_1.

**Taster_name**: The name of the taster who wrote the review.

**Taster_twitter_handle**: The Twitter handle of the taster who wrote the review.

**Title**: The title of the wine review.

**Variety**: The type of grapes used to make the wine.

**Winery**: The winery that produced the wine.

## Models

T5 is a massively multilingual language model that closely follows the recipe of the T5 model, which uses a unified "text-to-text" format for all text-based NLP problems. mT5 is pre-trained on a large-scale multilingual corpus called mC4, which covers 101 languages and is an extended version of the C4 pre-training dataset. 

The mT5 model uses a basic encoder-decoder Transformer architecture and is trained on a masked language modeling "span-corruption" objective. It is pre-trained on unlabeled data only and is available in different sizes, ranging from 60 million to 11 billion parameters

[image](https://user-images.githubusercontent.com/89152255/231827987-1c044a4e-0262-41fb-8b5a-1eb881be54c2.png)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FMT5-encoder-model-overview-with-additional-fully-connected-head_fig1_363234400&psig=AOvVaw06ZQ1iKTduHKpfd13fzN6K&ust=1681490537463000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCLi4zMOmp_4CFQAAAAAdAAAAABAQ)


## Results

## Conclusion

## Critical Analysis

## Resource Links

https://arxiv.org/pdf/2010.11934.pdf

## Video Recording
