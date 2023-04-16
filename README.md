# DS 5899 Final Project

# Wine Variety Recommendation Using mT5

**Team Member**: Enya Tan, Kelly Chen, Peter Tian

## Overview

The goal of this project is to build a wine recommendation system using transformers. Wine recommendation is an important application of data science and machine learning in the wine industry. By using machine learning techniques to build a recommendation system, we can provide personalized wine recommendations to consumers based on their preferences and past wine reviews. This can help consumers discover new wines that they may not have otherwise tried, while also increasing customer satisfaction and loyalty. Furthermore, wine recommendation systems can be used by wine retailers to improve sales and marketing strategies. 

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

**Sample Data**:

| Country  | Description                                                   | Designation   | Points | Price | Province          | Region_1           | Region_2           | Taster Name       | Taster Twitter Handle | Title                                                         | Variety      | Winery     |
| ---      | ---                                                           | ---           | ---    | ---   | ---               | ---                | ---                | ---              | ---                   | ---                                                           | ---          | ---        |
| Italy    | Aromas include tropical fruit, broom, brimstone and dried herb. | Vulkà Bianco  | 87     |       | Sicily & Sardinia | Etna               |                    | Kerin O’Keefe    | @kerinokeefe         | Nicosia 2013 Vulkà Bianco  (Etna)                             | White Blend   | Nicosia    |
| Portugal | This is ripe and fruity, a wine that is smooth while still...    | Avidagos      | 87     | 15    | Douro             |                    |                    | Roger Voss       | @vossroger           | Quinta dos Avidagos 2011 Avidagos Red (Douro)                  | Portuguese Red | Quinta...  |
| US       | Tart and snappy, the flavors of lime flesh and rind dominate.   |               | 87     | 14    | Oregon            | Willamette Valley | Willamette Valley | Paul Gregutt     | @paulgwine           | Rainstorm 2013 Pinot Gris (Willamette Valley)                 | Pinot Gris   | Rainstorm  |




## Models

Multilingual T5 (mT5) is a massively multilingual language model that closely follows the recipe of the T5 model, which uses a unified "text-to-text" format for all text-based NLP problems. mT5 is pre-trained on a large-scale multilingual corpus called mC4, which covers 101 languages and is an extended version of the C4 pre-training dataset. It is launched by Google, a multilingual model of Google's T5 model. The massively pre-trained multilingual model has schemed in the research paper "mT5: A massively multilingual pre-trained text-to-text transformer" and is presented by the group of researchers & authors including Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 

The mT5 model uses a basic encoder-decoder Transformer architecture and is trained on a masked language modeling "span-corruption" objective. It is pre-trained on unlabeled data only and is available in different sizes, ranging from 60 million to 11 billion parameters. The model architecture is based on a sequence-to-sequence transformer framework, which uses an encoder-decoder architecture similar to the T5 model. The encoder and decoder in the mT5 model are both composed of multiple layers of transformers, and each transformer layer is composed of multi-head self-attention mechanisms and feedforward neural networks (FFN). During training, the mT5 model predicts the next word by minimizing the negative log-likelihood (NLL) of the language model.

<div align=center><img src="https://github.com/PeterTian96/Transformer_Project_Wine_Detection/blob/main/images/Model-architecture-of-Transformer-2-that-was-used-in-the-mT5-model.jpg?raw=true"></div>
<div align="center"> Model architecture of Transformer that was used in the mT5 model [2] </div>     <br>

Important parameters in the mT5 model include:

 - Model size: The mT5 series includes different model sizes, including mT5-Small, mT5-Base, mT5-Large, mT5-XL, and mT5-XXL, with varying numbers of parameters and computational power.
 
- Number of hidden layers: Each encoder and decoder in the mT5 model is composed of multiple stacked transformer layers, with different model sizes having different numbers of transformer layers.

- Multi-head attention mechanism: This is a critical component in the transformer model, used to perform self-attention calculations on input sequences at different positions to capture dependencies between different positions.

- Feedforward neural network: Another key component in the transformer model, used to perform nonlinear transformations on the results of self-attention calculations in transformer layers.

- Sequence length: The maximum sequence length accepted by the mT5 model.

These parameters can affect the performance and training time of the model, so they should be considered when selecting and using the mT5 model. In addition to the parameters mentioned above, other parameters can also have a significant impact on the performance and training time of the mT5 model, such as learning rate, batch size, dropout rate, etc.

**mT5-Small** : 300 million parameters

**mT5-Base** : 580 million parameters

**mT5-Large** : 1.2 billion parameters

**mT5-XL** : 3.7 billion parameters

**mT5-XXL** : 13 billion parameters

All models in the mT5 series, including mT5-Small, mT5-Base, mT5-Large, mT5-XL, and mT5-XXL, are trained on the same training dataset which is called the C4 dataset, containing trillions of words of text at the word-level and was generated by Google by crawling and cleaning web pages. The dataset is designed for large-scale pre-training to help mT5 models learn general natural language processing knowledge and skills. Before training, the C4 dataset is preprocessed and encoded into a format that can be accepted by the T5 model. Therefore, the differences between different mT5 models mainly come from the differences in their model parameters and structure.

**Why mT5 for this project?**

- NLP Capability: MT5 is a pre-trained language model that has the ability to process natural language, so it can identify key information in the wine text, such as descriptions, ratings, and comments in Wine Reviews, which can help us better understand users' evaluations and preferences for wine.

- Multilingual Support: MT5 is a multilingual model that supports more than 100 languages, which means that MT5 can be used to process user reviews from different countries and regions, and recommend wine varieties from different countries and regions.

- Contextual Understanding Ability: MT5 has the ability to understand context, and can identify users' preferences for a certain wine variety through contextual information such as ratings and comments, which helps improve the accuracy of our wine recommendation.

- Pretrained Model Effectiveness: MT5 is pre-trained on large-scale datasets, so it can learn more language and semantic knowledge from the pre-trained model, which allows it to handle different natural language tasks such as Wine Recommendation more accurately.

In summary, using the MT5 model for wine recommendation can have better recommendation accuracy and coverage, thereby improving user satisfaction and experience.

## Code Pipeline

1. Data processing: Provide usable data for the model and perform necessary preprocessing. Set all the environment, read the wine review dataset, randomize and split into training, validation, and testing sets.

2. Model preparation: Prepare for training and evaluating the model. Define the tokenizer and set the maximum input and target lengths respectively for generating and referencing targets; load the rouge metric standards using rouge_score; load pre-trained models from the mT5 transformers library and set training parameters.

3. Training the model: Train the model and evaluate model performance. Define function to compute metrics for evaluating the quality of summaries generated by the model; group examples from the dataset into batches; instantiate Seq2SeqTrainer and train the model using the train() method.

4. Generating wine variety: Use the pipeline class to generate wine variety and print results on the test set.

5. Evaluating the model: Use the trainer.evaluate() function to calculate the ROUGE metric standards for the model on the test set and store the results in evaluation_results. The purpose of this step is to evaluate the performance of the model and determine whether it can be used in practical applications.

6. Result analysis: Analyze the results in evaluation_results to determine the performance and limitations of the model, which can comprehensively evaluate the performance of the model and provide guidance for improving.

## Demo Link

Try the wine recommendar by yourself: https://colab.research.google.com/drive/1Z3kgRHHv-AkuKPYGkLfBWbfFuSDUAeYh?usp=sharing

## Results

**validation_loss** measures the error of the model during training on a validation set. It is often used as an indicator of how well the model is able to fit the training data. The goal is to **minimize the loss during training**, which usually results in better performance on new data.

**validation_rouge** is a metric used to evaluate the quality of text summarization models. ROUGE stands for "Recall-Oriented Understudy for Gisting Evaluation" and it measures the overlap between the generated summary and the reference summary. Specifically, validation_rouge calculates the F1 score between the generated summary and the reference summary. **A higher validation_rouge score indicates better performance of the model in generating high-quality summaries.**

**mT5-Small (300 million parameters)**
| Metric | Value |
| --- | --- |
| validation_loss | 0.41913044452667236 |
| validation_rouge | 83.9338|

**mT5-Base (580 million parameters)**
| Metric | Value |
| --- | --- |
| validation_loss | 13.628491401672363 |
| validation_rouge | 63.2454 |


## Conclusions

Based on the results, it can be concluded that the mT5-Small model outperforms the mT5-Base model for the wine variety recommendation task. The mT5-Small model has a lower validation loss and a higher validation ROUGE score compared to the mT5-Base model. The lower validation loss indicates that the mT5-Small model is better at fitting the training data, while the higher validation ROUGE score suggests that it generates higher-quality summaries, which in this case means more accurate wine variety recommendations.

There could be several reasons for the difference in performance between the two models. It is possible that the mT5-Base model, despite having more parameters, overfits the training data due to its larger size and complexity. Additionally, the mT5-Small model may be more suitable for the wine recommendation task, as it might better capture the essential information needed to make accurate predictions without being overwhelmed by the large number of parameters.

In conclusion, the mT5-Small model is a better choice for the wine variety recommendation system using the Wine Reviews dataset. However, further experimentation with other models, hyperparameters, and model sizes may still yield better results.

## Critical Analysis

1. One limitation is the reliance on wine descriptions and ratings as the only input features for the model. While these are certainly important factors in wine recommendation, there may be other relevant features that are not captured in the dataset. Incorporating additional features or data sources could potentially improve the accuracy and relevance of the recommendations.

2. Timeliness of the dataset - the information in the dataset is previous released that may not represent current market and consumer preferences. To better match current trends and preferences, it is necessary to collect and use the latest updated information.

3. Due to storage limitations, we did not test the results on all mt5 models, but only selected the small and base models to generate the results. In the future work, if we could use all of the mt5 models, the impact of a larger number of parameters on the results could be analyzed, which would make the experiment more complete.

## Resource Links

[1] mT5 Hugging Face Site: https://huggingface.co/docs/transformers/model_doc/mt5

[2] Phakmongkol, Puri & Vateekul, Peerapon. (2021). Enhance Text-to-Text Transfer Transformer with Generated Questions for Thai Question Answering. Applied Sciences. 11. 10267. 10.3390/app112110267. 

[3] mT5-Small (300 million parameters): [gs://t5-data/pretrained_models/mt5/small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/small)

[4] mT5-Base (580 million parameters): [gs://t5-data/pretrained_models/mt5/base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/base)

[5] mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer： https://arxiv.org/pdf/2010.11934.pdf

[6] mT5: Multilingual T5： https://github.com/google-research/multilingual-t5

## Video Recording
