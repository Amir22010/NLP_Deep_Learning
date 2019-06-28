# Natural Language Processing Text Sentiment Analysis using Deep Learning Framework - Pytorch

## Task - Training a long short term memory recurrent network on a dataset of labeled movie reviews to learn how to predict whether some text is either positive or negative.

## The Process Flow 

- Collect text data (Movie reviews).

- Convert input review text into encoded vectors using an embedding layer.

- Input these word embeddings into the LSTM cells (allow for sequence learning).

- Pass LSTM outputs to a sigmoid output layer (positive or negative).

- Compute loss by comparing LSTM output to actual label, use calculus to optimize the network during training.


# Deep Learning Sentiment Analysis Process Flow 

- Import Data

- Data pre-processing

- Encoding the labels

- Removing Outliers

- Padding sequences

- Training, Validation, Test

- DataLoaders and Batching

## Sentiment Network with PyTorch

- The Embedding Layer

- The LSTM Layer(s)

- Instantiate the network

- Training

- Testing

## Notebook

-[Notebook](https://colab.research.google.com/drive/1XYJzL9ZmrGN44ftM-211259_05tLMbS_)

## Results

![UI_Image](https://raw.githubusercontent.com/Amir22010/NLP_Deep_Learning/master/7_June_2019_NLP_Deep_Sentiment/Image1.JPG)

![Result](https://raw.githubusercontent.com/Amir22010/NLP_Deep_Learning/master/7_June_2019_NLP_Deep_Sentiment/Image2.JPG)
