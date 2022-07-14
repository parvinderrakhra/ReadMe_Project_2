# Project 2 - Algo - Trading Bot

![AlgoTrading-challenge-image](https://user-images.githubusercontent.com/85688247/178852925-0a297d62-e9f3-426a-b95c-4a4be1cf1905.png)


In our project we created a Stock Market trading algorithm, using Tesla (TSLA) stock as a baseline. We createdpredicitbve models using 3 techinical indicators (Simple Moving Average, Bolinger Band and Exponential Moving Average) over 5 years of Tesla stock market data. We then then evaluated the performance of the models to determine which had the best predictive power to maximise potential portfolio value. 


## Overview

* We are interested in stock market trading algotrithms

* We used DataReader and yfinance (Yahoo Finance) to obtain our stock market data (TSLA)

* Machine learning models we expect to utilize are:

    * Random Forest
    * Support Vector Classifier (SVC)
    * LSTM

* Technical Indicators used:

    * Moving Average of Closing Prices
    * Exponential Moving Average of Closing Prices
    * Exponential Moving Average of Daily Return Volatility
    * Bollinger Band
    * Buy and Hold


### Libraries Used:
![Library](https://user-images.githubusercontent.com/85688247/178892027-4e999ab8-ecd6-4400-b3d3-454b52b73383.png)


### Data Import:
![Import_Data](https://user-images.githubusercontent.com/85688247/178892633-2df7840a-1370-4abd-9704-45dad4ca18e1.png)


## Model Evaluation
* Ending portfolio values/returns under each machine learning model will be used to evaluate the models

We wanted to create a number of machine learning models - to begin we simply examined the ‘Strategy Returns’ performance of Tesla stocks - based on different technical indicators. 


### Support Vector Machine (SVM)

the first machine learning model evaluated was Support Vector Machine (SVM).  

In machine learning, SVMs are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.  Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.  An SVM maps training examples to points in space so as to maximise the width of the gap between the two categories.  New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

Using the same set of Tesla stock price data as the prior two models, we then determine the correct trading signal to determine when to buy or sell Tesla's stock.  We will store +1 for buy signal and -1 for sell signal in the Signal column. 'y' is a target dataset storing the correct trading signal which the machine learning algorithm will try to predict.  The X is a dataset that holds the variables which are used to predict y, that is, whether Apple stock price will go up (1) or go down (-1) tomorrow. The X consists of variables such as 'Open - Close', 'High - Low' and 'Volume'. These can be understood as indicators based on which the algorithm will predict tomorrow's trend.


![Trading Signals](https://user-images.githubusercontent.com/85688247/178893910-4838b582-9c57-4512-b69f-b38f6983e597.png)


We then acalculated Actual Returns using percentage change and Strategy Returns using Actual Returns and Simple moving average and added to the dataframe. 
![Strategy   Actual Return ](https://user-images.githubusercontent.com/85688247/178894384-37b62a82-21af-4b11-bc98-de9f3db6594a.png)


#### Strategy Return based on SMA
![z_1 - Plot SMA Strt Return](https://user-images.githubusercontent.com/85688247/178895108-104d1d08-a07a-4375-8daa-a2d98e628a5b.png)


We then split the Data into training and testing
![z-2 SMA Training and Testing](https://user-images.githubusercontent.com/85688247/178898528-7c21e1d2-1a92-40c4-b69b-7b05d90c4d3f.png)

The classification report assiciated with SVC model predictions
![Z3 SVC Model Prediction](https://user-images.githubusercontent.com/85688247/178899176-7f635791-161f-43f0-8d46-fb0d6f73df04.png)

![Z4 SVC Actual vs Predicted Returns ](https://user-images.githubusercontent.com/85688247/178899470-656e0027-c800-4e4f-b032-b32b15d8e6b7.png)

We then backtested the model to evaluate performance
![z5 - backtest](https://user-images.githubusercontent.com/85688247/178899643-1063b1f9-ed0e-4d04-abba-b07776605a82.png)

#### Strategy Return based on EMA
![EMA50 100](https://user-images.githubusercontent.com/85688247/178901807-4177fff0-46b8-4d9b-ae48-b7b68a7cfa6b.png)

![EMA50 100_2](https://user-images.githubusercontent.com/85688247/178901977-3160bee5-50c6-4031-a75a-3f7db7cb1416.png)

#### Strategy Return based on Bollinger Band
![TSLA_BBAnd](https://user-images.githubusercontent.com/85688247/178901252-b3f6a050-11ef-4382-91b5-d651bf31b95e.png)


### Random Forest Trading
With the Random Forest, we create a trading signals CSV file based on (1) Simple Moving Average of Closing Prices (2) Exponential Moving Average of Closing Prices, and (3) Bollinger Bands. These signals are expressed as 1, or -1 to Buy or Sell respectively.

#### Strategy Return based on SMA

#### Strategy Return based on EMA

#### Strategy Return based on Bollinger Band


### Long short-term memory (LSTM)

One of the deep learning model we used in our project is Long short-term memory (LSTM) model. LTSM is an artificial recurrent neural network (RNN) architecture and is well-suited to classifying, processing, and making predictions based on time series data. We used LSTM neural network to analyse short term trade scenarios i.e. to decide whether to stay in the market or not. The LSTM data was captured in a particular shape involving "windows" and at each step we predicted the closing price of the day.  This helped us to find the vector v which identifies the days during which we are going to stay in the market. For this model, we again use the same data to run our model. 

#### Strategy Return based on SMA

![lstm 1](https://user-images.githubusercontent.com/85688247/178902377-338fcb81-7b81-4f5c-b72b-54b0c5c9f065.png)

Now for each day we have the closing price for the day, the open price for the day (proxy for the closing price of the previous day) and the open price of the following day. From this data we derive the feature RaPP (Reconstruction along Projection Pathway), which is the quotient between the open and closing prices of the day. It is used to give us the variation of the portfolio for the day. The gross yield was computed using RaPP.

![lstm 2](https://user-images.githubusercontent.com/85688247/178902574-03b8216a-a6cd-40c7-8fe8-ea344d27ebe0.png)

To initial dataframe we added SMA for 5 and 50 day, and removed the first 50 days since they didn’t have the 50 days moving average.  
![lstm 3](https://user-images.githubusercontent.com/85688247/178902869-578e3fe3-7bc3-4a4b-9914-4748164cf578.png)

The dataframe was then used to Train and Test sets (70/30) and defined the LTSM model. We then run the model with 100 epochs and a batch size of 30. Other epoch and batch size amounts were used but the significant increase in time was not worth the small incremental decrease in 'val_loss'.

![lstm 4](https://user-images.githubusercontent.com/85688247/178903067-c3d15cb7-d80a-40d4-967d-0593d342440b.png)

![lstm 5](https://user-images.githubusercontent.com/85688247/178903178-ba660522-f1fe-4e14-b087-943384caa343.png)

![lstm 6](https://user-images.githubusercontent.com/85688247/178903292-bf5c6031-1460-4ff2-9420-5266251c70c1.png)

![lstm 7](https://user-images.githubusercontent.com/85688247/178903401-8bbe49a2-f695-41c2-9004-e4fa736278f1.png)


