# Oil Temperature Forecasting 

## Problem and Goal 

With the ongoing rapid increase in global energy prices, accurately predicting future electric consumption has become increasingly critical. One potential method for predicting future load involves using oil temperature, a key component in transformers. Due to its strong correlation properties, oil temperature can provide valuable insights into future load prediction.

Our project aimed to compare the effectiveness of linear and non-linear adaptive filters trained on oil temperature data from the ETTm2 time series dataset in accurately predicting the next time step of electric consumption. We evaluated these models using both Mean Square Error (MSE) and Maximum Correlation Criterion (MCC) cost functions to enhance their predictive accuracy.

Additionally, we analyzed the models' capacity to predict multiple future time steps with a standard deviation of 0.2. While linear filtering showed strong performance for single-step predictions, our findings revealed that the non-linear filter, specifically the Kernel Least Mean Squares (KLMS) filter, outperformed the linear filter in generating better trajectory predictions and multi-step forecasts.

The goal of this project was to leverage advanced prediction tools to improve the accuracy of electric consumption forecasts, enabling more efficient and wise distribution of electricity in response to fluctuating energy prices.
