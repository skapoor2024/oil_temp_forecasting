# Oil Temperature Forecasting 

## Problem and Goal 

With the ongoing rapid increase in global energy prices, accurately predicting future electric consumption has become increasingly critical. One potential method for predicting future load involves using oil temperature, a key component in transformers. Due to its strong correlation properties, oil temperature can provide valuable insights into future load prediction.

Our project aimed to compare the effectiveness of linear and non-linear adaptive filters trained on oil temperature data from the ETTm2 time series dataset in accurately predicting the next time step of electric consumption. We evaluated these models using both Mean Square Error (MSE) and Maximum Correlation Criterion (MCC) cost functions to enhance their predictive accuracy.

Additionally, we analyzed the models' capacity to predict multiple future time steps with a standard deviation of 0.2. While linear filtering showed strong performance for single-step predictions, our findings revealed that the non-linear filter, specifically the Kernel Least Mean Squares (KLMS) filter, outperformed the linear filter in generating better trajectory predictions and multi-step forecasts.

## Process and Solution 

Our approach to predicting electric consumption from oil temperature data involved a two-step process: filtering and evaluation.

### Filtering:

#### KLMS Filter: 
We employed Kernel Least Mean Squares (KLMS) filtering, a non-linear adaptive filtering technique. KLMS works by mapping the input data to a higher-dimensional space using a kernel function. This allows the model to capture more complex relationships between oil temperature and electricity consumption compared to traditional linear methods.

#### Memory Management with QKLMS:  
To efficiently manage memory usage, we incorporated Quantized KLMS (QKLMS). QKLMS avoids storing redundant data by only adding new data points as centroids if they significantly differ from existing ones. This ensures the model focuses on informative data for improved prediction accuracy.

#### Loss Functions: 
We utilized two loss functions: Mean Squared Error (MSE) and Maximum Correlation Criterion (MCC).  MSE is a traditional measure of error, while MCC is an information-theoretic approach that aims to minimize the loss in a high-dimensional space. Combining these functions provided a robust framework for optimizing the model's predictions.

### Evaluation:

To assess the effectiveness of our approach, we used the ETTm2 dataset containing two years of hourly data (around 17,000 samples). We divided this data into training (10,000 samples), validation (1,000 samples), and testing sets (6,000 samples). This allowed us to train the model, fine-tune its performance based on the validation set, and finally evaluate its generalizability using unseen test data.

### Solution:

This process resulted in a robust model capable of accurate electric consumption prediction using oil temperature data. Evaluations demonstrated that while linear filtering provided good single-step predictions, the non-linear KLMS filter, particularly with the memory management benefits of QKLMS, offered superior performance in generating multi-step forecasts and capturing the overall trajectory of electricity consumption.
