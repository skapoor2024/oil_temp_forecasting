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

## Outcome

The project resulted in significant advancements in predicting electric consumption using oil temperature data, with various models demonstrating robust performance. The best learning rate (η) for the Linear Mean Square (LMS) model with MSE was 0.003, while for MCC, it was 0.0007 with a kernel size (σ) of 0.25. Both models converged to similar weight values, suggesting their robustness. Interestingly, certain weight tracks (2 & 7, 3 & 6, and 4 & 5) converged to the same values, indicating potential periodicity between these time steps. In the test set, the LMS model with MCC outperformed the one with MSE.

For the Kernel LMS (KLMS) model with MSE, the optimal step size (η) was 0.109 with a kernel size (h) of 0.922. When training with MCC, the best parameters were a kernel size (σ) of 0.8, a kernel size (h) of 1.4, and a step size of 0.02. Extensive parameter sweeps were conducted, generating heatmaps to identify the best configurations. For QKLMS, the optimal kernel size (h) was 0.4, step size (η) was 0.9, and threshold (ε) was 0.2, while for MCC, the best σ was 2.

Evaluating these models on the ETTm2 time series dataset, which contains two years of hourly data (approximately 17,000 samples), showed that the LMS model with MCC had the best overall performance across training, validation, and test sets. Multi-step predictions were used to evaluate overfitting and generalization. LMS models highlighted the importance of n-1 and n-2 weights, leading to larger MSE or MCC errors due to accumulated positive or negative errors. In contrast, KLMS models, with their more complex networks, generalized the input distribution better, resulting in lower errors.

Overall, models trained using MCC outperformed those using traditional MSE by an average of 25%. These outcomes indicate that leveraging advanced filtering techniques and optimization methods can significantly improve the accuracy of electric consumption forecasts, contributing to more efficient and wise distribution of electricity in response to fluctuating energy prices.
