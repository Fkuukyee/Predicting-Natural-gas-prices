# Predicting Natural Gas Prices in the Canadian Energy sector using Machine Learning
## Problem Statement
Natural gas holds a position of strategic importance in the Canadian energy sector. Therefore, understanding and predicting natural gas prices is crucial for both natural gas producers and consumers. The purpose of this project is to utilize machine learning (ML) techniques to predict natural gas prices providing insights, for decision making in the energy sector.

To achieve this goal I utilized a real world dataset that contains information on energy commodity prices such as 'Home Heating Oil', 'Diesel','Propane','Gasoline', 'Household heating fuel', 'Diesel_fuel','unleaded gasoline', 'Closing inventory', 'Heating Degree Days'. The dataset, was obtained from the website of the Canadian Natural Gas Association[https://www.cga.ca/natural-gas-statistics/]

The primary objective of this project is to develop a model capable of accurately forecasting natural gas prices using energy commodity prices.

Not only will this project contribute to the field of energy economics but to showcase how data science and machine learning can be practically applied to address real world challenges.

## Approach this Machine Learning Problem
- Explore the data and find correlations between inputs and targets
- Pick the right model, loss functions and optimizer for the problem at hand
- Scale numeric variables and one-hot encoding of categorical variables
- Set aside a test set (using a fraction of the training set)
- Train the model
- Make predictions on the test set and compute the loss
- Optimize the model using the hyperparameter tuning

## Exploratory Data Analysis
### Visualize the distributions of the numeric columns
![hist_png](https://github.com/Fkuukyee/Predicting-Natural-gas-prices/assets/147086232/b875aada-d00b-4381-813f-983569e44c2d)

### boxplot to identify and treat outliers

![box](https://github.com/Fkuukyee/Predicting-Natural-gas-prices/assets/147086232/bcd19504-47f9-4ce5-9fec-77bf71006b9a)

### Correlation with heatmap to understand relation 
![heatP](https://github.com/Fkuukyee/Predicting-Natural-gas-prices/assets/147086232/13a5a0a8-f168-4388-80a9-9e5d31313e97)

## Linear Regression Algorithm
```
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
import numpy as np
from sklearn.preprocessing import StandardScaler

# Scaling up the columns
numeric_cols = ['Home_Heating_Oil', 'Diesel','Propane',
       'Gasoline', 'Household_heating_fuel', 'Diesel_fuel',
       'unleaded_gasoline', 'Closing_inventory', 'Heating_Degree_Days'] 
scaler = StandardScaler()
scaler.fit(df[numeric_cols])
scaled_inputs = scaler.transform(df[numeric_cols])

#Create inputs and targets
scaled_inputs = ['Home_Heating_Oil', 'Diesel','Propane',
       'Gasoline', 'Household_heating_fuel', 'Diesel_fuel',
       'unleaded_gasoline', 'Closing_inventory', 'Heating_Degree_Days']
inputs, targets = df[scaled_inputs], df['Natural_gas']
# Split data into training and test data
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)

# Create and train the model
model = LinearRegression().fit(inputs_train, targets_train)

# Generate predictions 
predictions_test = model.predict(inputs_test)

# Compute loss to evalute the model
r2 = r2_score(targets_test, predictions_test)
loss = np.sqrt(mean_squared_error(targets_test, predictions_test))
print(f'Test Loss: {loss:.2f}')
```
loss of 0.55 indicates an average deviation of the model's predictions from the actual values.
r2 of 0.60 means the model explains 60% of the variance in the response variable.

## Random Forest Algorithm
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Random Forest Regressor
random_forest_model = RandomForestRegressor(random_state=0)

# Train the model
random_forest_model.fit(inputs_train, targets_train)

# Make predictions on the test set
targets_pred = random_forest_model.predict(inputs_test)

# Evaluate the model
Loss=np.sqrt(mean_squared_error(targets_test, targets_pred))
r2 = r2_score(targets_test, targets_pred)

print(f"{loss:.2f}, {r2:.2f}")
```
loss of 0.55 indicates an average deviation of the model's predictions from the actual values. r2 of 0.60 means the model explains 60% of the variance in the response variable.

### Hyperparameter Tuning of the model

```
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(inputs_train, targets_train)

# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
random_forest_model_optimized = RandomForestRegressor(**best_params, random_state=0)
random_forest_model_optimized.fit(inputs_train, targets_train)

# Make predictions and evaluate
targets_pred_optimized = random_forest_model_optimized.predict(inputs_test)
Loss_optimized = np.sqrt(mean_squared_error(targets_test, targets_pred_optimized))
r2_optimized = r2_score(targets_test, targets_pred_optimized)

print(f"Optimized Loss: {Loss_optimized:.2f}, Optimized R2: {r2_optimized:.2f}")

```
Optimized Loss: 0.44, Optimized R2: 0.46

## Summary
Both models aim to predict Natural_gas prices based on the various fuel-related features. The Linear Regression provides a baseline model with moderate accuracy. The Random Forest, particularly after optimization, shows improvement in prediction accuracy and loss reduction. The optimized Random Forest model demonstrates the effectiveness of hyperparameter tuning in enhancing model performance, albeit with a slightly lower R² score compared to the initial Random Forest model. This indicates a trade-off between reducing the prediction error and the variance explained by the model

