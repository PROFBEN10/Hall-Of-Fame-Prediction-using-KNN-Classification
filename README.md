### Hall of Fame Prediction Using K-Nearest Neighbors (KNN)

### Project Overview

This project applies the K-Nearest Neighbors (KNN) algorithm to predict the likelihood of a baseball player being inducted into the Hall of Fame (HOF) based on performance metrics and career statistics.
It demonstrates data preprocessing, model selection, hyperparameter tuning, and performance evaluation using Python’s scikit-learn library.

### Objectives

Build a predictive model that classifies players as HOF or Non-HOF.

Identify the optimal K-value for the KNN classifier.

Evaluate model performance using accuracy, precision, recall, and F1-score.

Visualize how the choice of K affects model accuracy and generalization.

### Dataset Description

The dataset includes baseball player statistics and Hall of Fame status.
Each record represents a player with several key attributes (features) and a target label.

### Column	Description
| **Column Name** | **Description**                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `Player`        | Name of the player being evaluated.                                                               |
| `Position`      | The player's primary playing position (e.g., Guard, Forward, Center).                             |
| `Points`        | Total points scored by the player.                                                                |
| `Rebounds`      | Number of rebounds secured by the player.                                                         |
| `Assists`       | Number of assists made by the player.                                                             |
| `Steals`        | Number of steals recorded.                                                                        |
| `Blocks`        | Number of shots blocked by the player.                                                            |
| `Turnovers`     | Number of times the player lost possession of the ball.                                           |
| `Games_Played`  | Total games the player participated in.                                                           |
| `Hall_of_Fame`  | Target variable — 1 indicates the player is inducted into the Hall of Fame, 0 means not inducted. |




### Methodology

1. Data Preprocessing

Scaled numerical data using StandardScaler to ensure fair distance-based comparisons.

2. Model Implementation

Split the dataset into training (80%) and testing (20%) sets.

Trained the model using multiple K values (k = 1–20) to find the most effective neighbor count.

Evaluated model performance across K values to prevent overfitting or underfitting.

3. Model Evaluation

Metrics used:

Accuracy Score

Precision

Recall

F1-Score

Confusion Matrix Visualization

Classification report

The model achieved an optimal performance at K=5 with 92% accuracy, and an alternative test at K=4 achieved 87% accuracy, showing robust predictive strength and generalization.

### Results & Insights

The KNN model with K=5 yielded the highest accuracy (0.92).

Lower values of K (e.g., K=4) maintained good performance (0.87), confirming model stability.

Accuracy vs. K visualization helped identify the sweet spot for neighbor count.

The model effectively distinguishes Hall of Fame players based on key stats.

### Technologies Used

Python 3.12+

Libraries:

pandas

numpy

scikit-learn

seaborn


### Future Improvements

Apply cross-validation to enhance robustness.

Experiment with feature selection or dimensionality reduction (PCA).

Compare KNN performance with Logistic Regression, Random Forest, or SVM.

Deploy as a simple Streamlit web app for interactive prediction.
