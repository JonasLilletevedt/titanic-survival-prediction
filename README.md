# Titanic: A End-to-End Machine Learning Workflow

This repository contains a complete, end-to-end machine learning project for the classic Kaggle "Titanic - Machine Learning from Disaster" competition. The project divided into three notebooks, each notebook displaying its own part of the full workflow. From initial data exploration to final model deployment and submission. 

The primary objective is twofold: to build a competitive model for the Kaggle leaderboard, and to create interpretable analysis that explains the data, it perks and how the model uses the data to make predictions. My own objective for making this project was to do something fun and educational for both me and possibly for others. As well as serving as a good portfolio piece.

**Final Kaggle Score: 0.80622** (Top 5-10% of legitimate submissions)

---

## Project Structure

The project is split into three individual notebook, each focusing on a specific part of the process in machine learning. 

### `00_initial_data_exploration.ipynb`
This notebook lays the foundation for the entire project. It is our first interaction with the data, and it is in this section we get to understand the data and its underlying patters. We make hypothesis, identify raw predictive features, together with engineerable features with large potential. This is a crucial part of the process, you will not be able to build a model which stands out without making hypothesis about engineerable features that other might overlook.

**Key Activities:**
*   **Initial Inspection:** Check everything is as expected: Is the data loaded properly? Do the data set include duplicates or impossible values?
*   **Statistical Summaries:** We explore data types,  missing values and key statistical traits.
*   **Exploratory Data Analysis:** 
    *   Univariate and bivariate analysis to understand variables distributions, as well as how they pair up with the target variable.
    *   Correlation heatmaps for numerical features, and bar plots for categorical features.


### `01_data_cleaning_and_feature_engineering`
Building on the insights from the EDA, this notebook constructs an end-to-end preprocessing pipeline. We start out by creating, and testing new feature-engineered features inspired by the previous notebook. Promising new features, are incorporated in the final pipeline, were we encapsulate all logic in custom transformers, for reusability and to prevent data leakage.

**Key Features Engineered:***
*   **Title_feat:** A feature created by extracting the passengers title (e.g., 'Mr' and 'Mrs') from the `Name` column. Turned out be the most predictive feature of all features in the final model.
*   **FamilySurvivalRate_feat:** A sophisticated feature that identifies family groups through surname and `Pclass`. It then calculates a smoothed survival rate for each passenger, excluding the passenger themselves to avoid data leakage.
*   **Zone_feat (vertical) and Deck_feat (horizontal):** Extracts the passengers vertical and horizontal cabin location from `Cabin`. 
*   **Age_binned and Zone_binned:** `Age` and `Zone` are binned to better capture non-linear relationships. 

### `02_modeling.ipynb`
The last and final notebook of this project, we use the pre-build pipeline created in `01_data_cleaning_and_feature_engineering.ipynb` to train, evaluate and optimize a series of models.

**Key Activities:**
1.  **Baseline Evaluation:** We train and evaluate five base models (Logistic Regression, KNN, Random Forest, Gradient Boosting, SVM) with 5-fold cross-validation and **F1-Score** as primary metric.
2.  **Hyperparameter Tuning:** The top two models, SVM and Gradient Boosting were hyperparameter tuned using `GridSearchCV`.
3.  **Explored Possible Gains with XGBoost:** Introduced, trained and tuned `XGBoost Classifier`, which emerged as the best-performing single model.
4.  **Ensembling:** A `Voting Classifier` was used and tested combining `SVM` and `XGBoost`.
5.  **Feature Importance Analysis:** A deep dived into the tuned `XGBoost` model's feature importance to understand it's decision making process.
6.  **Final Kaggle Submission:** The final `XGBoost` model was re-tuned specifically for the Kaggle competition metric (accuracy) instead of F1-Score. This was the final model used for the submission.