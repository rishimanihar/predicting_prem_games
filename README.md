# Premier League Match Predictor

## Overview

This data science project aims to predict the outcome of Premier League football matches using machine learning. The model is built on historical match data, which has been cleaned, processed, and used to train a `RandomForestClassifier`. The goal is to predict whether a match will be a win, loss, or draw for a given team.

The project uses a structured workflow, starting with exploratory data analysis in a Jupyter Notebook and concluding with a predictive model.

---

## Data

The project uses the `matches.csv` and `E0 (1).csv`/`E0 (2).csv` files, which contain historical Premier League game data. The `matches.csv` file provides per-match statistics like goals scored (`gf`), goals allowed (`ga`), shots (`sh`), and shots on target (`sot`). The odds data is sourced from two separate files that were combined for the full dataset.

The data was preprocessed to include features like `venue_code`, `opp_code`, `hour`, and `day_code`. I also created rolling averages for several key statistics (`gf_rolling`, `ga_rolling`, etc.) to use as predictors, which capture a team's recent form.

---

## Methodology

The core of the project is a machine learning model built with `scikit-learn`.

1. **Data Preparation**: The raw data was cleaned and new features were engineered to provide more predictive power to the model. Categorical variables like team and opponent names were encoded into numerical representations.

2. **Feature Engineering**: A key step was calculating rolling averages of performance metrics (goals scored, shots, etc.) over the last 3 games for each team. This provides the model with context about a team's recent form, which is a strong predictor of future performance.

3. **Model Training**: A `RandomForestClassifier` was chosen for its robustness and ability to handle complex feature interactions. The model was trained on data up to a certain point in the season (`2022-01-01`).

4. **Evaluation**: The model's performance was evaluated on a test set (data from `2022-01-01` onwards). The `accuracy_score` and a confusion matrix were used to assess how well the model predicts wins, losses, and draws.

---

## Results

The model achieved a precision score of approximately **0.39**, which is a solid starting point for this kind of problem. The confusion matrix shows how the model's predictions align with the actual outcomes:

|              | Predicted Draw | Predicted Loss | Predicted Win |
| :----------- | :------------- | :------------- | :------------ |
| **Actual Draw** | 13             | 43             | 0             |
| **Actual Loss** | 27             | 56             | 36            |
| **Actual Win** | 16             | 20             | 71            |

*Note: The actual output from the confusion matrix shows a precision score of 0.40, likely due to rounding in the notebook output.*

---

## Dependencies

This project requires Python 3.x and the following libraries:
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`

You can install these dependencies using `pip`:
`pip install -r requirements.txt`
