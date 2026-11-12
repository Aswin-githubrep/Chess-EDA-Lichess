# Chess EDA & Machine Learning – Lichess Games ♟️

An exploratory data analysis and machine learning project using **20,058 online chess games from Lichess**.

This project investigates player ratings, game outcomes, opening patterns, and game characteristics. It then applies Logistic Regression and Gaussian Naive Bayes to classify decisive game outcomes.

### Project Highlights

- **20,058 chess games** explored through EDA and visualizations.
- **Rating difference** has the largest coefficient magnitude in the fitted standardized Logistic Regression model.
- Both classification models achieve approximately **65% accuracy**, compared with a **52.34% majority-class baseline**.
- Gaussian Naive Bayes achieves the highest test accuracy at **65.23%**.

---

## 1. Research Question

**Can selected game characteristics, such as player rating difference, rated-game status, and opening length, help predict whether White or Black wins a decisive chess game?**

The project follows an end-to-end data analysis workflow:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and selection
- Machine learning model training
- Model evaluation and comparison

**Prediction scope:** Draws are excluded from the modelling stage to formulate the task as binary classification. The models predict only White or Black wins, not draws. Since `opening_ply` is derived from the completed game, this experiment is not a pre-game prediction system.

## 2. Dataset

**Dataset:** Chess Game Dataset (Lichess)

**Source:** [Kaggle – Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess)

The dataset contains 20,058 chess games collected from Lichess.org, with information about players, ratings, game outcomes, openings, and moves.

### Dataset Overview

| Attribute | Details |
|---|---:|
| Total games | 20,058 |
| Original features | 16 |
| Rated games | 16,155 |
| Unrated games | 3,903 |
| White wins | 10,001 |
| Black wins | 9,107 |
| Draws | 950 |

### Important Features

| Feature | Description |
|---|---|
| `rated` | Whether the game was rated |
| `winner` | Game outcome: White, Black, or Draw |
| `white_rating` | White player's rating |
| `black_rating` | Black player's rating |
| `turns` | Number of turns in the game |
| `victory_status` | How the game ended |
| `opening_name` | Chess opening played |
| `opening_ply` | Number of opening moves |

The dataset is loaded from `Data/games.csv`.

## 3. Exploratory Data Analysis – Key Findings

The notebook explores game outcomes, player ratings, victory methods, opening popularity, game length, rating differences, and monthly game activity.

### Finding 1: Distribution of Game Outcomes

| Outcome | Games | Percentage |
|---|---:|---:|
| White wins | 10,001 | 49.86% |
| Black wins | 9,107 | 45.40% |
| Draws | 950 | 4.74% |

White wins are slightly more frequent than Black wins in the original dataset.

### Finding 2: Player Rating Distribution

The rating distributions of White and Black players are approximately bell-shaped and largely overlap.

This suggests broadly similar rating distributions across the two player positions in this dataset.

### Finding 3: Rating Difference and Game Outcome

A new feature was created:

`rating_diff = white_rating - black_rating`

The analysis suggests an association between rating difference and game outcome:

- Positive rating differences are more commonly associated with White wins.
- Negative rating differences are more commonly associated with Black wins.

For games where the absolute rating difference is less than 50 points:

| Outcome | Proportion |
|---|---:|
| White wins | 48.78% |
| Black wins | 45.85% |
| Draws | 5.37% |

### Finding 4: Game Characteristics

- **Victory methods:** Resignation and checkmate account for most game endings, with timeouts also representing a notable category.
- **Opening popularity:** The ten most frequently occurring chess openings are visualized to examine opening patterns.
- **Game length:** Box plots compare the number of turns across victory methods and game outcomes.
- **Monthly activity:** A time-series plot explores how the number of games in the dataset varies across months.

These visualizations provide additional context for understanding the dataset before modelling.

## 4. Data Preprocessing and Feature Engineering

The following preprocessing steps were performed:

1. Removed irrelevant identifiers, timestamps, player IDs, move notation, and other unused columns.
2. Converted the target variable into numerical labels:
   - `0` – Black wins
   - `1` – White wins
3. Excluded draws to formulate a binary classification problem.
4. Converted the `rated` feature into an integer representation.
5. Created the `rating_diff` feature using the difference between the two player ratings.
6. Removed `white_rating` and `black_rating` from the final feature set to reduce redundancy with `rating_diff`.

### Feature Selection Experiment

The initial feature set contained `white_rating`, `black_rating`, and `rating_diff`.

Since `rating_diff` is directly calculated from the two individual player ratings, these features contain redundant information.

After removing the two individual rating columns, the reported results were:

| Model | Before | After |
|---|---:|---:|
| Logistic Regression | 64.76% | 64.78% |
| Gaussian Naive Bayes | 64.91% | 65.23% |

Removing the redundant rating columns simplified the feature set while maintaining or slightly improving the observed accuracy in this experiment.

### Final Features

| Feature | Description |
|---|---|
| `rated` | Whether the game was rated |
| `opening_ply` | Number of opening moves |
| `rating_diff` | Difference between White and Black player ratings |

The final modelling dataset contains three input features and one binary target variable.

## 5. Machine Learning Models

Two classification algorithms were trained and evaluated.

### A. Logistic Regression

- Implemented using Scikit-learn.
- Feature standardization performed using training-set statistics.
- Maximum iterations set to 1,000.
- Evaluated using accuracy, classification report, and confusion matrix.

### B. Gaussian Naive Bayes

- Implemented using Scikit-learn's `GaussianNB`.
- Trained using the original unscaled features.
- Evaluated using accuracy, classification report, and confusion matrix.

### Train-Test Split

- Training set: 80%
- Testing set: 20%
- Random state: 42
- Stratified splitting to preserve class proportions.

## 6. Model Results and Comparison

A majority-class baseline was calculated using the training labels. This baseline predicts the most frequent class for every test sample.

### Accuracy Comparison

| Model | Accuracy |
|---|---:|
| Majority-Class Baseline | 52.34% |
| Logistic Regression | 64.78% |
| Gaussian Naive Bayes | 65.23% |

### Improvement Over Baseline

| Model | Improvement |
|---|---:|
| Logistic Regression | +12.44 percentage points |
| Gaussian Naive Bayes | +12.89 percentage points |

### Logistic Regression – Standardized Coefficients

| Feature | Coefficient |
|---|---:|
| `rating_diff` | 0.9689 |
| `opening_ply` | 0.0282 |
| `rated` | 0.0110 |

In the fitted standardized Logistic Regression model, `rating_diff` has a substantially larger coefficient magnitude than the other two features. This suggests that rating difference has the strongest association with the predicted outcome within this fitted model.

These coefficients should not be interpreted as universal feature-importance scores or as evidence that the other features have no predictive value.

### Classification Report Summary

| Model | Class | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| Logistic Regression | Black | 0.65 | 0.58 | 0.61 |
| Logistic Regression | White | 0.65 | 0.71 | 0.68 |
| Gaussian Naive Bayes | Black | 0.65 | 0.58 | 0.62 |
| Gaussian Naive Bayes | White | 0.65 | 0.72 | 0.68 |

### Key Observations

- Both models outperform the 52.34% majority-class baseline.
- Logistic Regression achieves 64.78% accuracy.
- Gaussian Naive Bayes achieves 65.23% accuracy.
- Gaussian Naive Bayes performs slightly better on this particular train-test split.
- The difference between the models is small and should not be interpreted as evidence that one algorithm is consistently superior.

## 7. Technologies and Libraries

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 8. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/Aswin-githubrep/Chess-EDA-Lichess.git
cd Chess-EDA-Lichess
```

### Step 2: Download the dataset

Download the dataset from:

[Chess Game Dataset – Kaggle](https://www.kaggle.com/datasets/datasnaek/chess)

Extract the CSV file and place it inside the `Data` folder with the filename:

```text
Data/games.csv
```

Expected project structure:

```text
Chess-EDA-Lichess/
├── README.md
├── Chess_EDA_Lichess.ipynb
└── Data/
    └── games.csv
```

### Step 3: Install dependencies

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Chess_EDA_Lichess.ipynb` and execute the cells sequentially.

## 9. Limitations and Future Improvements

- Draw outcomes are excluded, so the current models cannot predict draws.
- The models use only three selected features.
- Results are based on one stratified train-test split.
- The small accuracy difference between the two models may be due to variation in the particular split.
- The current experiment uses completed-game metadata, including `opening_ply`, so it is not a pre-game prediction system.

Possible future improvements:

- Experiment with a pre-game prediction model using features available before a game starts, such as `rating_diff` and `rated`, excluding `opening_ply`.
- Evaluate models using cross-validation.
- Explore additional features and feature interactions.
- Experiment with Decision Trees, Random Forests, and other classification algorithms.
- Tune model hyperparameters and compare performance across multiple splits.

## 10. Conclusion

This project combines exploratory data analysis with supervised machine learning to investigate patterns in online chess games.

The analysis highlights the association between player rating differences and game outcomes. Logistic Regression and Gaussian Naive Bayes both outperform the majority-class baseline, demonstrating that the selected features contain useful predictive information for distinguishing decisive White and Black wins.

The project helped strengthen practical skills in data exploration, visualization, feature engineering, classification, and model evaluation.

---

**Project developed as part of my continuous learning journey in Data Science and Machine Learning.**
