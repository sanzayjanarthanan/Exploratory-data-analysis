# Task 30 – Exploratory Data Analysis (Titanic Dataset)

**Veda Technology · AI & ML Track · Level 1 · Day 30**

## Objective
Perform basic exploratory data analysis (EDA) using Pandas and Matplotlib and identify meaningful patterns in the data, combining data exploration, cleaning, statistics and visualization into one EDA workflow.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn (only to load the dataset)
- Jupyter Notebook

## Dataset
Titanic dataset (`sns.load_dataset("titanic")`) – 891 passengers, 15 columns. The untouched copy is saved as `titanic_original.csv`.

## Files in this Repository
| File | Description |
|------|-------------|
| `eda_titanic.ipynb` | Complete EDA notebook |
| `titanic_original.csv` | Original dataset |
| `titanic_cleaned.csv` | Cleaned dataset used for analysis |
| `plot1_survival_sex.png` | Survival rate by sex |
| `plot2_survival_class.png` | Survival rate by passenger class |
| `plot3_age_hist.png` | Age distribution |
| `plot4_fare_boxplot.png` | Fare by passenger class |
| `README.md` | Workflow, visualizations and observations |

## EDA Workflow

### Step 1 – Understand the Dataset Structure
Checked with `df.shape`, `df.info()`, `df.describe()`. The dataset has 891 rows and 15 columns, including `survived`, `pclass`, `sex`, `age`, `fare` and `embarked`.

### Step 2 – Check Missing Values and Duplicates

| Column | Missing Count |
|--------|---------------|
| age | 177 |
| embarked | 2 |
| embark_town | 2 |
| deck | 688 (~77%) |

`df.duplicated().sum()` reported 107 duplicate rows.

### Step 3 – Clean the Data
```python
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df = df.drop(columns=["deck"])
```
- **age → median** (skewed numerical column), **embarked / embark_town → mode** (categorical, 2 values), **deck → dropped** (~77% missing).
- **Duplicate rows were kept.** This dataset has no passenger ID, so rows that look identical (same class, age, fare, etc.) may be different people. Dropping them would remove real passengers and change the results (the survival rate would move from 38.4% to 41.3%). The cleaned dataset therefore keeps all 891 rows and 14 columns.

### Step 4 – Statistics
Overall survival rate: **38.4%**. Survival rate by group:

| Group | Survival rate |
|-------|---------------|
| Female | 74.2% |
| Male | 18.9% |
| 1st class | 63.0% |
| 2nd class | 47.3% |
| 3rd class | 24.2% |
| Embarked at C | 55.4% |
| Embarked at Q | 39.0% |
| Embarked at S | 33.9% |
| Travelling alone | 30.4% |
| Travelling with family | 50.6% |

Age: mean 29.4, median 28, range 0.4 to 80. Fare: mean 32.2, median 14.5, maximum 512.3.

### Step 5 – Visualizations

**1. Survival rate by sex (bar chart)**

![Survival by sex](plot1_survival_sex.png)

**2. Survival rate by passenger class (bar chart)**

![Survival by class](plot2_survival_class.png)

**3. Age distribution (histogram)**

![Age histogram](plot3_age_hist.png)

**4. Fare by passenger class (box plot)**

![Fare box plot](plot4_fare_boxplot.png)

Chart types were chosen by variable type: bar charts for category-wise rates, a histogram for the distribution of a numerical variable, and a box plot to compare spread and outliers across groups.

## Observations
1. **Most passengers did not survive.** The overall survival rate was 38.4%.
2. **Sex was the strongest factor.** 74.2% of women survived compared with 18.9% of men, about four times higher.
3. **Survival dropped with passenger class.** 63.0% in 1st class, 47.3% in 2nd class and 24.2% in 3rd class.
4. **Children had the best chances, the elderly the worst.** Passengers aged 12 or below had a 58.0% survival rate (69 passengers), while those above 60 had 22.7% (22 passengers). The median age was 28, so young adults formed the largest group.
5. **Fare is heavily right-skewed and linked to class.** Skewness is 4.79, with a median fare of 14.5 but a maximum of 512.3. Median fares were 60.3 (1st class), 14.3 (2nd) and 8.05 (3rd). Fare has a positive correlation with survival (0.26), which largely reflects passenger class.
6. **Travelling with family helped.** Passengers travelling alone had a 30.4% survival rate compared with 50.6% for those with family.

*These are patterns in the data, not proof of cause.*

## Key Learnings
- EDA should follow a clear order: structure, missing values and duplicates, cleaning, statistics, visualization, observations.
- Always support an observation with a number from the data.
- Cleaning decisions affect the results. Dropping apparent duplicates without a unique ID would have changed the survival rate.

## Interview Questions – Key Answers
1. **What is EDA?** The process of exploring a dataset with summary statistics and visualizations to understand its structure, patterns, problems and relationships.
2. **Why is EDA performed before machine learning?** It reveals data quality issues, outliers, useful features and relationships, and helps avoid wrong assumptions before modelling.
3. **What steps are commonly included in EDA?** Understanding the structure, checking missing values and duplicates, cleaning, calculating statistics, visualizing variables, and writing observations.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook eda_titanic.ipynb
```
