# WIDS Data Science Assignments

This repository contains two data science assignments focusing on data analysis, visualization, and machine learning using Python libraries such as Pandas, NumPy, Matplotlib, and Scikit-learn.

## Repository Structure

```
WIDS/
├── Assignment1/          # Questions 6-9: Data Analysis with NumPy and Pandas
│   ├── q6Toq9.ipynb
│   ├── farm_metrics.csv
│   ├── sonar_points.csv
│   ├── choco_quality.csv
│   ├── game_economy.csv
│   └── quake_grid.csv
│
└── Assignment2/          # California Housing Price Prediction
    ├── a2.ipynb
    └── housing.csv
```

---

## Assignment 1: Data Analysis Fundamentals (Questions 6-9)

### Overview
This assignment explores fundamental data analysis techniques using NumPy and Pandas, covering statistical analysis, matrix operations, data visualization, and data preprocessing.

### Questions & Datasets

#### **Question 6: Farm Metrics Analysis**
**Dataset:** `farm_metrics.csv` (100 rows × 4 columns)
- **Columns:** plant_id, height_cm, nutrient_level, yield_grams
- **Description:** Agricultural data tracking plant growth metrics and yield performance

**Tasks Performed:**
- Computed descriptive statistics (mean, median, variance) for plant height
- Analyzed correlation between nutrient levels and yield
- Generated simulated yield values using normal distribution
- Performed matrix operations (transpose and inverse) on nutrient interaction matrix
- Created visualizations: scatter plot (nutrient level vs yield) and histogram of simulated yields

**Approach:**
- Used Pandas for data loading and statistical computations
- Applied NumPy for matrix operations and random number generation
- Employed Matplotlib for data visualization

---

#### **Question 7: Sonar Points Analysis**
**Dataset:** `sonar_points.csv`
- **Columns:** x, y, z coordinates
- **Description:** 3D sonar point cloud data representing underwater terrain

**Tasks Performed:**
- Calculated Euclidean distances from origin for each point
- Normalized coordinates (mean=0, variance=1)
- Computed covariance matrix of normalized points
- Identified top 5 deepest points using Pandas sorting
- Created visualizations: 2D scatter plot (x vs y) and histogram of depth distribution

**Approach:**
- Used NumPy for distance calculations and normalization
- Applied linear algebra operations for covariance matrix
- Utilized Pandas for data sorting and filtering
- Implemented data standardization techniques

---

#### **Question 8: Chocolate Quality Analysis**
**Dataset:** `choco_quality.csv`
- **Columns:** sugar, cocoa, weight
- **Description:** Chocolate product quality metrics

**Tasks Performed:**
- Data cleaning: removed missing values using `.dropna()`
- Generated descriptive statistics with `.describe()`
- Computed custom quality score: `0.3×sugar + 0.7×cocoa`
- Generated random noise using normal distribution
- Created visualizations: scatter plot (cocoa % vs sugar %) and weight distribution histogram

**Approach:**
- Data preprocessing with Pandas
- Feature engineering using weighted formula
- Statistical simulation with NumPy
- Data visualization with Matplotlib

---

#### **Question 9: Signal Processing with Matrices**
**Task:** Matrix operations on sine and cosine signals

**Tasks Performed:**
- Generated 4×3 matrices using sine and cosine of random values (0 to 2π)
- Computed dot product of matrices (A · B^T)
- Calculated row-wise cross products

**Approach:**
- Used NumPy's `random.uniform()` for value generation
- Applied trigonometric functions for signal generation
- Performed linear algebra operations (dot product, cross product)

---

## Assignment 2: California Housing Price Prediction

### Overview
A comprehensive machine learning project involving exploratory data analysis (EDA), feature engineering, dimensionality reduction using PCA, and multiple regression modeling techniques.

### Dataset
**File:** `housing.csv` (20,641 rows × 10 columns)
- **Source:** California Housing Dataset (originally from sklearn.datasets, loaded locally from Kaggle)
- **Columns:**
  - `longitude`, `latitude` - Geographic coordinates
  - `housing_median_age` - Age of housing units
  - `total_rooms`, `total_bedrooms` - Room counts
  - `population`, `households` - Demographic data
  - `median_income` - Economic indicator
  - `median_house_value` - Target variable (housing prices)
  - `ocean_proximity` - Categorical location feature

### Tasks & Methodology

#### **Task 1: Data Loading & Exploration**
- Loaded dataset using Pandas
- Examined dataset shape, column names, and first 10 rows
- Used `.info()` and `.describe()` for data understanding
- Identified feature with largest variance (population-related features)

#### **Task 2: Univariate Analysis**
**Histogram Analysis:**
- Plotted distributions for all numeric columns
- Experimented with different bin sizes (10, 30, 50, 100)
- **Skewness Handling Methods** (documented):
  - Log transformation
  - Square-root transformation
  - Box-Cox transformation
  - Outlier filtering and winsorization

**Outlier Detection:**
- Created box plots for median_income, total_rooms, and population
- Identified outliers using interquartile range (IQR)

**Correlation Analysis:**
- Generated correlation heatmap using Seaborn
- Identified relationships between features
- Used for feature selection (excluded latitude/longitude due to weak linear correlation)

#### **Task 3: Geospatial Visualization**
- Created scatter plot with longitude vs latitude
- Color-coded by median house value (viridis colormap)
- Point sizes proportional to population density
- Visualized California's coastal housing price patterns

#### **Task 4: Feature Scaling & PCA**
**Why Scaling Before PCA?**
Features with large numerical ranges dominate variance calculations, introducing scale bias. Scaling ensures all features contribute proportionally to principal components.

**PCA Implementation:**
1. Handled missing values using `SimpleImputer` (median strategy)
2. Applied `StandardScaler` for normalization
3. Generated explained variance plot
4. Selected top 2 principal components
5. Created PC1 vs PC2 scatter plot colored by house value

**Key Insight:** Small number of components explained most variance, indicating redundancy among original features.

#### **Task 5: Multiple Linear Regression**

**Pipeline Construction:**
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])
```

**Feature Selection:**
- Excluded: latitude, longitude (weak linear correlation)
- Excluded: ocean_proximity (categorical, requires encoding)
- Target: median_house_value

**Model Evaluation Metrics:**
- **MSE (Mean Squared Error)** - Average squared prediction error
- **MAE (Mean Absolute Error)** - Average absolute prediction error
- **R² Score** - Proportion of variance explained
- **Adjusted R²** - R² adjusted for number of predictors

**Diagnostic Plots:**
1. **Actual vs Predicted** - Checks overall model fit
2. **Residuals vs Predicted** - Detects non-linearity and heteroscedasticity

**Critical Questions Addressed:**
- **Is high R² always good?** No - may indicate overfitting without generalization
- **Is low training loss always preferred?** No - extremely low loss suggests memorization over learning

#### **Bonus: Regularization Techniques**

**Ridge Regression (L2 Regularization):**
- Penalty term: α × Σ(coefficients²)
- Shrinks coefficients toward zero
- Reduces overfitting while keeping all features

**Lasso Regression (L1 Regularization):**
- Penalty term: α × Σ|coefficients|
- Performs feature selection by zeroing out coefficients
- Creates sparse models

**Implementation:** Both models used same pipeline structure with different regularization parameters (α=1.0 for Ridge, α=0.01 for Lasso)

---

## Technologies & Libraries

### Core Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing and linear algebra
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

### Machine Learning
- **Scikit-learn:**
  - `StandardScaler` - Feature scaling
  - `PCA` - Dimensionality reduction
  - `LinearRegression` - Linear modeling
  - `Ridge`, `Lasso` - Regularized regression
  - `train_test_split` - Data splitting
  - `Pipeline` - Workflow management
  - `SimpleImputer` - Missing value handling

### Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Adjusted R² Score

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Assignment 1
```bash
cd Assignment1
jupyter notebook q6Toq9.ipynb
```

### Assignment 2
```bash
cd Assignment2
jupyter notebook a2.ipynb
```

---

## Results & Insights

### Assignment 1
- Effective use of NumPy for mathematical computations
- Clear visualization of data distributions and relationships using plt

### Assignment 2
- Basic EDA stuff before applying regression
- Built interpretable linear regression model for housing price prediction
- PCA reduced dimensionality while retaining significant variance
- Tried Regularization techniques (Ridge/Lasso) 
