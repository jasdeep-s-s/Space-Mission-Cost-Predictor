# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
This is a **Space Mission Cost Predictor** AI/ML project that uses machine learning regression algorithms to predict the cost of future space missions based on historical space mission data. The model achieves approximately 90% accuracy on testing data.

## Development Environment
This project is developed in **Jupyter Notebook** format (.ipynb files) and uses Python with data science libraries. The project runs on Windows (PowerShell).

## Key Commands

### Running the Notebooks
Since this is a Jupyter Notebook project, you'll need to use Jupyter to work with the files:

```powershell
# Start Jupyter Notebook
jupyter notebook

# Or start Jupyter Lab (if installed)
jupyter lab
```

### Python Environment
The project uses the following key libraries (ensure they're installed):
```powershell
# Install required packages
pip install pandas seaborn matplotlib numpy scikit-learn xgboost
```

## Repository Structure

### Main Files
- **`SMC_Predictor.ipynb`** - Original model implementation using Decision Trees, Random Forest, and XGBoost
- **`SMC_Predictor_Regularized.ipynb`** - Regularized version using Ridge Regression to handle feature dominance and overfitting
- **`space_missions_dataset.csv`** - Dataset containing 500 space mission records with 15 columns

### Dataset Features
The dataset includes features such as:
- Mission metadata (ID, Name, Date, Target Type/Name, Mission Type)
- Physical metrics (Distance from Earth, Duration, Scientific Yield)
- Resource metrics (Crew Size, Fuel Consumption, Payload Weight)
- Target: **Mission Cost (billion USD)**

## Model Architecture

### Data Preprocessing
1. **Feature Engineering**: The project creates derived features to improve predictions:
   - `Time Efficiency` = Scientific Yield / (Mission Duration + small epsilon)
   - `Energy Demand` = Payload Weight × Distance from Earth
   - `Target Distance` = Target Name Code × Distance from Earth

2. **Categorical Encoding**: Categorical features are converted to numeric codes:
   - Target Type, Mission Type, Launch Vehicle, Target Name, Mission ID
   - Launch Date is converted to Launch Year

3. **Feature Scaling**: StandardScaler is applied to normalize all features before training

### Model Pipeline
The project implements three regression approaches:

1. **Decision Tree Regressor** (baseline)
2. **Random Forest Regressor** (ensemble method)
   - Achieves ~87% R² score
   - ~19.8% MAPE (Mean Absolute Percentage Error)

3. **XGBoost Regressor** (gradient boosting with regularization)
   - Best performance: ~91% R² score
   - ~16.7% MAPE
   - Hyperparameters: max_depth=5, learning_rate=0.1, n_estimators=100, reg_lambda=1.0

### Key Challenge Addressed
The main challenge was **feature dominance and overfitting**. The regularized version (SMC_Predictor_Regularized.ipynb) addresses this using:
- **Ridge Regression** methods to reduce the influence of dominant features
- **XGBoost with L2 regularization** (reg_lambda parameter) for better generalization

### Train/Test Split
- Training data: 80% of dataset
- Testing data: 20% of dataset
- Random state: 42 (for reproducibility)

## Model Evaluation

### Visualization
Both notebooks include:
- Feature importance plots (horizontal bar charts showing which features contribute most to predictions)
- Correlation analysis with pair plots

### Performance Metrics
- **R² Score**: Coefficient of determination (percentage of variance explained)
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

## Important Notes

### Working with Notebooks
- Each cell should be run sequentially from top to bottom
- The notebooks contain data exploration, preprocessing, model training, and evaluation steps
- Model objects (tree, forest, xg_reg) are created and trained within the notebooks

### Model Features List
The final feature set used for training includes (in order of creation):
```python
features = [
    'Target Distance', 'Time Efficiency', 'Energy Demand', 
    'Distance from Earth (light-years)', 'Mission Duration (years)', 
    'Scientific Yield (points)', 'Target_Type_Code', 'Mission_Type_Code', 
    'Launch_Vehicle_Code', 'Target_Name_Code', 'Mission_ID_Code', 
    'Launch Year', 'Crew Size', 'Fuel Consumption (tons)', 'Payload Weight (tons)'
]
```

### Reproducing Results
To reproduce the model results:
1. Ensure you have the exact versions of libraries (especially scikit-learn and xgboost)
2. Use the same random_state=42 for train_test_split
3. Run all cells in sequence from the beginning of the notebook
