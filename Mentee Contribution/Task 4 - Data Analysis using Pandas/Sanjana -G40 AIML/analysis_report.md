# Pandas Data Analysis Report

## Dataset Overview
- Dataset: Titanic
- Rows: 891
- Columns: 12

## Data Quality Issues
1. Missing values in Age.
2. Missing values in Embarked.
3. Some duplicate records.

## Cleaning Steps
- Filled missing Age with median.
- Filled missing Embarked with mode.
- Removed duplicates.
- Standardized column names.

## Exploratory Data Analysis
- Female passengers had higher survival rates.
- Most passengers were in Class 3.
- Majority of passengers were young adults.

## Grouping Results
Grouped by gender and calculated survival rate.

## Feature Engineering
Created:
- age_group

## Visualizations
1. Survival by Gender
2. Age Distribution
3. Passenger Class Distribution

## Key Insights

### Insight 1
Female passengers survived more often than males.

### Insight 2
Class 3 had the highest passenger count.

### Insight 3
Young adults formed the largest age group.

### Insight 4
Missing values were mainly found in Age.

### Insight 5
Higher-class passengers showed better survival rates.

### Insight 6
Most passengers embarked from Southampton.

### Insight 7
Children had relatively higher survival chances.

### Insight 8
Data cleaning improved dataset consistency.

## Recommendations
- Handle missing values before analysis.
- Focus on demographic segmentation.
- Use class and age as important predictive variables.

## Conclusion
Pandas successfully cleaned, analyzed, and summarized the Titanic dataset.