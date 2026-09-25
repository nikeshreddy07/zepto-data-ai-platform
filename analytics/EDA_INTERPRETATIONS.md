# EDA Results and Interpretations

## Missing-value strategy
- **age**: 19.87% missing -> impute (5-30% rule).
- **embarked**: 0.22% missing -> drop affected rows (<5% rule).
- **deck**: 77.22% missing -> drop column because missingness is very high.
- **embark_town**: 0.22% missing -> drop affected rows (<5% rule).

## Univariate analysis

### Age histogram and box plot
The age distribution is concentrated among young and middle-aged passengers, with a smaller tail toward older ages. The IQR rule identifies **65** age outliers after the documented cleaning/imputation strategy.

### Fare histogram and box plot
Fare is **right-skewed**: mean = **32.10**, median = **14.45**, mode = **8.05**, and skewness = **4.80**. The box plot identifies **114** IQR-based fare outliers, showing a small number of very expensive tickets.

## Bivariate analysis

### Survival by sex
The observed survival rate is **74.04%** for females and **18.89%** for males. This shows a substantial difference in survival by sex in this dataset.

### Survival by passenger class
Survival varies across passenger classes: {1: 0.6261682242990654, 2: 0.47282608695652173, 3: 0.24236252545824846}. This indicates that passenger class is associated with survival in the observed sample.

### Survival by sex and class
The combined grouping shows that survival differences by sex also vary across passenger classes. The chart makes the interaction between sex and class visible rather than treating either variable in isolation.

## Multivariate data story

### Survival by sex and class
The grouped survival chart shows how class changes the observed survival rate within each sex category. The pattern demonstrates that both variables contribute useful segmentation of survival outcomes.

### Fare by class and sex
The fare distributions differ substantially across passenger classes, while sex adds another layer of variation within classes. Higher passenger classes generally contain higher fares and a wider range of premium-ticket values.

### Age vs fare by survival
The scatter plot shows fare values concentrated at lower prices but extending to high-ticket outliers, while survival status is distributed across age and fare. This visualizes several variables simultaneously rather than a single pair of summary statistics.

### Survival by sex
The sex-level survival chart provides a clear story when interpreted alongside class and fare: survival differs strongly by sex, and the class/fare charts show additional structure behind that difference.

## Strongest correlations
- **pclass vs fare: -0.548** (absolute correlation 0.548). This is one of the two strongest linear relationships among the six required variables.
- **sibsp vs parch: 0.415** (absolute correlation 0.415). This is one of the two strongest linear relationships among the six required variables.

## Standardization sanity check
The standardized `age` and `fare` columns have means approximately equal to 0 and population standard deviations approximately equal to 1. This is an EDA sanity check and is separate from the leakage-safe modeling Pipeline.