import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE / "titanic.csv"
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)

try:
    df = sns.load_dataset("titanic")
except Exception:
    # Offline fallback: use the committed titanic.csv if the public dataset host is unavailable.
    if not CSV.exists():
        raise
    df = pd.read_csv(CSV)
df.to_csv(CSV, index=False)
print(df.info())
print(df.describe(include="all"))
print("Shape:", df.shape)

missing = df.isna().mean().mul(100)
print("\nMissing percentages:")
print(missing[missing > 0])

clean = df.copy()
strategy_lines = []
for col in clean.columns:
    pct = clean[col].isna().mean() * 100
    if pct == 0:
        continue
    if pct < 5:
        strategy_lines.append(f"- **{col}**: {pct:.2f}% missing -> drop affected rows (<5% rule).")
    elif pct <= 30:
        strategy_lines.append(f"- **{col}**: {pct:.2f}% missing -> impute (5-30% rule).")
    else:
        strategy_lines.append(f"- **{col}**: {pct:.2f}% missing -> drop column because missingness is very high.")

if "deck" in clean and clean["deck"].isna().mean() > 0.30:
    clean = clean.drop(columns=["deck"])
if 0.05 <= clean["age"].isna().mean() <= 0.30:
    clean["age"] = clean["age"].fillna(clean["age"].median())
for col in ["embarked", "embark_town"]:
    if col in clean and clean[col].isna().mean() < 0.05:
        clean = clean.dropna(subset=[col])
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])

def outlier_count(s):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

age_outliers = outlier_count(clean["age"])
fare_outliers = outlier_count(clean["fare"])
fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]
fare_skew = clean["fare"].skew()
print("Age IQR outliers:", age_outliers)
print("Fare IQR outliers:", fare_outliers)
print("Fare mean:", fare_mean)
print("Fare median:", fare_median)
print("Fare mode:", fare_mode)
print("Fare skewness:", fare_skew)

sex_survival = clean.groupby("sex")["survived"].mean()
class_survival = clean.groupby("pclass")["survived"].mean()
sex_class_survival = clean.groupby(["sex", "pclass"])["survived"].mean()
print("\nSurvival by sex:\n", sex_survival)
print("\nSurvival by pclass:\n", class_survival)
print("\nSurvival by sex and pclass:\n", sex_class_survival)

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()
print("\nCorrelation:\n", corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", square=True)
plt.title("Titanic Numeric Correlation Matrix")
plt.tight_layout(); plt.savefig(CHARTS / "correlation_heatmap.png", dpi=160); plt.close()

plt.figure(figsize=(7, 5)); plt.hist(clean["age"].dropna(), bins=20); plt.title("Age Distribution"); plt.xlabel("Age"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(CHARTS / "age_histogram.png", dpi=160); plt.close()
plt.figure(figsize=(7, 4)); plt.boxplot(clean["age"].dropna(), orientation="horizontal"); plt.title("Age Box Plot"); plt.xlabel("Age"); plt.tight_layout(); plt.savefig(CHARTS / "age_boxplot.png", dpi=160); plt.close()
plt.figure(figsize=(7, 5)); plt.hist(clean["fare"].dropna(), bins=30); plt.title("Fare Distribution"); plt.xlabel("Fare"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(CHARTS / "fare_histogram.png", dpi=160); plt.close()
plt.figure(figsize=(7, 4)); plt.boxplot(clean["fare"].dropna(), orientation="horizontal"); plt.title("Fare Box Plot"); plt.xlabel("Fare"); plt.tight_layout(); plt.savefig(CHARTS / "fare_boxplot.png", dpi=160); plt.close()

grouped=clean.groupby(["sex","pclass"])["survived"].mean().unstack(); ax=grouped.plot(kind="bar", figsize=(7,5)); ax.set_title("Survival Rate by Sex and Passenger Class"); ax.set_ylabel("Survival rate"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(CHARTS / "survival_by_class_sex.png", dpi=160); plt.close()
grouped=clean.groupby("sex")["survived"].mean(); ax=grouped.plot(kind="bar", figsize=(7,5)); ax.set_title("Survival Rate by Sex"); ax.set_ylabel("Survival rate"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(CHARTS / "survival_by_sex.png", dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(7,5)); positions=[]; labels=[]; pos=1
for cls in sorted(clean["pclass"].dropna().unique()):
    for sex in ["female","male"]:
        vals=clean[(clean["pclass"]==cls)&(clean["sex"]==sex)]["fare"].dropna(); positions.append(vals.values); labels.append(f"{cls}-{sex}"); pos+=1
ax.boxplot(positions, tick_labels=labels); ax.set_title("Fare Distribution by Class and Sex"); ax.set_ylabel("Fare"); ax.tick_params(axis="x", rotation=45); plt.tight_layout(); plt.savefig(CHARTS / "fare_by_class_sex.png", dpi=160); plt.close()
plt.figure(figsize=(7,5));
for status in sorted(clean["survived"].dropna().unique()):
    part=clean[clean["survived"]==status]; plt.scatter(part["age"],part["fare"],alpha=.55,label=f"Survived={int(status)}")
plt.title("Age vs Fare by Survival"); plt.xlabel("Age"); plt.ylabel("Fare"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / "age_fare_survival.png", dpi=160); plt.close()

scaled = clean[["age", "fare"]].copy()
scaled[["age", "fare"]] = StandardScaler().fit_transform(scaled[["age", "fare"]])
print("\nStandardization means:\n", scaled.mean())
print("Standardization std (population):\n", scaled.std(ddof=0))

pairs = []
for i, a in enumerate(corr.columns):
    for b in corr.columns[i+1:]:
        pairs.append((abs(corr.loc[a, b]), a, b, corr.loc[a, b]))
pairs.sort(reverse=True)
strongest = pairs[:2]

fare_shape = "right-skewed" if fare_mean > fare_median and fare_median > fare_mode else "not strongly right-skewed"
female = sex_survival.get("female", float("nan")); male = sex_survival.get("male", float("nan"))
lines = [
"# EDA Results and Interpretations", "", "## Missing-value strategy", *strategy_lines, "",
"## Univariate analysis", "",
"### Age histogram and box plot", f"The age distribution is concentrated among young and middle-aged passengers, with a smaller tail toward older ages. The IQR rule identifies **{age_outliers}** age outliers after the documented cleaning/imputation strategy.", "",
"### Fare histogram and box plot", f"Fare is **{fare_shape}**: mean = **{fare_mean:.2f}**, median = **{fare_median:.2f}**, mode = **{fare_mode:.2f}**, and skewness = **{fare_skew:.2f}**. The box plot identifies **{fare_outliers}** IQR-based fare outliers, showing a small number of very expensive tickets.", "",
"## Bivariate analysis", "",
"### Survival by sex", f"The observed survival rate is **{female:.2%}** for females and **{male:.2%}** for males. This shows a substantial difference in survival by sex in this dataset.", "",
"### Survival by passenger class", f"Survival varies across passenger classes: {class_survival.to_dict()}. This indicates that passenger class is associated with survival in the observed sample.", "",
"### Survival by sex and class", "The combined grouping shows that survival differences by sex also vary across passenger classes. The chart makes the interaction between sex and class visible rather than treating either variable in isolation.", "",
"## Multivariate data story", "",
"### Survival by sex and class", "The grouped survival chart shows how class changes the observed survival rate within each sex category. The pattern demonstrates that both variables contribute useful segmentation of survival outcomes.", "",
"### Fare by class and sex", "The fare distributions differ substantially across passenger classes, while sex adds another layer of variation within classes. Higher passenger classes generally contain higher fares and a wider range of premium-ticket values.", "",
"### Age vs fare by survival", "The scatter plot shows fare values concentrated at lower prices but extending to high-ticket outliers, while survival status is distributed across age and fare. This visualizes several variables simultaneously rather than a single pair of summary statistics.", "",
"### Survival by sex", "The sex-level survival chart provides a clear story when interpreted alongside class and fare: survival differs strongly by sex, and the class/fare charts show additional structure behind that difference.", "",
"## Strongest correlations"
]
for _, a, b, val in strongest:
    lines.append(f"- **{a} vs {b}: {val:.3f}** (absolute correlation {abs(val):.3f}). This is one of the two strongest linear relationships among the six required variables.")
lines += ["", "## Standardization sanity check", "The standardized `age` and `fare` columns have means approximately equal to 0 and population standard deviations approximately equal to 1. This is an EDA sanity check and is separate from the leakage-safe modeling Pipeline."]
(HERE / "EDA_INTERPRETATIONS.md").write_text("\n".join(lines), encoding="utf-8")
