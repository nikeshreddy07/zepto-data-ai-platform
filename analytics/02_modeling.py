import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

HERE = Path(__file__).parent
CHARTS = HERE / "charts"; CHARTS.mkdir(exist_ok=True)
df = pd.read_csv(HERE / "titanic.csv")
model = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
X = model.drop(columns="survived"); y = model["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
num_cols = ["pclass", "age", "sibsp", "parch", "fare"]; cat_cols = ["sex", "embarked"]
preprocessor = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])
models = {"Logistic Regression": LogisticRegression(max_iter=2000), "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5), "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200)}
results=[]; fitted={}; roc_data={}
for name, estimator in models.items():
    pipe=Pipeline([("preprocess", preprocessor), ("model", estimator)]); pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]; fpr,tpr,_=roc_curve(y_test,prob); auc=roc_auc_score(y_test,prob)
    roc_data[name]=(fpr,tpr,auc); fitted[name]=pipe
    metrics={"Model":name,"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0),"AUC":auc}
    results.append(metrics); print("\n",name,metrics); print("Confusion matrix:\n",confusion_matrix(y_test,pred))
comparison=pd.DataFrame(results); print("\nClassifier comparison:\n",comparison.to_string(index=False)); comparison.to_csv(HERE/"classifier_comparison.csv",index=False)

tree_pipe=fitted["Decision Tree"]; tree_model=tree_pipe.named_steps["model"]; feature_names=tree_pipe.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(20,10)); plot_tree(tree_model,feature_names=feature_names,class_names=["Not Survived","Survived"],filled=False,max_depth=3,fontsize=7); plt.title("Decision Tree (first 3 levels shown)"); plt.tight_layout(); plt.savefig(CHARTS/"decision_tree.png",dpi=160,bbox_inches="tight"); plt.close()
plt.figure(figsize=(8,6))
for name,(fpr,tpr,auc) in roc_data.items(): plt.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],linestyle="--",label="Random baseline"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves — Titanic Classifiers"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS/"roc_curves.png",dpi=160); plt.close()

variants={"baseline":Pipeline([("preprocess",preprocessor),("model",LogisticRegression(max_iter=2000))]),"class_weight_balanced":Pipeline([("preprocess",preprocessor),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))]),"SMOTE":ImbPipeline([("preprocess",preprocessor),("smote",SMOTE(random_state=42)),("model",LogisticRegression(max_iter=2000))])}
imbalance_rows=[]
for name,pipe in variants.items():
    pipe.fit(X_train,y_train); pred=pipe.predict(X_test); imbalance_rows.append({"Variant":name,"Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0)})
imbalance=pd.DataFrame(imbalance_rows); print("\nImbalance comparison:\n",imbalance.to_string(index=False)); imbalance.to_csv(HERE/"imbalance_comparison.csv",index=False)

rf_pipe=Pipeline([("preprocess",preprocessor),("model",RandomForestClassifier(random_state=42,oob_score=True,n_jobs=-1))])
grid=GridSearchCV(rf_pipe,{"model__n_estimators":[100,200],"model__max_depth":[None,5,10],"model__max_features":["sqrt","log2"]},cv=5,scoring="f1",n_jobs=-1); grid.fit(X_train,y_train)
oob=grid.best_estimator_.named_steps["model"].oob_score_; print("\nBest RF parameters:",grid.best_params_); print("Best CV score:",grid.best_score_); print("OOB score:",oob)

reg=model.dropna(subset=["fare"]).copy(); reg_X=pd.get_dummies(reg.drop(columns="fare"),columns=["sex","embarked"],drop_first=True); reg_y=reg["fare"]
rx_train,rx_test,ry_train,ry_test=train_test_split(reg_X,reg_y,test_size=.20,random_state=42)
reg_model=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",LinearRegression())]); reg_model.fit(rx_train,ry_train); rpred=reg_model.predict(rx_test)
mae=mean_absolute_error(ry_test,rpred); rmse=np.sqrt(mean_squared_error(ry_test,rpred)); r2=r2_score(ry_test,rpred); n,p=rx_test.shape; adj_r2=1-(1-r2)*(n-1)/(n-p-1) if n-p-1>0 else np.nan; residuals=ry_test.to_numpy()-rpred
plt.figure(figsize=(8,6)); plt.scatter(rpred,residuals,alpha=.65); plt.axhline(0,linestyle="--"); plt.xlabel("Predicted Fare"); plt.ylabel("Residual (Actual - Predicted)"); plt.title("Residual Plot — Fare Regression"); plt.tight_layout(); plt.savefig(CHARTS/"residual_plot.png",dpi=160); plt.close()
mid=np.median(rpred); low_std=np.std(residuals[rpred<=mid]); high_std=np.std(residuals[rpred>mid]); ratio=max(low_std,high_std)/max(min(low_std,high_std),1e-12)
hetero_note=(f"Residual spread is approximately constant across the two predicted-fare halves (spread ratio {ratio:.2f})." if ratio<1.5 else f"Residual spread differs noticeably across the two predicted-fare halves (spread ratio {ratio:.2f}), suggesting possible heteroscedasticity.")
print("\nRegression:",{"MAE":mae,"RMSE":rmse,"R2":r2,"Adjusted_R2":adj_r2}); print("Heteroscedasticity discussion:",hetero_note)

final_row=comparison.sort_values(["F1","AUC"],ascending=False).iloc[0]
recommendation=(f"Based on the held-out test set, {final_row['Model']} has the highest F1 score ({final_row['F1']:.3f}) among the three classifiers, with accuracy {final_row['Accuracy']:.3f}, precision {final_row['Precision']:.3f}, recall {final_row['Recall']:.3f}, and AUC {final_row['AUC']:.3f}. These metrics provide the basis for selecting the final classifier, while the class-imbalance comparison shows how precision, recall, and F1 change under balanced and SMOTE training.")

best_classifier=grid.best_estimator_; joblib.dump(best_classifier,HERE/"best_pipeline.joblib"); joblib.dump(best_classifier,HERE/"model_pipeline.joblib"); loaded=joblib.load(HERE/"best_pipeline.joblib"); print("\nReloaded pipeline predictions:",loaded.predict(X_test.head(5)))

report="# Modeling Results\n\n## Classification comparison\n\n"+comparison.to_string(index=False)+"\n\n## Imbalance comparison\n\n"+imbalance.to_string(index=False)+f"\n\n## Random Forest GridSearchCV\n\n- Best parameters: `{grid.best_params_}`\n- Best CV F1: **{grid.best_score_:.3f}**\n- OOB score: **{oob:.3f}**\n\n## Regression — Fare\n\n- MAE: **{mae:.3f}**\n- RMSE: **{rmse:.3f}**\n- R²: **{r2:.3f}**\n- Adjusted R²: **{adj_r2:.3f}**\n\n### Residual / heteroscedasticity discussion\n\n{hetero_note} The residual plot is saved as `charts/residual_plot.png`.\n\n## Final classifier recommendation\n\n{recommendation}\n\n## Saved pipeline verification\n\nThe complete preprocessing + Random Forest pipeline was saved as `best_pipeline.joblib`, reloaded, and used to predict the first five raw test rows successfully.\n"
(HERE/"MODEL_RESULTS.md").write_text(report,encoding="utf-8")
