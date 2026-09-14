from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid"); OUT=Path("outputs"); OUT.mkdir(exist_ok=True)
df=pd.read_csv("car data.csv"); print("Original shape:",df.shape); print(df.head())
# Cleaning and consistent categorical values
df.columns=df.columns.str.strip(); df=df.drop_duplicates().dropna().copy()
for c in ["Fuel_Type","Seller_Type","Transmission"]: df[c]=df[c].astype(str).str.strip().str.title()
# Feature engineering: car age and brand
df["Car_Age"]=pd.Timestamp.now().year-df["Year"]
df["Brand"]=df["Car_Name"].str.split().str[0].str.title()
print("Cleaned shape:",df.shape); print("Missing values:\n",df.isna().sum()); print("Brands:",df.Brand.nunique())
# EDA
plt.figure(figsize=(9,5)); sns.histplot(df["Selling_Price"],kde=True); plt.title("Distribution of Selling Prices"); plt.tight_layout(); plt.savefig(OUT/"selling_price_distribution.png",dpi=180); plt.show()
plt.figure(figsize=(9,5)); sns.boxplot(data=df,x="Fuel_Type",y="Selling_Price",hue="Fuel_Type",legend=False); plt.title("Selling Price by Fuel Type"); plt.tight_layout(); plt.savefig(OUT/"price_by_fuel.png",dpi=180); plt.show()
plt.figure(figsize=(9,5)); sns.scatterplot(data=df,x="Car_Age",y="Selling_Price",hue="Fuel_Type"); plt.title("Selling Price vs Car Age"); plt.tight_layout(); plt.savefig(OUT/"price_vs_age.png",dpi=180); plt.show()
# Encode categories and split
X=df.drop(columns=["Selling_Price","Car_Name"]); y=df["Selling_Price"]
cat=X.select_dtypes(include="object").columns.tolist(); num=X.select_dtypes(exclude="object").columns.tolist()
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={"Linear Regression":LinearRegression(),"Random Forest Regressor":RandomForestRegressor(n_estimators=300,random_state=42)}; results=[]; fitted={}
for name,m in models.items():
 pipe=Pipeline([("preprocessor",pre),("model",m)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=pipe
 results.append({"Model":name,"MAE":mean_absolute_error(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"R2":r2_score(yte,pred)})
results=pd.DataFrame(results).sort_values("R2",ascending=False); print("Model evaluation:\n",results.round(3)); results.to_csv(OUT/"model_evaluation.csv",index=False)
best=results.iloc[0]["Model"]; print("Best model:",best)
# Feature importance for the selected best model
pipe=fitted[best]; names=pipe.named_steps["preprocessor"].get_feature_names_out(); model=pipe.named_steps["model"]
imp=model.feature_importances_ if hasattr(model,"feature_importances_") else np.abs(model.coef_)
fi=pd.DataFrame({"Feature":names,"Importance":np.ravel(imp)}).sort_values("Importance",ascending=False).head(15); fi.to_csv(OUT/"feature_importance.csv",index=False); plt.figure(figsize=(10,6)); sns.barplot(data=fi,x="Importance",y="Feature",hue="Feature",legend=False); plt.title(f"Top Feature Importances — {best}"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png",dpi=180); plt.show()
# correlation heatmap of numeric variables
plt.figure(figsize=(9,7)); sns.heatmap(df.select_dtypes(exclude="object").corr(),annot=True,fmt=".2f",cmap="coolwarm"); plt.title("Feature Correlation Heatmap"); plt.tight_layout(); plt.savefig(OUT/"correlation_heatmap.png",dpi=180); plt.show()
df.to_csv(OUT/"cleaned_car_data.csv",index=False)
