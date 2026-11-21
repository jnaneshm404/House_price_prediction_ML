import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error
import joblib
import numpy as np

df=pd.read_csv("bengaluru_house_prices.csv")

le=LabelEncoder()
df["locations"]=le.fit_transform(df["location"])

X=df[["area_sqft","bedrooms","bathrooms","locations","age_years","parking"]]
y=df[["price_lakhs"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

preds=model.predict(X_test)
print("MAE: ",mean_absolute_error(y_test,preds))
print("RMSE: ",root_mean_squared_error(y_test,preds))
print("R2: ",r2_score(y_test,preds))

joblib.dump(model, "house_price_model.pkl")
joblib.dump(le, "locations.pkl")
print("Model saved!")