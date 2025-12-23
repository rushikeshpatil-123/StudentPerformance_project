import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Student Performance App", layout= "wide")
st.title("Student Performance Streamlit App")
st.markdown("---")

st.header("Data Ingestion")

@st.cache_data
def load_data():
    return pd.read_csv("C:\StudentPerformance_project\Data\Student_Performance.csv")

df = load_data()

st.dataframe(df.head())
st.markdown("---")

st.subheader("Descriptive Statistics")
st.write(df.describe(include="all"))

st.header("Exploratory Data Analysis")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
cols = df.columns

for i, ax in enumerate(axes.flat):
    sns.histplot(df[cols[i]], kde=True, ax=ax)
    ax.set_title(cols[i])

plt.tight_layout()
st.pyplot(fig)

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=[np.number])

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.markdown("---")

st.header("Data Preprocessing")

df_model = df.copy()

if "Extracurricular Activities" in df_model.columns:
    df_model["Extracurricular Activities"] = df_model["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )

df_model = df_model.apply(pd.to_numeric, errors="coerce")

df_model = df_model.dropna()

st.write("Processed Data Preview")
st.dataframe(df_model.head())

X = df_model.drop("Performance Index", axis=1)
y = df_model["Performance Index"]

test_size = st.slider("Test size", 0.1, 0.4, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

st.header("Random Forest Model")

n_estimators = st.slider("Number of Trees", 100, 500, 200, step=50)
max_depth = st.slider("Max Depth", 2, 20, 10)

rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model Performance")
c1, c2 = st.columns(2)
c1.metric("R2 Score", f"{r2:.3f}")
c2.metric("RMSE", f"{rmse:.2f}")

st.subheader("Actual vs Predicted")

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred)
ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.set_title("Actual vs Predicted Performance Index")
st.pyplot(fig3)

st.header("Predict Performance Index")

col1, col2, col3 = st.columns(3)

with col1:
    hours = st.number_input("Hours Studied", 0.0, 24.0, 6.0)

with col2:
    prev = st.number_input("Previous Scores", 0.0, 100.0, 70.0)

with col3:
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

col4, col5 = st.columns(2)

with col4:
    sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)

with col5:
    papers = st.number_input("Sample Question Papers Practiced", 0, 50, 5)

extra_val = 1 if extra == "Yes" else 0

input_df = pd.DataFrame([{
    "Hours Studied": hours,
    "Previous Scores": prev,
    "Extracurricular Activities": extra_val,
    "Sleep Hours": sleep,
    "Sample Question Papers Practiced": papers
}])

if st.button("Predict Performance"):
    prediction = rf.predict(input_df)[0]
    st.success(f"Predicted Performance Index: {prediction:.2f}")
