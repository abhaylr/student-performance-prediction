import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data.csv")

# -----------------  DATA READING  -----------------------
print("shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head)

print("\nCollumn Info:")
print(df.info())

print("\nMissing valuse:")
print(df.isna().sum())

# ------------------ DATA HANDLING ------------------

nums_colms = df.select_dtypes(include=['int64', "float64"]).columns
categ_colms = df.select_dtypes(include=['object']).columns

print("Numeric:", nums_colms)
print("Categorical:", categ_colms)



#----- Filling missing values by mean---

for col in nums_colms:
    df[col] = df[col].fillna(df[col].mean())

# ------Filling missing values by mode

for col in categ_colms:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isna().sum()) # Got zero missing values

# _---------------------- DATA VISUALIZE ------------------


plt.hist(df["Exam_Score"])
plt.xlabel("Exam Score")
plt.ylabel("Count")
plt.title("Distribution of Exam Score")
plt.show()


# --------- Scattr plot #Hours studied vs Exam Score ---
plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.title("Hours Studied vs Exam Score")
plt.show()


# --------- Scattr plot Attendence vs Exam Score ---
plt.scatter(df["Attendance"], df["Exam_Score"])
plt.xlabel("Attendance")
plt.ylabel("Exam Scores")
plt.title("Hours Studied vs Exam Score")
plt.show()

     # <----- Checking corelation for numeric colms -------

numeric_df = df.select_dtypes(include=["int64", "float64"])

corr = numeric_df.corr()
print(corr["Exam_Score"].sort_values(ascending=False))


     # <------ Heatmap of correlations --------------->

plt.figure(figsize=(12, 8))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# --------------- Converting categorical data to numbers(Encoding) ---------

df = pd.get_dummies(df, columns=categ_colms) # One-Hot Encoding

print(df.head())
print(df.shape)
print(df.dtypes)    # 'int64/bool


# ---------- Split Data into Input and Target(Exam_Score)

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# -------------- Train-test split ---------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ Prediction for given test set ---------

y_pred = model.predict(X_test)
print(y_pred[:5])


# ------------- Check the error(how good the model)--------

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print("MAE:", mae)
print("R2 Score:", r2)


# -------------- Compare actual vs predicted values -----------

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(10))



##---------- Plot actual vs predicted -------------

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted")
plt.show()