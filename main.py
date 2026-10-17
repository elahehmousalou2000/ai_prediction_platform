from fastapi import FastAPI
from sklearn.linear_model import LinearRegression

app = FastAPI()
model = LinearRegression()
model.fit([[50], [70], [100]], [100, 140, 200])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(size: float):
    return {"prediction": float(model.predict([[size]])[0])}
