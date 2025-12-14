from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.model.xor_from_scratch import train_xor, predict_xor

app = FastAPI(title="XOR Neural Network Demo")

# ---- In-memory model state ----
MODEL_STATE = {
    "trained": False,
    "params": None
}

# ---- Request schemas ----
class PredictRequest(BaseModel):
    x1: int
    x2: int


# ---- Endpoints ----

@app.post("/train")
def train():
    """
    Train XOR model from scratch.
    """
    params, history = train_xor(verbose=True, print_every=1000)

    MODEL_STATE["trained"] = True
    MODEL_STATE["params"] = params

    return {
        "status": "trained",
        "final_loss": history[-1],
        "predictions": params["final_predictions"].tolist()
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict XOR output for given inputs.
    """
    if not MODEL_STATE["trained"]:
        return {"error": "Model not trained yet. Call /train first."}

    x = np.array([[req.x1, req.x2]])
    prob = predict_xor(x, MODEL_STATE["params"])

    return {
        "input": [req.x1, req.x2],
        "probability of response being 1": float(prob),
        "Final Output": int(prob >= 0.5)
    }
