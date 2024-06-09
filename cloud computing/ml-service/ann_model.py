from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import uvicorn
import signal
import sys
from pydantic import BaseModel

app = FastAPI()

# Load the pre-trained Keras model
loaded_model = tf.keras.models.load_model("ann_model.h5")


class Payload(BaseModel):
    stayDurationInDay: int
    bookingWindow: int
    distanceToCoastline: int
    lat: float
    bedroom: int
    reviewSentimentScore: float
    beds: int
    lng: float


@app.post("/predict")
async def predict(body: Payload):
    try:
        input_data = np.array(
            [
                [
                    body.stayDurationInDay,
                    body.bookingWindow,
                    body.distanceToCoastline,
                    body.lat,
                    body.bedroom,
                    body.reviewSentimentScore,
                    body.beds,
                    body.lng,
                ]
            ]
        )
        predictions = loaded_model.predict(input_data)
        predictions_list = (
            predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        )
        return JSONResponse(
            content={"status": "success", "result_prediction": predictions_list}
        )
    except Exception as e:
        return JSONResponse(content={"status": "fail", "error": str(e)})


@app.get("/")
def read_root():
    return {"message": "Hello world!"}


def shutdown_handler(signal, frame):
    print("Shutting down gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    uvicorn.run(app, host="localhost", port=8080)