from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import signal
import sys
from datetime import date, datetime
from pydantic import BaseModel
import os
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

data_path = os.path.join(os.getcwd(), "random_forest_distinct_data.csv")
distinct_data = pd.read_csv(data_path)
random_forest_model = joblib.load("random_forest_model.pkl")


def booking_window(today, check_in):
    return (check_in - today).days


def stay_duration(check_in, check_out):
    return (check_out - check_in).days


def format_currency(value):
    value_str = f"{value:,.2f}"
    value_str = value_str.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {value_str}"


def predictRoom(predicted_df, property_name, room_id, check_in, check_out):
    filtered_df = predicted_df[
        (predicted_df["property_name"] == property_name)
        & (predicted_df["room_id"] == room_id)
    ].copy()

    if filtered_df.empty:
        raise ValueError("No matching property_name and room_id found in the data")

    current_price = filtered_df["average_daily_rate"].values

    filtered_df.drop(
        columns=["property_name", "room_id", "average_daily_rate"], inplace=True
    )

    date_check_in = datetime.strptime(check_in, "%Y-%m-%d").date()
    date_check_out = datetime.strptime(check_out, "%Y-%m-%d").date()

    today = date.today()
    booking_window_days = booking_window(today, date_check_in)

    stay_duration_days = stay_duration(date_check_in, date_check_out)

    filtered_df["stay_duration_in_days"] = stay_duration_days
    filtered_df["booking_window"] = booking_window_days

    xgboost_predictions_test = random_forest_model.predict(filtered_df)

    return (
        xgboost_predictions_test,
        current_price,
        stay_duration_days,
        booking_window_days,
    )


def predict_new(distance_to_coastline, lat, lng, check_in, check_out):
    today = date.today()

    date_check_in = datetime.strptime(check_in, "%Y-%m-%d").date()
    date_check_out = datetime.strptime(check_out, "%Y-%m-%d").date()

    booking_window_days = booking_window(today, date_check_in)
    stay_duration_days = stay_duration(date_check_in, date_check_out)

    data = np.array(
        [[booking_window_days, distance_to_coastline, stay_duration_days, lat, lng]]
    )

    prediction_result = random_forest_model.predict(data)

    return prediction_result.tolist()


class Payload(BaseModel):
    property_name: str
    room_id: int
    check_in: str
    check_out: str


class PayloadNew(BaseModel):
    lat: float
    beds: int
    bedroom: int
    bathroom: int
    parking: int
    pool: int
    beachfront: int
    lng: float
    check_in: str
    check_out: str


@app.post("/predict")
async def predict(body: Payload):
    try:
        prediction, current_price, stay_duration, booking_window = predictRoom(
            distinct_data,
            body.property_name,
            body.room_id,
            body.check_in,
            body.check_out,
        )
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "prediction": format_currency(prediction[0]),
                    "current_price": format_currency(current_price[0]),
                    "stay_duration_day": stay_duration,
                    "booking_window_day": booking_window,
                },
            }
        )
    except Exception as e:
        return JSONResponse(content={"status": "fail", "error": str(e)})


@app.post("/predict/new")
async def predictNew(body: PayloadNew):
    try:
        today = date.today()
        date_check_in = datetime.strptime(body.check_in, "%Y-%m-%d").date()
        date_check_out = datetime.strptime(body.check_out, "%Y-%m-%d").date()

        booking_window_days = booking_window(today, date_check_in)

        stay_duration_days = stay_duration(date_check_in, date_check_out)

        array_data = np.array(
            [
                [
                    booking_window_days,
                    stay_duration_days,
                    body.lat,
                    body.beds,
                    body.bedroom,
                    body.bathroom,
                    body.parking,
                    body.pool,
                    body.beachfront,
                    body.lng,
                ]
            ]
        )

        columns = [
            "booking_window",
            "stay_duration_in_days",
            "lat",
            "beds",
            "bedroom",
            "bathroom",
            "parking",
            "pool",
            "beachfront",
            "lng",
        ]

        data = pd.DataFrame(array_data, columns=columns)

        model = joblib.load("random_forest_model.pkl")
        prediction = model.predict(data)

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "prediction": format_currency(prediction[0]),
                },
            }
        )
    except Exception as e:
        return JSONResponse(content={"status": "fail", "error": str(e)})


@app.get("/properties")
async def get_propert1ies():
    try:
        properties = distinct_data["property_name"].unique().tolist()

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "properties": properties,
                },
            }
        )

    except Exception as e:
        return JSONResponse(content={"status": "fail", "error": str(e)})


@app.get("/rooms/{property_name}")
def get_rooms(property_name: str):
    try:
        filtered_data = distinct_data[distinct_data["property_name"] == property_name]
        rooms = filtered_data["room_id"].unique().tolist()
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "room_id": rooms,
                },
            }
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
