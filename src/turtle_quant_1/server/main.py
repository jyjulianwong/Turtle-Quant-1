import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from turtle_quant_1.config import (
    CANDLE_UNIT,
    ENV,
    GCLOUD_PROJECT_ID,
    GCLOUD_REGION,
    GCLOUD_STB_DATA_NAME,
    HOST_TIMEZONE,
    MAX_CANDLE_GAPS_TO_FILL,
    MAX_HISTORY_DAYS,
)

app = FastAPI()

# Define the allowed origins
origins = [
    "http://localhost:8080",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", tags=["root"])
def root() -> JSONResponse:
    return {"message": "success"}


@app.get("/system-info", tags=["system-info"])
def system_info() -> JSONResponse:
    return {
        "env": ENV,
        "gcloud_region": GCLOUD_REGION,
        "gcloud_project_id": GCLOUD_PROJECT_ID,
        "gcloud_stb_data_name": GCLOUD_STB_DATA_NAME,
        "candle_unit": CANDLE_UNIT,
        "max_history_days": MAX_HISTORY_DAYS,
        "max_candle_gaps_to_fill": MAX_CANDLE_GAPS_TO_FILL,
        "host_timezone": HOST_TIMEZONE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
