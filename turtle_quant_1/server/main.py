import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
def root():
    return {"message": "success"}


# Include the routers in the main app with a prefix
# TODO

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
