# from langcorn import create_service

# app = create_service(
#     "finalProject1:qa_chain"
# )

from fastapi.middleware.cors import CORSMiddleware
from langcorn import create_service

# Create the FastAPI app using Langcorn
app = create_service("finalProject1:qa_chain")

# Define the list of origins that should be allowed to make requests to your server
origins = [
    "http://localhost:3000",  # Your frontend running on localhost:3000
]

# Add the CORS middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

