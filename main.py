# from fastapi import FastAPI
from fastapi_offline import FastAPIOffline
# from routes.api import api_router
from views.api_v2 import V2_API_ROUTER
from fastapi.middleware.cors import CORSMiddleware

# fast api init
app = FastAPIOffline()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(V2_API_ROUTER)
