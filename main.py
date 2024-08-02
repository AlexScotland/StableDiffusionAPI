import shutil

try:
    from models.LoRA.lora_conf import ALL_LORAS
except ModuleNotFoundError:
    shutil.copyfile("models/LoRA/lora_conf.template", "models/LoRA/lora_conf.py")
    print("Script may fail, please rerun :)")
from fastapi_offline import FastAPIOffline
from views.api_v2 import V2_API_ROUTER
from fastapi.middleware.cors import CORSMiddleware

# Create LoRA

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
