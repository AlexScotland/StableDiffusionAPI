import importlib
import os

from fastapi import APIRouter

__globals = globals()

from routers.main import MAIN_ROUTER

# Dynamically import all routers
for file in os.listdir(os.path.dirname(__file__)):
    if file.startswith("__") or file == "main.py":
        continue
    __globals[file[:-3]] = importlib.import_module(f'.{file[:-3]}', package=__name__)
    MAIN_ROUTER.include_router(__globals[f"{file[:-3]}"].ROUTER)
