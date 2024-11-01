import os

from settings import PACKAGE_DIR

ALL_ROUTERS_TO_INSTALL = []

for package_directory in os.scandir(PACKAGE_DIR):
    if not package_directory.is_dir() or package_directory.name.startswith("__"):
        continue
    for router in os.scandir(f"{package_directory.path}/routers"):
        if router.name.startswith("__"):
            continue
        ALL_ROUTERS_TO_INSTALL.append(f"api_packages.{package_directory.name}.routers.{router.name}")