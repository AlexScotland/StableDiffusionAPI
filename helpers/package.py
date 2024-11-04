import os

from settings import PACKAGE_DIR

def find_package_name(os_path):
    for folder in os.scandir(PACKAGE_DIR):
        if folder.name == os_path: return folder
    return None