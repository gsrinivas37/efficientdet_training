import os
import gdown
import zipfile

def download_gdrive_file(FILE_ID, output):
    if os.path.exists(output):
        return
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, output, quiet=False)
    if output.endswith('.zip'):
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")