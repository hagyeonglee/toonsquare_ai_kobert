import os
import sys
from azure.storage.file import FileService
import logging
from util import constant

try:
    AZURE_STORAGE_ACCOUNT_NAME = os.environ['AZURE_STORAGE_ACCOUNT_NAME']
    AZURE_STORAGE_ACCOUNT_KEY = os.environ['AZURE_STORAGE_ACCOUNT_KEY']
    AZURE_STORAGE_NAME = os.environ['AZURE_STORAGE_NAME']
except KeyError:
    AZURE_STORAGE_ACCOUNT_NAME ="check Account name"
    AZURE_STORAGE_ACCOUNT_KEY = "check Account key"
    AZURE_STORAGE_NAME = "check storage name"

REAL_PATH = os.path.realpath(__file__)
DIRNAME = os.path.dirname(REAL_PATH)

clear = lambda: os.system('clear')

def init_azure_service():
    try:
        azure_file_service = FileService(account_name=AZURE_STORAGE_ACCOUNT_NAME,
                                         account_key=AZURE_STORAGE_ACCOUNT_KEY)
    except Exception as e:
        print("init_azure_service error : ", sys.exc_info()[0], e)
        sys.exit()
    return azure_file_service

def compress_model(dirname):
    model_abs_path = os.path.join(dirname, constant.MODEL_DIR_PATH)
    shutil.make_archive(model_abs_path, 'zip', root_dir=dirname, base_dir=constant.MODEL_DIR_PATH)


def upload_file(azure_file_service, dir, filename, upload_file_abs_path):
    print("upload filename :", filename, "abs_path : ", upload_file_abs_path, sep='', end='\r')
    try:
        azure_file_service.create_file_from_path(
            AZURE_STORAGE_NAME,
            dir,
            filename,
            upload_file_abs_path)
    except Exception as e:
        print("upload_file error:", sys.exc_info()[0], e)
        sys.exit()