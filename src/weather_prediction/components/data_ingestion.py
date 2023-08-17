import os
import os.path
import urllib.request as rq
import zipfile

from weather_prediction import logger
from weather_prediction.constants import *
from weather_prediction.utils.common import get_size
from weather_prediction.entity.config_entity import DataIngestionConfig




class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = rq.urlretrieve(
                url = self.config.source_url,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} downloaded, info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory

        args:
            zip_file_path:str
        returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
