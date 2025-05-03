import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.train_pipeline import main as train_pipeline_main

if __name__=="__main__":
    try:
        # Run the training pipeline
        train_pipeline_main()
    except Exception as e:
        logging.error(f"Exception occurred in main: {e}")
        raise CustomException(e, sys)
