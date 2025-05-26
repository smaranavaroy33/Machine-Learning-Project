import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """
        Executes the data ingestion process which includes reading the dataset, saving the raw data,
        performing a train-test split, and saving the resulting datasets to specified file paths.
        Steps:
            1. Reads the dataset from a CSV file.
            2. Saves the raw dataset to the configured raw data path.
            3. Splits the dataset into training and testing sets.
            4. Saves the training and testing sets to their respective file paths.
            5. Logs each step of the process for traceability.
        Returns:
            tuple: Paths to the training and testing data files.
        Raises:
            CustomException: If any error occurs during the ingestion process.
        """
        logging.info("Data Ingestion Started")

        try:
            df =pd.read_csv("notebook\data\stud.csv")
            logging.info("Dataset Read as a Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header = True)
            logging.info("Raw Data Saved")

            logging.info("Train Test Split Initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data Ingestion Completed!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
             

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)