from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys
from sensor.utils import dump_csv_file_to_mongodb_collection

'''def test_exception():
    try:
        logging.info("Exception raised")
        a=1/0
    except Exception as e:
        raise SensorException(e,sys)'''



if __name__ == "__main__":
    file_path = r"C:\Users\Lenovo\OneDrive\Desktop\sensor live\Llivesensor\aps_failure_training_set1.csv"
    database_name="aps_fault_sensor"
    collection_name="sensor"
    dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)



