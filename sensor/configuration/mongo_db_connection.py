import pymongo
from dotenv import load_dotenv
from sensor.constant.database import DATABASE_NAME
from sensor.constant.env_variable import MONGODB_URL_KEY
import certifi
import os
ca = certifi.where()
import logging

load_dotenv()
class MongoDBClient:
    client = None
    
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                logging.info(f"Retrieved MongoDB URL: {mongo_db_url}")
                
                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)  # TLS/SSL
                
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            logging.info(f"Connected to database: {database_name}")
            
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise