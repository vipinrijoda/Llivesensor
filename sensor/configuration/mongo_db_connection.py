import streamlit as st # Add this import
import pymongo
import os
import certifi
import logging
from sensor.constant.database import DATABASE_NAME
from sensor.constant.env_variable import MONGODB_URL_KEY

ca = certifi.where()

class MongoDBClient:
    client = None
    
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # 1. Try to get URL from Streamlit Secrets first, then OS Env
                mongo_db_url = st.secrets.get(MONGODB_URL_KEY) or os.getenv(MONGODB_URL_KEY)
                
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set in Secrets or .env file")

                logging.info(f"Retrieved MongoDB URL")
                
                # 2. Connection Logic
                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    # Use the certificate for Atlas connections
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name # Added for clarity
            
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise e