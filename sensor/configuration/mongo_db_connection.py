import pymongo
from dotenv import load_dotenv
from sensor.constant.database import DATABASE_NAME
from sensor.constant.env_variable import MONGODB_URL_KEY
import certifi
import os
import logging

ca = certifi.where()
load_dotenv()

class MongoDBClient:
    client = None
    
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # Get MongoDB URL from environment variables FIRST
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                
                # Try Streamlit secrets ONLY if running on Streamlit Cloud
                if mongo_db_url is None:
                    try:
                        import streamlit as st
                        if hasattr(st, 'secrets') and MONGODB_URL_KEY in st.secrets:
                            mongo_db_url = st.secrets[MONGODB_URL_KEY]
                            logging.info("Using MongoDB URL from Streamlit secrets")
                    except:
                        pass
                
                # Final fallback to localhost
                if mongo_db_url is None:
                    mongo_db_url = "mongodb://localhost:27017/"
                    logging.warning("Using default local MongoDB URL")
                
                logging.info(f"Connecting to MongoDB...")
                
                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            logging.info(f"Connected to database: {database_name}")
            
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise e