from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from sensor.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelPusherConfig,ModelEvaluationConfig,ModelTrainerConfig
from sensor.exception import SensorException
import sys,os
from sensor.logger import logging
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from datetime import datetime

from sensor.constant.training_pipeline import SAVED_MODEL_DIR

class TrainPipeline:
    is_pipeline_running=False
    
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
    def create_directories(self):
        """Create all necessary directories before starting pipeline"""
        try:
            # Create main directories
            directories = [
                "logs",
                "artifacts",
                "saved_models",
                os.path.join("artifacts", "data_ingestion"),
                os.path.join("artifacts", "data_validation"),
                os.path.join("artifacts", "data_transformation"),
                os.path.join("artifacts", "model_trainer"),
                os.path.join("artifacts", "model_evaluation"),
                os.path.join("artifacts", "model_pusher"),
            ]
            
            for dir_path in directories:
                os.makedirs(dir_path, exist_ok=True)
                # REMOVED EMOJI - using plain text
                logging.info(f"Created directory: {dir_path}")
                
            # Create a test log file to verify write permissions
            test_log_path = os.path.join("logs", f"test_{datetime.now().strftime('%m%d%Y_%H%M%S')}.log")
            with open(test_log_path, 'w') as f:
                f.write("Test log file - directory is writable")
            os.remove(test_log_path)
            # REMOVED EMOJI - using plain text
            logging.info("Logs directory is writable")
            
        except Exception as e:
            logging.error(f"Error creating directories: {e}")
            raise SensorException(e, sys)

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def start_data_validaton(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
            )
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except  Exception as e:
            raise  SensorException(e,sys)
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def run_pipeline(self):
        try:
            # Create all necessary directories first
            self.create_directories()
            
            TrainPipeline.is_pipeline_running=True
            logging.info("="*60)
            logging.info("Starting Training Pipeline")
            logging.info("="*60)

            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            logging.info("Data Ingestion completed")
            
            data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Data Validation completed")
            
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            logging.info("Data Transformation completed")
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logging.info("Model Trainer completed")
            
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            logging.info("Model Evaluation completed")
            
            if not model_eval_artifact.is_model_accepted:
                logging.warning("Trained model is not better than the best model")
                raise Exception("Trained model is not better than the best model")
            
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            logging.info("Model Pusher completed")
            
            TrainPipeline.is_pipeline_running=False
            logging.info("="*60)
            logging.info("Training Pipeline Completed Successfully")
            logging.info("="*60)

        except Exception as e:
            TrainPipeline.is_pipeline_running=False
            logging.error(f"Training Pipeline Failed: {e}")
            raise SensorException(e, sys)