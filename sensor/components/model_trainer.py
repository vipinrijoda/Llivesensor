from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
import os,sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object,load_object
import json

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)

    def get_reduced_param_grid(self):
        """
        Returns a REDUCED parameter grid for XGBoost only
        This is much lighter on memory
        """
        try:
            # REDUCED parameter grid - only 24 combinations instead of 864
            param_grid = {
                'n_estimators': [100, 200],  # Reduced from 3 to 2
                'max_depth': [3, 5, 7],       # Reduced from 4 to 3
                'learning_rate': [0.05, 0.1], # Reduced from 4 to 2
                'subsample': [0.8, 1.0],      # Reduced from 3 to 2
                'colsample_bytree': [0.8, 1.0], # Reduced from 3 to 2
                'gamma': [0, 0.1]              # Reduced from 3 to 2
            }
            # Total combinations: 2*3*2*2*2*2 = 96 (still high)
            # We'll use RandomizedSearchCV instead of GridSearch
            return param_grid
            
        except Exception as e:
            raise SensorException(e, sys)

    def perform_lightweight_tuning(self, x_train, y_train, cv=3, n_iter=10):
        """
        Lightweight hyperparameter tuning - uses RandomizedSearch with fewer iterations
        """
        try:
            logging.info("Starting LIGHTWEIGHT hyperparameter tuning")
            
            # Use only XGBoost (most efficient for this use case)
            model = XGBClassifier(random_state=42, eval_metric='logloss', 
                                 n_jobs=1,  # Use single job to prevent memory issues
                                 verbosity=0)
            
            # Get reduced parameter grid
            param_grid = self.get_reduced_param_grid()
            
            # Define scoring metric
            f1_scorer = make_scorer(f1_score, average='macro')
            
            # Use RandomizedSearchCV with FEW iterations
            from sklearn.model_selection import RandomizedSearchCV
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,  # Small number of iterations (10-20)
                cv=cv,           # Reduced cross-validation folds (3 instead of 5)
                scoring=f1_scorer,
                n_jobs=1,        # Single job to prevent memory overload
                verbose=1,
                random_state=42,
                return_train_score=False  # Don't store train scores to save memory
            )
            
            # Fit the search (this will still take time but less memory)
            logging.info(f"Fitting RandomizedSearch with {n_iter} iterations...")
            search.fit(x_train, y_train)
            
            # Log results
            logging.info(f"Best parameters found: {search.best_params_}")
            logging.info(f"Best cross-validation score: {search.best_score_:.4f}")
            
            # Save only best parameters (not all results)
            self.save_best_params_only(search.best_params_, search.best_score_)
            
            return search.best_estimator_
            
        except Exception as e:
            raise SensorException(e, sys)

    def save_best_params_only(self, best_params, best_score):
        """
        Save only the best parameters (lightweight, no CSV)
        """
        try:
            # Create results directory
            results_dir = os.path.join(os.path.dirname(self.model_trainer_config.trained_model_file_path), 'tuning_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save as JSON (lighter than CSV)
            best_params_file = os.path.join(results_dir, 'best_params.json')
            
            data = {
                'best_params': best_params,
                'best_score': float(best_score),
                'model_type': 'XGBClassifier'
            }
            
            with open(best_params_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logging.info(f"Best parameters saved to {best_params_file}")
                    
        except Exception as e:
            logging.warning(f"Could not save best parameters: {e}")

    def train_model(self, x_train, y_train):
        """
        Memory-efficient training with optional tuning
        """
        try:
            # Check data size
            data_size_mb = x_train.nbytes / (1024 * 1024)
            logging.info(f"Training data size: {data_size_mb:.2f} MB")
            
            # Decide tuning strategy based on data size
            if data_size_mb > 100:  # If data > 100MB, be very conservative
                logging.info("Large dataset detected. Using minimal tuning...")
                
                # Train with default parameters first
                model = XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    n_jobs=1,
                    n_estimators=100,  # Default
                    max_depth=5,        # Default
                    learning_rate=0.1   # Default
                )
                model.fit(x_train, y_train)
                
            else:
                # Use lightweight tuning for smaller datasets
                logging.info("Using lightweight hyperparameter tuning...")
                
                # Use very conservative tuning parameters
                try:
                    model = self.perform_lightweight_tuning(
                        x_train, y_train, 
                        cv=3,        # Only 3 folds
                        n_iter=5     # Only 5 random combinations
                    )
                except MemoryError:
                    logging.warning("Memory error during tuning. Falling back to default model...")
                    model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)
                    model.fit(x_train, y_train)
                except Exception as e:
                    logging.warning(f"Tuning failed: {e}. Using default model...")
                    model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)
                    model.fit(x_train, y_train)
            
            return model
            
        except Exception as e:
            raise SensorException(e, sys)
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"{'='*60}")
            logging.info("Starting Model Trainer (Memory-Optimized Version)")
            logging.info(f"{'='*60}")
            
            # Load data
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Log memory usage
            train_memory = train_arr.nbytes / (1024 * 1024)
            test_memory = test_arr.nbytes / (1024 * 1024)
            logging.info(f"Training data memory: {train_memory:.2f} MB")
            logging.info(f"Testing data memory: {test_memory:.2f} MB")
            logging.info(f"Training data shape: {train_arr.shape}")
            logging.info(f"Testing data shape: {test_arr.shape}")

            # Split data
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Use data sampling if dataset is too large
            total_samples = len(x_train)
            if total_samples > 50000:
                logging.info(f"Large dataset detected ({total_samples} samples). Using 50% for tuning...")
                # Use subset for tuning to save memory
                sample_size = min(25000, total_samples // 2)
                indices = np.random.choice(total_samples, sample_size, replace=False)
                x_train_subset = x_train[indices]
                y_train_subset = y_train[indices]
            else:
                x_train_subset = x_train
                y_train_subset = y_train

            # Train model with memory-efficient approach
            model = self.train_model(x_train_subset, y_train_subset)
            
            # Predict on full training set
            logging.info("Making predictions on full training set...")
            
            # Predict in batches to save memory
            batch_size = 10000
            n_batches = int(np.ceil(len(x_train) / batch_size))
            y_train_pred = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_train))
                batch_pred = model.predict(x_train[start_idx:end_idx])
                y_train_pred.extend(batch_pred)
            
            y_train_pred = np.array(y_train_pred)
            
            # Calculate metrics
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            logging.info(f"Training F1 Score: {classification_train_metric.f1_score:.4f}")
            
            # Check accuracy threshold
            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                raise Exception(f"Model not good enough. Expected F1 > {self.model_trainer_config.expected_accuracy}, got {classification_train_metric.f1_score:.4f}")
            
            # Predict on test set (also in batches)
            logging.info("Making predictions on test set...")
            n_test_batches = int(np.ceil(len(x_test) / batch_size))
            y_test_pred = []
            
            for i in range(n_test_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_test))
                batch_pred = model.predict(x_test[start_idx:end_idx])
                y_test_pred.extend(batch_pred)
            
            y_test_pred = np.array(y_test_pred)
            
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            logging.info(f"Testing F1 Score: {classification_test_metric.f1_score:.4f}")

            # Check overfitting
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            logging.info(f"Train-Test F1 difference: {diff:.4f}")
            
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(f"Model may be overfitting. Difference: {diff:.4f}")

            # Load preprocessor and save model
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)
            
            # Save minimal model info
            self.save_model_info(model)
            
            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            logging.info(f"Model trainer completed successfully")
            logging.info(f"{'='*60}")
            
            return model_trainer_artifact
            
        except MemoryError as e:
            logging.error(f"Memory Error: {e}")
            logging.error("Try reducing batch size or using a smaller dataset")
            raise SensorException("Out of memory. Please use a smaller dataset or increase RAM.", sys)
        except Exception as e:
            raise SensorException(e, sys)
    
    def save_model_info(self, model):
        """
        Save minimal model info (lightweight)
        """
        try:
            info_dir = os.path.join(os.path.dirname(self.model_trainer_config.trained_model_file_path), 'model_info')
            os.makedirs(info_dir, exist_ok=True)
            
            info_file = os.path.join(info_dir, 'model_info.txt')
            
            with open(info_file, 'w') as f:
                f.write("Model Type: XGBClassifier\n")
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    f.write("\nKey Parameters:\n")
                    for key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample']:
                        if key in params:
                            f.write(f"{key}: {params[key]}\n")
                
        except Exception as e:
            logging.warning(f"Could not save model info: {e}")