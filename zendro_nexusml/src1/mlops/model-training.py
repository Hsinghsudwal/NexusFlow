def _train_gradient_boosting(self, X_train, y_train):
        """
        Train a Gradient Boosting classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            
        Returns:
            GradientBoostingClassifier: Trained model
        """
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        
        # Initialize model
        gb = GradientBoostingClassifier(random_state=42)
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            gb, 
            param_grid=param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log hyperparameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        return best_model
    
    def _train_logistic_regression(self, X_train, y_train):
        """
        Train a Logistic Regression classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            
        Returns:
            LogisticRegression: Trained model
        """
        # Define hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
        
        # Initialize model
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            lr, 
            param_grid=param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log hyperparameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        return best_model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Cannot evaluate: model not trained")
            raise ValueError("Model is not trained")
        
        logger.info(f"Evaluating model on {X_test.shape[0]} test samples")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics to MLflow
        with mlflow.start_run(experiment_id=self.experiment_id):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Create and log confusion matrix
            self._log_confusion_matrix(y_test, y_pred)
            
            # Create and log feature importance plot if available
            if hasattr(self.model, 'feature_importances_'):
                self._log_feature_importance()
        
        logger.info(f"Model evaluation results: {metrics}")
        return metrics
    
    def _log_confusion_matrix(self, y_true, y_pred):
        """
        Create and log confusion matrix visualization.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=False,
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save figure
        confusion_matrix_path = os.path.join(self.config['model_dir'], 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        
        # Log to MLflow
        mlflow.log_artifact(confusion_matrix_path)
    
    def _log_feature_importance(self):
        """
        Create and log feature importance visualization.
        """
        feature_importances = self.model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(feature_importances)[::-1]
        
        # Get feature names if available, otherwise use indices
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), feature_importances[indices], align='center')
        plt.xticks(
            range(len(indices)), 
            [feature_names[i] for i in indices], 
            rotation=90
        )
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        
        # Save figure
        feature_importance_path = os.path.join(self.config['model_dir'], 'feature_importance.png')
        plt.savefig(feature_importance_path)
        
        # Log to MLflow
        mlflow.log_artifact(feature_importance_path)
    
    def save_model(self):
        """
        Save the trained model to disk.
        
        Returns:
            str: Path where model is saved
        """
        if self.model is None:
            logger.error("Cannot save model: not trained")
            raise ValueError("Model is not trained")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.config['model_dir']
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """
        Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            object: Loaded model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
