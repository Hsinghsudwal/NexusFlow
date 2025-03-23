import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Class to load the data
class LoadData:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load(self):
        # For demonstration, we assume it's a CSV file
        self.data = pd.read_csv(self.data_path)
        print("Data loaded successfully.")
        return self.data

# Class for preprocessing the data
class PreprocessData:
    def __init__(self, data):
        self.data = data
        
    def preprocess(self):
        # Example preprocessing: dropping NA values and scaling the data
        self.data = self.data.dropna()
        X = self.data.drop(columns=["target"])  # Assume 'target' is the label column
        y = self.data["target"]
        
        # Scaling the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Data preprocessing completed.")
        return X_scaled, y

# Class to train a machine learning model
class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestClassifier(n_estimators=100)
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")
        return self.model

# Class to evaluate the model
class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy

