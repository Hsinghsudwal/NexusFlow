# Pipeline class that ties everything together
class Pipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def run(self):
        # Step 1: Load the data
        loader = LoadData(self.data_path)
        data = loader.load()
        
        # Step 2: Preprocess the data
        preprocessor = PreprocessData(data)
        X, y = preprocessor.preprocess()
        
        # Step 3: Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 4: Train the model
        trainer = ModelTrainer(X_train, y_train)
        model = trainer.train()
        
        # Step 5: Evaluate the model
        evaluator = ModelEvaluator(model, X_test, y_test)
        evaluator.evaluate()


# Define steps using decorators
@step
def load_data(data_path: str) -> Any:
    import pandas as pd
    return pd.read_csv(data_path)

@step
def preprocess(data):
    # Data preprocessing logic
    return processed_data

# Connect steps in a pipeline
data = load_data().connect(data_path="data/raw/data.csv")
processed_data = preprocess().connect(data=data)

# Create and run pipeline
pipeline = Pipeline(name="data_pipeline")
pipeline.add_step(data)
pipeline.add_step(processed_data)
