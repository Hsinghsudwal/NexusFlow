


# Example steps
def load_data():
    return [1, 2, 3, 4, 5]

def process_data(data):
    return [x * 2 for x in data]

def train_model(processed_data):
    return sum(processed_data) / len(processed_data)  # Mock model

# Define steps
load_step = Step("load_data", load_data)
process_step = Step("process_data", process_data, dependencies=["load_data"])
train_step = Step("train_model", train_model, dependencies=["process_data"])

# Run pipeline
if __name__ == "__main__":
    pipeline = Pipeline([load_step, process_step, train_step])
    pipeline.run()
