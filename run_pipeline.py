from pipeline.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"C:\Users\arifz\Downloads\customer-satisfaction-mlops-main\CustomerSatisfaction_MLOps\data\olist_customers_dataset.csv")