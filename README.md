# Customer Satisfaction MLOps

## Overview
This repository contains a comprehensive MLOps project focused on customer satisfaction. The project covers the entire machine learning lifecycle, from data ingestion to deployment, using cutting-edge tools such as MLflow, ZenML, and other advanced technologies.

### Prerequisites

- Python 3.7+
- Docker
- MLflow
- ZenML

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/arifzanko/CustomerSatisfaction_MLOps.git
    cd CustomerSatisfaction_MLops
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```



### Usage

1. **Data Ingestion**: Place raw data files in `data/` and execute scripts in `steps/ingest_data.py`.

2. **Data Cleaning**: Execute preprocessing scripts in `steps/clean_data.py`.

3. **Model Training**: Run training scripts in `steps/model_train.py`. MLflow will track experiments and store artifacts.

4. **Model Evaluation**: Evaluate model performance using scripts in `steps/evaluation.py`.

5. **Deployment**: Deploy the model using deployment scripts in `steps/deployment.py`.