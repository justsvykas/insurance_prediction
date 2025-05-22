# Travel Insurance Prediction

This project goal is to model: "Which customers to target for travel insurance? ". To do that I used [Travel Insurance Prediction dataset](https://www.kaggle.com/datasets/teerthsharma/travel-insurance-prediction-data-set) from Kaggle which contains 1688 samples of customers with 8 features and target variable (0 or 1) indicating whether the customer purchased travel insurance.

# Main results

In [main.ipynb](./src/main.ipynb) you can find the processing of the data and final model evaluation. After careful consideration in [Exploratory_analysis.ipynb](./src/Exploratory_analysis.ipynb) I used XGBoostClassifier and tested it on the test set at following metrics:
- Model correctly predicts 82.3% of customers who will buy or will not buy insurance. But data has class imbalance, so it is better to look at precision and recall.
- When model predicts that customer will buy insurance it is correct 84.6% of the time.
- The model identifies 61.7% of all actual insurance buyers.

This model can be particularly useful for targeted marketing campaigns where the focus is on customers who are more likely to buy insurance. Note that target variable is imbalanced thus I stratify the data for train, val and test sets using split_data.py script.

# Exploratory analysis
The results of exploratory analysis are in [Exploratory_analysis.ipynb](./src/Exploratory_analysis.ipynb).

I used this dataset to do exploratory modeling: I fitted a logistic regression model and found that income, frequent flyer significantly affect travel insurance purchase likelihood and chronic diseases do not at confidence level 0.05 confidence level.

In the notebook I answered the following questions extensively:
- Which customers to target as they are more likely to buy travel insurance?
- How should I process the data?
- Which features should I include in model?
- Which machine learning algorithms show the most promise for prediction task?
- Which model and what hyperparameters to choose to predict travel insurance purchases?
- What biases are present in the data?

# Installation
Below is the process to install this project on your local machine. However, there is **main.html** and **Exploratory_analysis.html** files in the root of the project that can be used to view the analysis without installing the project. Just download the file and open it in your browser.

This analysis is structured to be easily continued by another developer, with dependency management handled via the uv library. It follows consistent coding standards, enforced using Ruff for linting and pre-commit hooks. To further ease distribution, this project is packaged for ease of use.

After placing yourself in your desired directory, run this command in your terminal to copy this repo.
```bash
git clone https://github.com/jusvyka/travel-insurance-prediction
```
Go to project directory.
```bash
cd travel-insurance-prediction
```
install package and create virtual environment
```bash
uv sync
```
Open main analysis file
```bash
code notebooks/main.ipynb
```
Run the notebook cells sequentially or review the precomputed outputs.

# Deployment

To run the API locally, run:
```bash
cd uv_app
python api.py
```
then go to http://localhost:8000/docs to see the API documentation with example request for predict endpoint.

To build a containerized version of the project, run below at the root of the project:
```bash
docker build --network host -t insurance-prediction .
```

then run docker container:
```bash
docker run insurance-prediction
```
then go to http://localhost:8000/docs to see the API documentation with example request for predict endpoint.


# Deployed on cloud

This model is deploed on Render. To see example of predictions go to https://insurance-prediction-nc3h.onrender.com/docs

to test model in cloud run:
```bash
poe integration_tests
```

# Optional commands

To run tests locally, run:
```bash
poe tests
```

to lint, sort and format the code, run:
```bash
poe x
```

Note that there is script to split the data into train, val and test sets to avoid data leakage. To create the splits run:
```bash
python src/split_data.py
```

To initialize pre-commit hooks run:
```bash
pre-commit install
```

# Dependencies

- Python 3.11
- uv
- docker

# Project structure

```bash
notebooks/            # Additional notebooks
├── main.ipynb       # Main notebook
├── Exploratory_analysis.ipynb  # EDA notebook
├── split_data.py    # Data splitting
├── models/          # Model files
├── data/           # Data files
└── utils/          # Utility functions

uv_app/                # FastAPI service
├── api.py            # API implementation
├── schemas.py        # Data schemas
├── settings.py       # Configuration
├── __init__.py       # Package initialization
└── full_pipeline.pkl # Saved model pipeline

tests/               # Test files
├── test_api.py     # API tests
└── __init__.py     # Test package initialization

Dockerfile          # Container config
pyproject.toml      # Dependencies
main.html           # Main analysis (HTML)
Exploratory_analysis.html  # EDA (HTML)
.dockerignore       # Docker ignore
.gitignore          # Git ignore
.pre-commit-config.yaml # Pre-commit config
README.md           # Documentation
```
