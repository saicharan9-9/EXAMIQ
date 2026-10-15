# EXAMIQ — Exam Score Intelligence

EXAMIQ is a machine learning-powered exam score prediction application built with Python and Streamlit. It uses a trained XGBoost regression pipeline to predict student exam scores from academic and study-related features.
Check out It is live right now : https://examiq-bdrwrhbx75rm3jwcwb4mag.streamlit.app/

## Features

- Interactive exam score prediction.
- Automated data preprocessing through a machine learning pipeline.
- Model performance evaluation using MAE, RMSE, and R-squared.
- Interactive charts and model performance visualizations.
- Premium dark glassmorphism user interface.
- Feature importance analysis when supported by the trained model.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly

## Model Performance

Reported holdout test results from the training notebook:

| Metric | Score |
|---|---:|
| R-squared | 0.7305 |
| MAE | 7.9223 |
| RMSE | 9.8185 |

These metrics describe performance on the reported evaluation split. Individual predictions may differ from actual exam scores.

## Project Structure

```text
EXAMIQ/
├── app.py
├── model.pkl
├── Exam_Score_Prediction.csv
├── Boosting.ipynb
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deployment

Deploy EXAMIQ using [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push the project to GitHub.
2. Connect your GitHub repository to Streamlit Community Cloud.
3. Select `app.py` as the application entrypoint.
4. Deploy the application.

Ensure the saved model and all required dependencies are available.

## Disclaimer

EXAMIQ generates estimates using a machine learning model. Predictions are not guaranteed academic outcomes.

## Author

Developed as a machine learning project exploring exam score prediction, ensemble regression, and interactive data visualization.
