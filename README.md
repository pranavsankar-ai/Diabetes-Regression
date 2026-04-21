# Diabetes Progression Regression Model

This repository contains a machine learning model to predict diabetes progression based on various health metrics. The project includes a Jupyter Notebook for model training and analysis, and a Flask application for serving predictions via a web interface.

## Features

- **Data Loading and Preprocessing**: Utilizes the `scikit-learn` diabetes dataset.
- **Model Training**: Employs a Ridge Regression model for robust prediction.
- **Feature Scaling**: Uses `StandardScaler` to normalize input features.
- **Interactive Web Application**: A Flask-based web interface for real-time predictions with a modern Bootstrap UI.
- **Model Persistence**: Trained model and scaler are saved using `joblib` for easy deployment.

## Project Structure

- `diabetes-regression.ipynb`: Jupyter Notebook detailing data exploration, model training, and evaluation.
- `train_model.py`: Python script to train and save the Ridge Regression model and `StandardScaler`.
- `app.py`: Flask application to serve the prediction model via a web interface.
- `diabetes_model.pkl`: Trained Ridge Regression model.
- `scaler.pkl`: Fitted `StandardScaler` object.
- `requirements.txt`: List of Python dependencies.

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pranavsankar-ai/Diabetes-Regression.git
    cd Diabetes-Regression
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model (if not already trained or if you want to retrain):**
    ```bash
    python train_model.py
    ```
    This script will train the model using all available features and save `diabetes_model.pkl` and `scaler.pkl`.

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will be accessible at `http://127.0.0.1:5000`.

## Usage

Navigate to the Flask application in your web browser. You will see a form where you can input values for the ten diabetes features (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6). Enter the values and click "Predict Progression" to get the predicted diabetes progression score.

## Model Improvements

- **Full Feature Utilization**: The model now utilizes all ten features from the diabetes dataset (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6) for a more comprehensive prediction, instead of just BMI.
- **Improved Model**: Switched from a basic Linear Regression to Ridge Regression, which includes L2 regularization to prevent overfitting and improve generalization on unseen data.
- **Standardized Scaling**: Implemented `StandardScaler` for all input features, ensuring that each feature contributes equally to the model training process.
- **Enhanced Web Interface**: The Flask application now features a modern and user-friendly interface built with Bootstrap, allowing users to input all ten features and receive predictions.

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details. (Note: A LICENSE.md file is not currently in the repo, but it's good practice to include one.)
