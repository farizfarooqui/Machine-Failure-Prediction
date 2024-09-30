## Machine Failure Prediction App

This is a Streamlit-based web application that predicts machine failure using a K-Nearest Neighbors (KNN) classification model. The app allows users to train a machine learning model on provided data, test it on new data, and make predictions for new machine configurations.

### Features:
- **Data Preprocessing:** Reads data from a CSV file, applies transformations such as label encoding for categorical features, and creates a new feature (`nf` = Tool wear * Torque).
- **Model Training:** The app uses KNN for classification, allowing users to train the model directly from the interface.
- **Real-time Prediction:** Users can input machine characteristics, and the app predicts whether the machine is likely to fail.
- **Custom Test Size:** Users can select a portion of the dataset for testing purposes through an interactive slider.

### Technologies Used:
- **Streamlit:** For building the interactive web application.
- **Pandas:** For data manipulation and preprocessing.
- **Scikit-learn:** For machine learning, including KNN classification, label encoding, and train-test split.

### How to Use:
1. Clone the repository and install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload the `machine_failure_prediction.csv` file, and the app will handle the preprocessing.
4. Train the model by clicking the "Start Training My Model" button.
5. Input machine details to predict failure for a new machine or use the slider to test the model on a different dataset split.

### Key Dependencies:
- `streamlit`
- `pandas`
- `scikit-learn`

### Model Explanation:
- **KNN (K-Nearest Neighbors):** The app uses KNN with `n_neighbors=1` to classify machines as either likely to fail or not based on the provided feature set.
  
### Custom Modelling:
The application leverages a `Modelling` class (imported from `modelling.py`) to handle model fitting and evaluation. The results include model accuracy and runtime.

### Data:
The app expects a CSV file (`machine_failure_prediction.csv`) with the following columns:
- **Type:** Type of machine (categorical).
- **Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]:** Machine attributes.
- **Target:** Binary target (1 for failure, 0 for non-failure).
- **Failure Type:** (Not used in the model, but available in the dataset for other purposes).

This app is a demonstration of integrating machine learning into an interactive web interface using Streamlit.

## Connect with Me

Feel free to connect with me on LinkedIn and GitHub for professional networking and to explore my projects:

<p align="left">
  <a href="https://www.linkedin.com/in/fariz-farooqui-97b48026b/" target="_blank">
    <img align="center" src="https://img.icons8.com/fluent/96/000000/linkedin.png" alt="LinkedIn - Fariz Farooqui" height="40" width="40" />
</p>
