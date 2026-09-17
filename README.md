# Bengaluru House Price Prediction (End-to-End)

![house_price](https://github.com/Nidhushan01/house_price_prediction_using_machine_learning/assets/169471036/1be84106-1a8b-4af7-a428-4739b62929d4)

## Project Overview
This project predicts residential property prices in Bengaluru using a machine learning regression model and serves predictions through a Flask web app.

The workflow includes:
- data cleaning and feature engineering in a notebook,
- model training and artifact generation,
- a simple web UI that submits form data to a Flask prediction endpoint.

## Key Features
- End-to-end notebook workflow from raw CSV to trained model.
- Feature engineering for location, BHK, bathroom count, and total square feet.
- Outlier handling steps tailored to Bengaluru housing data.
- Flask-based inference API with browser form integration.
- Pre-generated model artifacts included in `model/`.

## Repository Structure
- `/home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning/training_model.ipynb`  
  Notebook for data loading, preprocessing, model training, and exporting artifacts.
- `/home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning/flask/app.py`  
  Flask app that loads model artifacts and exposes prediction routes.
- `/home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning/flask/templates/index.html`  
  Front-end form for entering property inputs (`sqft`, `location`, `bhk`, `bath`).
- `/home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning/flask/static/style.css`  
  Basic styling for the prediction page.
- `/home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning/model/columns.json`  
  Ordered feature/column list used for one-hot encoded inference input.
- `/home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning/model/banglore_home_prices_model.pickle`  
  Serialized trained regression model used by the Flask app.

## Dataset and Preprocessing Summary (from notebook)
The notebook uses Bengaluru house-price data from a CSV named **`Bengaluru_House_Data.csv`**.

High-level preprocessing flow:
1. Load raw dataset (`df1`) with **13,320 rows** and 9 columns.
2. Keep selected columns: `location`, `size`, `total_sqft`, `bath`, `price`.
3. Drop rows with missing values in the selected columns.
4. Extract numeric **BHK** from the `size` text field (for example, `2 BHK` -> `2`).
5. Convert `total_sqft` values, including ranges such as `2100 - 2850`, into numeric values.
6. Engineer `price_per_sqft` and apply outlier reduction rules.
7. Reduce sparse location categories by mapping low-frequency locations to `other`.
8. One-hot encode location and train the regression model.
9. Export artifacts: model pickle and feature columns JSON.

## Model Serving and API Behavior
Implemented in `flask/app.py`:

- **GET `/`**  
  Renders `index.html` with available locations loaded from `columns.json`.

- **POST `/predict`**  
  Expects form fields:
  - `sqft` (float)
  - `location` (string)
  - `bhk` (integer)
  - `bath` (integer)

  Returns JSON:
  ```json
  {"estimated_price": 123.45}
  ```

### Example Request/Response
Request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "sqft=1200" \
  -d "location=whitefield" \
  -d "bhk=2" \
  -d "bath=2"
```

Example response:
```json
{"estimated_price": 67.89}
```

## Prerequisites
- Python 3.8+
- `pip`
- `venv` (recommended)
- Jupyter Notebook (to run `training_model.ipynb`)

## Installation and Local Run (Flask)
From repository root:

```bash
cd /home/runner/work/house_price_prediction_using_machine_learning/house_price_prediction_using_machine_learning
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install flask numpy pandas scikit-learn jupyter matplotlib
```

Run the Flask app:

```bash
cd flask
python app.py
```

Open: `http://127.0.0.1:5000/`

## Notebook Usage
1. Place the dataset file as `Bengaluru_House_Data.csv` where `training_model.ipynb` can read it.
2. Open and run `training_model.ipynb` from top to bottom.
3. The notebook writes model artifacts (`banglore_home_prices_model.pickle`, `columns.json`) after training.
4. Move or reference those artifacts from the `model/` directory used by the Flask app.

## Notes, Assumptions, and Limitations
- `flask/app.py` currently loads artifacts using hard-coded absolute Windows paths:
  - `C:\Users\ASUS\Desktop\home price\model\columns.json`
  - `C:\Users\ASUS\Desktop\home price\model\banglore_home_prices_model.pickle`

  For local execution on another machine, update these paths to your local repository paths (for example, files under `model/`).
- Prediction output is returned in the same unit used during training target (`price` from the dataset, displayed in UI as lakhs).
- Input validation in `/predict` is minimal; invalid/missing form values can raise runtime errors.
