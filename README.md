
# 🔥 Algerian Forest Fire FWI Prediction

Predicts the **Fire Weather Index (FWI)** from weather conditions and fire-index values using a **Ridge regression** model. The project includes two web apps built on the same trained model: one with **Flask** and one with **Streamlit**.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://algerian-fwi-predictor.streamlit.app/)

## About the project

The Fire Weather Index is a numeric rating of how intense a forest fire could be under the current conditions. It comes from the Canadian Forest Fire Weather Index System. This project trains a regression model on the Algerian Forest Fires dataset and serves it through simple web interfaces, so you can enter conditions and get a predicted FWI instantly.

## Features

- Ridge regression model with feature scaling (`StandardScaler`)
- Flask web app with an HTML form for entering inputs
- Streamlit web app with number inputs, dropdowns and a clean results display
- Trained model and scaler saved as pickle files for fast loading

## Tech stack

- Python
- scikit-learn (Ridge regression, StandardScaler)
- Flask
- Streamlit
- NumPy, pandas

## Dataset

The model is trained on the [Algerian Forest Fires Dataset](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) from the UCI Machine Learning Repository. It contains daily weather and fire-index observations from two regions of Algeria, **Bejaia** and **Sidi Bel-Abbes**, recorded between June and September 2012.

## Model inputs

| Feature     | Description                                   |
| ----------- | --------------------------------------------- |
| Temperature | Temperature in °C                            |
| RH          | Relative humidity in %                        |
| Ws          | Wind speed in km/h                            |
| Rain        | Rainfall in mm                                |
| FFMC        | Fine Fuel Moisture Code                       |
| DMC         | Duff Moisture Code                            |
| ISI         | Initial Spread Index                          |
| Classes     | Fire occurred: `0` = not fire, `1` = fire |
| Region      | `0` = Bejaia, `1` = Sidi Bel-Abbes        |

The output is the predicted **FWI** value.

## How it works

```
User inputs → StandardScaler → Ridge regression → Predicted FWI
```

Both apps load `models/scaler.pkl` and `models/ridge.pkl`, scale the 9 input values in the same order used during training, and return the model's prediction.

## Project structure

```
algerian-forest-fire-fwi-prediction/
├── models/
│   ├── ridge.pkl          # Trained Ridge regression model
│   └── scaler.pkl         # Fitted StandardScaler
├── Notebook/              # Data analysis and model training notebooks
├── Templates/
│   ├── index.html         # Landing page (Flask)
│   └── home.html          # Prediction form (Flask)
├── application.py         # Flask app
├── streamlit_app.py       # Streamlit app
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/visheshsavarn/algerian-forest-fire-fwi-prediction.git
cd algerian-forest-fire-fwi-prediction
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Flask app

```bash
python application.py
```

Open `http://127.0.0.1:5000/` in your browser. The prediction form is at `http://127.0.0.1:5000/predictdata`.

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

## Notes

- Use the same scikit-learn version that was used to train the model, otherwise loading the pickle files may fail or give slightly different results.
- The model learned from a limited range of data (Algerian summer conditions), so predictions for values far outside that range may be unreliable.
- This is a learning project and should not be used for real fire-safety decisions.

## Author

**Vishesh Savarn**
GitHub: [@visheshsavarn](https://github.com/visheshsavarn)
