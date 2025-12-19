#  Nigerian House Price Predictor

## Project Overview
This project applies Machine Learning to predict real estate prices in Nigeria. 
Using a dataset of Nigerian housing listings, I built a **Gradient Boosting Regressor** model that estimates prices based on location, house type, and amenities (bedrooms, parking, etc.).

The project includes a **Streamlit Web App** for easy interaction.

## Features
- **Data Cleaning:** Removed extreme outliers (e.g., erroneous Trillion Naira listings).
- **EDA:** Visualized market trends, price distribution, and location impact.
- **Model:** Trained a Gradient Boosting model with **~60% Accuracy**.
- **App:** User-friendly interface to get instant price predictions.

##  File Structure
- `app.py`: The Streamlit application code.
- `notebook.ipynb`: The Jupyter Notebook containing all Data Cleaning, EDA, and Training steps.
- `requirements.txt`: List of dependencies.
- `*.pkl`: Saved model files.

##  How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📊 Key Insights
- **Town** is the #1 driver of price in Nigeria.
- The market is heavily **Right-Skewed**, with a vast difference between standard and luxury homes.
