# FinTech Dashboard & Predictive Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fintech-dashboard-predictive-analytics.streamlit.app/)

This project is a comprehensive platform for visualizing financial data and leveraging machine learning to generate predictive insights. The interactive dashboard, built with Streamlit, provides users with tools to analyze market trends, assess risk, and make data-driven decisions using historical stock data and predictive models.

<img width="1882" height="919" alt="image" src="https://github.com/user-attachments/assets/5acfbbe0-4e71-45be-b2c5-98d63647bfbe" />
<img width="1899" height="889" alt="image" src="https://github.com/user-attachments/assets/fe808ad4-886b-45d8-b603-0da470bdb89b" />
<img width="1909" height="823" alt="image" src="https://github.com/user-attachments/assets/663f6d0f-be2c-4c87-bf41-a9ee037e04db" />
<img width="1898" height="896" alt="image" src="https://github.com/user-attachments/assets/49937df3-fad4-432b-b18b-305e4274fa9e" />
<img width="1901" height="822" alt="image" src="https://github.com/user-attachments/assets/2b69bf3e-6891-40e0-beca-20464712adfa" />


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

In the fast-paced world of finance, timely and accurate data analysis is crucial. This project aims to provide a powerful yet intuitive tool for financial analysts, traders, and enthusiasts. By combining a real-time data fetching mechanism with a suite of analytical and predictive models, the platform empowers users to:

- **Visualize** historical stock performance and key technical indicators.
- **Analyze** market trends and volatility.
- **Assess** potential credit risk or detect fraudulent activities through integrated machine learning models.

---

## Features

- **Interactive Dashboard**: A user-friendly web interface built with **Streamlit** to visualize complex financial datasets, including candlestick charts, moving averages, and trading volumes.
- **On-Demand Data**: Fetches historical stock market data using the `yfinance` library and caches it locally for performance.
- **Technical Analysis**: Implements various technical indicators (e.g., SMA, EMA, RSI) using the `ta` library to help users identify trading signals.
- **Volatility Analysis**: Analyzes and visualizes stock volatility using a **GARCH(1,1)** model to understand risk.
- **Portfolio Optimization**: Calculates optimal portfolio allocations based on **Markowitz Portfolio Theory**, visualizing the Efficient Frontier for multiple assets.
- **Predictive Modeling**:
  - **Directional Prediction**: Predicts the next day's price direction (Up/Down) using a **Random Forest Classifier** trained on technical indicators.
  - **Credit Risk Assessment**: (Placeholder) A trained model to predict the likelihood of loan defaults based on applicant data.
  - **Fraud Detection**: (Placeholder) A model to identify potentially fraudulent transactions.

---

## Tech Stack

- **Backend**: Python
- **Dashboard**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Keras
- **Data Fetching**: yfinance
- **Financial Modeling**: arch (for GARCH)
- **Technical Analysis**: ta
- **Data Visualization**: Plotly, Matplotlib, Seaborn

---

## Project Structure

The repository is organized as follows:

```
.
├── data/                 # Contains raw and processed datasets
├── fintech-dashboard/    # Source code for the Streamlit dashboard
│   └── app.py            # The main application script
├── models/               # Saved (pickled) machine learning models
├── notebooks/            # Jupyter notebooks for analysis and model training
├── .gitignore            # Files and directories to be ignored by Git
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

---

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

### 1. Prerequisites

- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/downloads/)

### 2. Clone the Repository

```bash
git clone https://github.com/diwakar2905/FinTech-Dashboard-Predictive-Analytics.git
cd FinTech-Dashboard-Predictive-Analytics
```

### 3. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project-specific dependencies and avoid conflicts.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows (PowerShell):
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## How to Run

Once the setup is complete, you can run the Streamlit dashboard.

1.  Make sure your virtual environment is activated.
2.  Run the `app.py` script using Streamlit.

```bash
streamlit run fintech-dashboard/app.py
```

3.  Open your web browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

---

## Usage

Once the application is running, you can interact with the dashboard through the sidebar and main panel.

1.  **Single-Stock Analysis**:
    -   Use the sidebar dropdown to select a stock ticker (e.g., MSFT, AAPL).
    -   The main panel will display the candlestick chart, GARCH volatility analysis, and the next-day price direction forecast for the selected stock.

2.  **Portfolio Optimization**:
    -   In the sidebar, enter a comma-separated list of stock tickers for your desired portfolio (e.g., `AAPL,MSFT,GOOGL,AMZN`).
    -   The main panel will display the Efficient Frontier graph and provide the optimal portfolio weights for both maximum Sharpe ratio and minimum volatility.

---

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` file for more information.
