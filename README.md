# FPL Points Predictor (Automated Version)

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://dragonballsuper-1995.github.io/FPL_Points_Predictor_Dynamic/) 
[![Daily Prediction Update](https://github.com/Dragonballsuper-1995/FPL_Points_Predictor_Dynamic/actions/workflows/update_predictions.yml/badge.svg)](https://github.com/Dragonballsuper-1995/FPL_Points_Predictor_Dynamic/actions/workflows/update_predictions.yml)

A fully automated web-based dashboard for Fantasy Premier League (FPL) managers. This project uses machine learning and a GitHub Action to fetch live data, run predictions, and update the website daily.



## ✨ Key Features

* **Player vs. Player Comparison:** A side-by-side dashboard to compare player stats, upcoming fixtures, and predicted points.
* **Dynamic Watchlists:** Automatically generated "Top 5" lists for Forwards, Midfielders, Defenders, Goalkeepers, and "Best Value" (points-per-million).
* **Optimal Squad Builder:** A tool to build a 15-man squad within the £100m budget, based on the highest predicted points for a selected formation.
* **Visual Pitch Display:** See your optimized "Starting XI" in a familiar pitch layout, complete with Captain (C), Vice-Captain (V), and Top 3 (★) badges.
* **List View Toggle:** Switch between the visual pitch display and a simple list view for your squad.
* **Multi-Model Predictions:** Leverages 5 different ML models (XGBoost, Random Forest, Ridge, Lasso, SVR) to provide a range of predictions.
* **Fully Automated:** A GitHub Action runs daily to fetch new data and update all predictions.

## 🛠️ Tech Stack

* **Backend (Data & ML):** Python
    * `pandas` for data manipulation.
    * `scikit-learn` & `xgboost` for machine learning models.
* **Frontend:** HTML, CSS, JavaScript
    * `Tailwind CSS` for styling.
    * `Chart.js` for data visualization.
* **Automation & Deployment:**
    * `GitHub Actions` for CI/CD and automated data processing.
    * `GitHub Pages` for static site hosting.

---

## 🤖 How It Works: The Automated Workflow

This repository is a fully automated system that requires zero manual data updates to stay current.

1.  **Daily Schedule:** A GitHub Action workflow (defined in `.github/workflows/update_predictions.yml`) runs on a daily schedule (at 08:00 UTC).
2.  **Fetch Live Data:** The action runs `main.py`, which uses `config.py` to fetch the latest FPL data (stats, fixtures, team data) directly from the [Vaastav FPL data repository](https://github.com/vaastav/Fantasy-Premier-League/tree/master/data) URLs.
3.  **Train & Predict:** The action first runs `train_offline.py` to ensure models are up-to-date, then runs `main.py` to generate a new `predictions.json` file using the live data.
4.  **Auto-Commit:** The GitHub Action automatically checks if `predictions.json` has changed. If it has, it commits the new file back to the `main` branch.
5.  **Auto-Deploy:** This push to `main` automatically triggers GitHub Pages to redeploy the website, making the fresh predictions live for all users.

## 🚀 How to Deploy Your Own Automated Version

1.  **Fork** this repository.
2.  Go to the **"Actions"** tab of your forked repository and click the **"I understand my workflows, go ahead and enable them"** button.
3.  Go to the **"Settings"** tab > **"Pages"**.
4.  Under "Build and deployment," set the "Source" to **"Deploy from a branch"**.
5.  Select the **`main`** branch and the **`/(root)`** folder. Click **Save**.
6.  (Optional) To run your first update immediately, go to the **"Actions"** tab, click **"Daily Prediction Update"** on the left, and use the **"Run workflow"** button.

Your new live site will be deployed within a few minutes.

## 🙏 Acknowledgments

This entire project is made possible by the incredible work of **Vaastav Bhatia** and all the contributors to the [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) data repository.
