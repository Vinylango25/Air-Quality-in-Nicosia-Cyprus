# Sensor Calibration and Air Quality Monitoring in Nicosia, Cyprus

> A comprehensive study applying machine learning techniques to calibrate low-cost gas sensors for urban air quality monitoring in Nicosia, Cyprus.

---

## 📍 Introduction

Air pollution poses serious health risks, especially in urban environments. Accurate, high-resolution monitoring is critical for policymaking and public awareness.  
However, **reference-grade air quality analyzers** are expensive and sparse.

**Low-cost electrochemical sensors** offer a promising alternative but require **calibration** to overcome issues like:
- Sensor drift over time
- Cross-sensitivity to interfering gases
- Environmental dependency (temperature, humidity effects)

This project investigates the **machine learning calibration** of low-cost sensors measuring **CO, NO, NO₂, O₃, and NOx** using data from **Strovolou Avenue, Nicosia**, compared against Cyprus Ministry of Environment reference stations.

---

## 🎯 Project Objectives

- Evaluate raw sensor performance against regulatory standards.
- Apply and compare machine learning models for sensor calibration.
- Validate post-calibration outputs according to **EU Directive 2008/50/EC** standards.
- Analyze the impact of **temporal aggregation** (1-min vs 1-hour).
- Discuss the **transferability** and **practical deployment** of calibration models.

---

## 🛠️ Experimental Setup

- **Location:** Urban roadside station at Strovolou Avenue, Nicosia.
 <img src="fig1.png" alt="Cost Evaluation" width="850">
- **Sensors:** Electrochemical gas sensors measuring CO, NO, NO₂, O₃.
- **Reference System:** Standard EU regulatory analyzers.
- **Auxiliary Data:** Temperature and humidity readings.
- **Temporal Resolution:** Data recorded at 1-min intervals, aggregated to 1-hour for analysis.
  <img src="fg3.png" alt="Cost Evaluation" width="850">

---

## 📈 Data Sources

- Raw voltage signals from low-cost gas sensors.
- Co-located temperature and humidity measurements.
- Ground-truth reference measurements of pollutant concentrations.

---

## 🤖 Machine Learning Models Applied

1. **Linear Regression (LR)** — Baseline traditional calibration approach.
2. **Support Vector Regression (SVR)** — Captures non-linearities via kernel transformations.
3. **Random Forest Regression (RF)** — Ensemble of decision trees, strong against overfitting.
4. **Artificial Neural Networks (ANN)** — Nonlinear mappings through dense network layers.
5. **Extreme Gradient Boosting (XGBoost)** — Optimized gradient boosting for tabular datasets.

All models incorporated **temperature and humidity** as additional input features to correct environmental influences.

---

## 📏 Evaluation Metrics

- **R² (Coefficient of Determination):** Measures explained variance.
- **RMSE (Root Mean Squared Error):** Penalizes large errors heavily.
- **MAE (Mean Absolute Error):** Average magnitude of errors.
- **EU Compliance Check:** Based on minimum R² thresholds set by **EU Directive 2008/50/EC**.

---

# 📊 Results and Discussion

---

## 2.1.1 Data Splitting Schemes

![Figure 3: Data Splitting Strategies](fig3.png)

- Calibration was performed every 1, 3, and 6 months using different data splitting strategies.
- For monthly calibration, data was sampled continuously from the start of each month.
- For 3- and 6-month calibration, both continuous and interceptive sampling strategies were explored.
- Results showed minimal difference between different sampling approaches.
- Only the schemes illustrated in Figure 3 were considered for the main analysis.

---

## 2.1.2 Calibration Results Overview

![Figure 4: Correlation between Calibrated LCSs and Reference Measurements](fig4.png)

- 80% of the data was used for training and 20% for validation with 2-min time resolution.
- CO LCS exhibited good agreement with reference data even without ML calibration.
- NO₂, O₃, and SO₂ LCSs initially showed moderate to poor correlation.
- ML calibration significantly improved performance for CO, NO₂, and O₃ (r > 0.9).
- SO₂ calibration remained poor post-ML calibration due to ambient concentrations being below the sensor detection limit.

---

## 2.1.3 ML Model Performance Evaluation

![Figure 5: Heatmaps of Calibration Performance (R² and NRMSE)](fig5.png)

- Random Forest (RF) consistently achieved the highest R², followed by ANN and XGBoost.
- ML calibrations demonstrated lower NRMSE compared to laboratory calibrations.
- CO showed good natural performance; NO₂ and O₃ performance improved dramatically after cross-sensitivity corrections.
- Including NO₂ and O₃ cross-sensitivities as features significantly enhanced model predictions.
- SO₂ remained the most challenging pollutant to calibrate effectively.

---

## 2.1.4 Target Diagram Analysis

![Figure 6: Target Diagrams for Bias and Variance Assessment](fig6.png)

- RF, ANN, and XGBoost models achieved lower normalized RMSE (nRMSE) compared to LR and SVR.
- LAB calibrations for NO₂, O₃, and SO₂ exhibited large positive bias; ML models corrected this bias.
- ML models achieved near-zero bias for CO, NO₂, and O₃.
- LAB calibrated data had higher variance compared to ML calibrated data.
- All ML models' points fell within unit circles, indicating no overfitting.

---

## 2.1.5 EU Directive Compliance Evaluation

![Figure 7: Target Diagrams for EU DQO Compliance](fig7.png)

- EU DQOs require uncertainty below 25% for CO, NO₂, SO₂ and 30% for O₃.
- RF, ANN, and XGBoost models met DQOs for CO, NO₂, and O₃, but at concentrations well below the EU limit values.
- LAB calibrations had uncertainties exceeding 75% and were excluded from compliance analysis.
- LR and SVR models failed to meet DQO requirements for NO₂, O₃, and SO₂.
- SO₂ calibrations never met DQOs even after ML recalibration due to low ambient concentrations.

---

## 2.1.6 US EPA Data Quality Objectives (DQO) Compliance Assessment

![Figure 8: Spider Plots for Precision and Bias Error Analysis](fig8.png)

- EPA sets 10%, 25%, 30%, 50% error thresholds for regulatory and non-regulatory uses.
- LAB calibrations generally exhibited higher errors compared to ML calibrations.
- ML calibrations (RF, ANN, XGBoost) reduced precision and bias errors to below 25–30% for CO, NO₂, and O₃.
- CO LAB calibration alone allowed for hotspot and citizen science applications (30–50% error).
- SO₂ calibrations achieved moderate improvement post-ML (30–45% error).
- Neither LAB nor ML consistently met the 10% threshold required for regulatory applications.

---

## 2.1.7 Feature Sensitivity Analysis

![Figure 9: Variable Importance for LCS Calibration](fig9.png)

- Sensitivity analysis used Random Forest models to identify feature importance.
- CO calibration depended mainly on CO concentration; little influence from temperature or RH.
- NO₂ and O₃ calibrations were heavily influenced by temperature, RH, and cross-sensitivities.
- Including cross-sensitive gases improved NO₂ and O₃ calibrations by 6–9% in R².
- SO₂ model performance was affected equally by target gas, environmental, and temporal factors.

---

## 2.1.8 Effect of Training Data Fraction and Calibration Frequency

![Figure 11: NRMSE and REUmax as a Function of Training Data Fraction and Calibration Frequency](fig11.png)

- NRMSE and REUmax decreased as the fraction of training data increased.
- Monthly calibration (1-month frequency) achieved better performance than 3- or 6-month calibration.
- RF models performed better with frequent recalibration due to limited extrapolation ability.
- For monthly calibration, 70% of data was needed for CO and NO₂ and 50% for O₃ to meet EU DQOs.
- Interpretive sampling strategies significantly reduced data requirements (down to 22% for long-term calibration).

---

# 📋 Conclusion

- Random Forest models consistently achieved the best calibration performance for CO, NO₂, O₃, and SO₂ sensors.
- CO LCS was minimally influenced by temperature and RH, while NO₂ and O₃ were highly sensitive to environmental variables.
- Increasing temporal resolution to 2-min reduced training data requirements by ~6.3%.
- Monthly calibrations required about 50% of data, while 3- or 6-month calibrations needed around 60%.
- Interpretive sampling reduced data requirements to as low as 22%, cutting collocation costs significantly.
- These findings enable more cost-effective, scalable, and high-quality deployment of low-cost sensor networks.

---

# 📚 Project Structure

```bash
Air-Quality-in-Nicosia-Cyprus/
│
├── data/                # Raw and processed datasets
├── figures/             # Visualizations (fig3.png to fig11.png)
├── notebooks/           # Calibration Jupyter Notebooks
├── README.md            # Full project documentation
