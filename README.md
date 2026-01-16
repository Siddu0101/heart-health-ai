# Heart Health AI – Heart Disease Prediction System

## Project Overview
Heart Health AI is a machine learning–based web application designed to predict the risk of heart disease using patient clinical parameters. The system assists in early detection by classifying patients into **High Risk** or **Low Risk** categories, along with prediction confidence visualization.

The project is built using supervised machine learning techniques and deployed as a Flask-based web application.

---

## Objectives
- To analyze patient health parameters related to cardiovascular disease
- To build an accurate machine learning model for heart disease prediction
- To provide a simple and user-friendly web interface
- To visualize prediction confidence for better interpretability
- To align the implementation with published research studies

---

##  Dataset
- **Source:** UCI Machine Learning Repository – Heart Disease Dataset  
- **Attributes:** Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, ECG results, Maximum Heart Rate, Exercise-induced Angina, ST Depression, and other clinical parameters.

---

##  Technology Stack
- Programming Language:Python
-Machine Learning: Scikit-learn
- Web Framework: Flask
- Frontend:HTML, CSS
- Visualization: Matplotlib
- Model Serialization:Pickle
  

---

## Machine Learning Approach
- Supervised classification model trained on historical heart disease data
- Ensemble-based learning (Random Forest / Decision Tree–based model)
- Probability estimation using `predict_proba()`
- Input validation to ensure medical plausibility
- Confidence visualization using bar charts

---

## Application Workflow
1. User enters patient health details via web interface  
2. Input data is validated and preprocessed  
3. Trained ML model predicts heart disease risk  
4. Prediction result and confidence graph are displayed  

---

## Output
- **Prediction Result:**  
  - High Risk of Heart Disease  
  - Low Risk (Healthy)  
- **Confidence Visualization:**  
  - Probability distribution between Healthy and Disease classes  
## 📸 Application Screenshots

### Home Page
![Home Page](screenshots/home_page.png)

### Prediction Result
![Prediction Result](screenshots/prediction_result.png)

---

## Research Papers Referenced
This project was developed by referring to established and peer-reviewed research papers in the field of heart disease prediction using machine learning.

The referenced papers are included in the `papers/` directory for academic and learning purposes.

---


## Future Enhancements
- Integration of Explainable AI (XAI) techniques
- Deep learning–based prediction models
- Cloud deployment with real-time monitoring
- Mobile application integration
- Secure healthcare data integration

---

## License
This project is developed for academic and educational purposes.

---

##  Author
**Siddu Sidharth**  
Department of CSE (AI & ML)  
Vignana Bharathi Institute of Technology (VBIT)


