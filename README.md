# predictive_maintenance
Predictive maintenance model that uses engine sensor data to detect potential failures and enable proactive maintenance using machine learning.

## Business Context

Vehicle breakdowns and engine failures lead to significant financial losses for both individual owners and fleet operators. Unexpected engine failures can result in costly repairs, operational downtime, and safety risks. Predictive maintenance helps address these challenges by using sensor data to identify early signs of engine failure.

Automobile manufacturers, fleet managers, and service providers increasingly rely on data driven approaches to improve engine reliability and optimize maintenance schedules. By analyzing engine health parameters such as RPM, pressure, and temperature, machine learning models can help identify patterns that indicate potential engine faults.

The dataset used in this project contains sensor readings that reflect the operating conditions of engines used in vehicles and small machinery such as lawnmowers, generators, and compact equipment. These sensor measurements provide useful signals that can be analyzed to detect abnormal engine behavior.

---

## Objective

The objective of this project is to develop a machine learning based predictive maintenance model that analyzes engine sensor data to determine whether an engine is operating normally or may require maintenance.

The key goals of this analysis are to:

- Understand the distribution and relationships between different engine sensor parameters  
- Identify patterns associated with engine faults  
- Build and evaluate machine learning models that can classify engine condition  

---

## Project Approach

To achieve the objective, the following steps are performed as part of the interim project work:

### 1. Data Registration
- Create a project folder structure with a master folder and a **data** subfolder  
- Register the dataset on the **Hugging Face dataset hub** for centralized access  

### 2. Exploratory Data Analysis (EDA)
- Review dataset structure and background information  
- Perform **data overview** to understand variables and distributions  
- Conduct **univariate analysis** to examine the distribution of each variable  
- Conduct **bivariate analysis** to explore relationships between sensor variables and engine condition  
- Conduct **multivariate analysis** to identify combined relationships among multiple features  
- Summarize key insights obtained from the analysis  

### 3. Data Preparation
- Load the dataset directly from the **Hugging Face dataset space**  
- Perform necessary data cleaning and ensure consistent formatting of variables  
- Split the dataset into **training and testing datasets**  
- Save the processed datasets locally and upload them back to the **Hugging Face dataset space** for reuse in model training  

### 4. Model Building with Experimentation Tracking
- Load training and testing datasets from the **Hugging Face dataset space**  
- Train machine learning models such as Decision Tree, Random Forest, Gradient Boosting, or XGBoost  
- Tune model parameters and track experimentation results  
- Evaluate model performance using recall
- Register the best performing model in the **Hugging Face model hub**

---

## Data Description

| Feature | Description |
|-------|-------------|
| **Engine_RPM** | Number of revolutions per minute of the engine, indicating engine speed. |
| **Lub_Oil_Pressure** | Pressure of the lubricating oil that helps reduce friction and wear in engine components. |
| **Fuel_Pressure** | Pressure at which fuel is delivered to the engine for combustion. |
| **Coolant_Pressure** | Pressure of the engine coolant responsible for temperature regulation. |
| **Lub_Oil_Temperature** | Temperature of the lubricating oil affecting viscosity and lubrication efficiency. |
| **Coolant_Temperature** | Temperature of the coolant that prevents engine overheating. |
| **Engine_Condition** | Target variable representing engine health where **0 indicates normal operation** and **1 indicates a faulty engine condition**. |
