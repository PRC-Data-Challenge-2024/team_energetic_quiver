
```markdown
# PRC Data Challenge Submission

This repository contains the code and methodology developed for the **PRC Data Challenge**. The challenge invites participants to create machine learning models for estimating Actual TakeOff Weight (ATOW) using extensive flight and trajectory data. By improving ATOW estimations, this project aims to support environmentally responsible decision-making in aviation

For more details, please visit the [PRC Data Challenge website](https://ansperformance.eu/study/data-challenge/).

## Prerequisites
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - Additional libraries as required in each script

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yanyuwan95/PRC-data-challenge.git
   ```
2. Navigate into the project directory:
   ```bash
   cd PRC-data-challenge
   ```
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing Steps
To replicate the analysis, run each Python script in sequence as described below. Ensure each step completes successfully before proceeding to the next.

### Step 1: Smooth the Trajectory
**File:** `Step 1.py`  
This step smooths the raw trajectory data, reducing noise and outliers to obtain a cleaner flight path representation.

### Step 2: Extract Flight Information
**File:** `Step 2.py`  
Key flight information, including departure time, altitudes, and speed, is extracted to build a foundation for further analysis.

### Step 3: Identify Abnormal Flights
**File:** `Step 3.py`  
Abnormal flights, such as those with unexpected trajectory shapes, are identified to enhance the quality and reliability of the data set.

### Step 4: Update Flight Information Based on Average Strategy
**File:** `Step 4.py`  
Missing or inconsistent flight data is updated based on an average strategy, ensuring continuity and accuracy within the dataset.

### Step 5: Identify Standard Tow from Historical Data
**File:** `Step 5.py`  
Historical data is analyzed to define a standard tow baseline, which serves as a reference for predicting future tow behavior.

### Step 6: Predict Tow Using XGBoost
**File:** `Step 6.py`  
An XGBoost model is trained using the flight information and standard tow data to predict tow behavior accurately.

## Results
The results is our predicted tow values for final_submission_set (my_submission.csv).

## License
This project is licensed under the terms of the [GNU General Public License v3.0](./LICENSE).

---

For questions or further information, please contact [Yanyu Wang <yanyuwang@lsu.edu>, Emir Ganic <e.ganic@sf.bg.ac.rs>].
```
