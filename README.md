# Travel Insurance Prediction Analysis

## Project Overview

This project aims to predict which customers are likely to purchase a travel insurance package offered by a Tour & Travels Company, including Covid cover. Using a dataset from 2019 with details of almost 2000 customers, we perform data exploration, statistical inference, and apply machine learning models to make predictions.

## Setup Guide

To replicate the analysis locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/vytautas-bunevicius/travel-insurance-classifier.git
   ```
2. Navigate to the repository directory:
   ```
   cd travel-insurance-classifier
   ```
3. Install necessary Python libraries:
   ```
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

## Dataset Features

- Age: Customer's age
- Employment Type: Sector in which the customer works
- GraduateOrNot: Whether the customer graduated from college
- AnnualIncome: Customer's yearly income in Indian Rupees (rounded to nearest 50 thousand)
- FamilyMembers: Number of people in the customer's family
- ChronicDiseases: Whether the customer has a major disease like diabetes or high BP
- FrequentFlyer: Whether the customer booked air tickets at least 4 times in the last 2 years
- EverTravelledAbroad: Whether the customer has traveled to a foreign country
- TravelInsurance: Whether the customer bought the insurance in 2019

## Key Findings

1. Customer Characteristics:
   - Age range: 25-35 years, mean age around 30
   - Family size: 2-9 members, mean of 4.75
   - Annual income: 4,400 to 20,000 Euros

2. Significant Factors:
   - Employment Type: Significant association with insurance purchase
   - FrequentFlyer: More likely to buy insurance
   - EverTravelledAbroad: Strong correlation with insurance purchase
   - Annual Income: Higher income is a significant predictor

3. Predictive Modeling:
   - Random Forest: Best overall performance after hyperparameter tuning
   - Gradient Boosting: High performance in initial testing
   - Recall-Optimized Models: Achieved 100% recall but with increased false positives

## Recommendations

1. Implement the tuned Random Forest model for predicting potential customers.
2. Focus marketing efforts on key customer segments:
   - Government and private sector/self-employed individuals
   - Frequent flyers
   - Those with prior international travel experience
   - Higher income individuals
3. Tailor pricing and product strategies for different age groups and family sizes.
4. Continuously gather more customer data and retrain the model periodically.
5. Conduct a cost-benefit analysis to optimize the trade-off between customer acquisition and false positives.