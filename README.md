# Travel Insurance Prediction Analysis

## Setup Guide

To replicate the analysis locally, follow these setup steps:

- Clone the Repository:

      git clone https://github.com/vytautas-bunevicius/travel-insurance-prediction.git

- Navigate to the repository directory:

      cd travel-insurance-prediction

- Install necessary Python libraries using the command:

      pip install -r requirements.txt

- Launch Jupyter Notebook to interact with the analysis:

      jupyter notebook

## Project Overview

This project aims to predict which customers will buy a travel insurance package offered by a Tour & Travels Company, including Covid cover. The dataset from 2019 includes details about almost 2000 customers. The analysis involves data exploration, statistical inference, and applying machine learning models to make predictions.

## Research Objectives

1. **Customer Profiling:** Identify key characteristics of customers likely to purchase travel insurance.
2. **Predictive Modeling:** Develop models to predict insurance purchase based on customer data.
3. **Statistical Analysis:** Validate the relationships between customer attributes and insurance purchase decisions.
4. **Feature Importance:** Determine the most influential factors in predicting insurance purchase.

## Dataset Features

- **Age:** Customer’s age
- **Employment Type:** Sector in which the customer works
- **GraduateOrNot:** Whether the customer graduated from college
- **AnnualIncome:** Customer’s yearly income in Indian Rupees (rounded to the nearest 50 thousand)
- **FamilyMembers:** Number of people in the customer’s family
- **ChronicDiseases:** Whether the customer has a major disease like diabetes or high BP
- **FrequentFlyer:** Whether the customer booked air tickets at least 4 times in the last 2 years
- **EverTravelledAbroad:** Whether the customer has traveled to a foreign country
- **TravelInsurance:** Whether the customer bought the insurance in 2019

## Exploratory Data Analysis Questions

1. What are the distributions of key customer attributes in the dataset?
2. How do these attributes correlate with the purchase of travel insurance?
3. Are there identifiable patterns that can help predict insurance purchase?

## Findings and Insights

### 1. Customer Characteristics

- **Age:** Most customers are between 25 and 35 years old, with a mean age of around 30.
- **Family Members:** Family sizes range from 2 to 9 members, with a mean of 4.75.
- **Annual Income:** Annual incomes range from 4,400 to 20,000 Euros, with notable peaks at 5,000, 10,000, and 15,000 Euros.

### 2. Significant Factors

- **Employment Type:** Significant association with travel insurance purchase.
- **FrequentFlyer:** Frequent flyers are more likely to buy travel insurance.
- **EverTravelledAbroad:** Strong correlation with insurance purchase.
- **Annual Income:** Higher income is a significant predictor of insurance purchase.

### 3. Predictive Modeling

- **Gradient Boosting:** Achieved the highest accuracy at 84.42%, with a good balance between precision and recall.
- **Voting Classifier:** Combined multiple models to achieve robust performance with an accuracy of 83.92%.

## Recommendations

1. **Target Marketing:** Focus on high-income customers, frequent flyers, and those with international travel experience.
2. **Model Enhancements:** Further tune models and explore additional features to improve prediction accuracy.
3. **Future Data:** Incorporate more recent data to reflect post-Covid travel habits.

## Future Work

- **Advanced Models:** Experiment with neural networks and other advanced machine learning techniques.
- **Feature Engineering:** Create new features to better capture customer behavior and improve model performance.
