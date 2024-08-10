# recipe-popularity-prediction  - DataCamp Professional Data Scientist Practical Exam Project
![models_metrics](https://github.com/user-attachments/assets/d2b09dbb-fc31-4dfd-a13f-3d0185af18d9)

# Frame the Problem
#### **Business objective**
Predicting **which recipes will lead to high traffic** on the homepage to increase traffic and subscriptions, with a **target precision of 80%.** (Correctly predict high traffic recipes 80% of the time).

The **main motive** behind this project is to increase website traffic and subscriptions by accurately predicting and displaying popular recipes on the homepage.

> **Traffic to the rest of the website goes up by as much as 40% if a popular recipe is chosen.** More traffic
means more subscriptions so this is really important to the company.


<br>

#### **What are the current solutions/workarounds (if any)?**
Currently, the responsible team member selects their favorite recipe from a selection and displays that on the homepage. This method has led to an approximate **40% increase in traffic** when a popular recipe is chosen. However, there is no systematic way to identify which recipes will be popular, resulting in inconsistent traffic increases.

<br>

#### **How should performance be measured?**
- The primary metric for our goal is **Precision**. This directly measures the proportion of correctly identified high traffic recipes among all recipes predicted as high traffic. Given the target precision of 80%, it aligns well with our requirement to ensure that a significant majority of the high traffic predictions are correct.

- Additionally, **Recall** is important. Recall will measure the proportion of actual high traffic recipes that are correctly identified by the model.

- **Accuracy** for high traffic predictions: While less relevant for the specific goal, overall accuracy will still be monitored to ensure that the model is performing well in general.

<br>

#### **What would be the minimum performance needed to reach the business objective?**
The current manual selection process has shown that choosing a popular recipe can increase traffic by 40%. To automate the process effectively and ensure consistent high traffic, the model needs to correctly predict high traffic recipes with **at least 80% precision**. While the target is 80%, demonstrating a significant improvement over the current inconsistent selection method, even if slightly below 80%, would still be valuable.

<br>

#### **How should you frame this problem (supervised/unsupervised, online/offline, etc.)?**
This is a **supervised learning** problem because we have historical data with labeled outcomes indicating whether a recipe led to high traffic or not. The model will be trained on this labeled data to predict future instances. The solution will be developed in an **offline setting**, where the model is trained and evaluated before being deployed. Regular updates with new data will be performed periodically to ensure the model remains accurate and relevant.

<br>

#### **How will your solution be used?**
The solution will be integrated into the website’s backend system to suggest recipes for daily display on the homepage. The model will analyze available recipes each day and predict which ones are likely to lead to high traffic. The chosen recipes will then be displayed on the homepage to maximize traffic and subscriptions. The system will be monitored and updated regularly to maintain performance and adapt to changing user preferences.

# Data Information
![image](https://github.com/user-attachments/assets/fa7e88ae-a0a2-4070-a7b1-ca815e40774a)
