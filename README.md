ASTraM – Adaptive Smart Traffic & Resource Management

INTRODUCTION

ASTraM is a traffic management system developed to help authorities manage congestion caused by planned and unplanned events such as processions, protests, public gatherings, construction activities, vehicle breakdowns, and water logging.

The system uses historical traffic event data and machine learning models to estimate the likelihood of road closure and recommend suitable traffic management actions.

OBJECTIVE

The objective of ASTraM is to support traffic authorities by:

• Predicting road closure requirements
• Recommending manpower deployment
• Suggesting barricade allocation
• Providing corridor-level risk information
• Assisting in diversion planning

DATASET

The solution was developed using the event dataset provided in the hackathon.

Dataset Information:

• 8,173 traffic events
• 46 columns/features
• Historical event records from multiple traffic corridors

Important Fields Used:

• Event Cause
• Event Type
• Corridor
• Zone
• Vehicle Type
• Priority
• Latitude
• Longitude
• Road Closure Requirement

WORKING OF THE SYSTEM

Step 1: Event Input

The user enters details about a traffic event:

• Event Cause
• Corridor
• Zone
• Vehicle Type
• Priority
• Event Type
• Location

Step 2: Prediction

The machine learning model processes the event details and predicts:

• Closure Probability

Step 3: Recommendation Generation

Based on:

• Closure Probability
• Event Priority
• Corridor Risk Profile
• Historical Incident Patterns

The system recommends:

• Required Manpower
• Barricade Sections
• Tow Trucks (if required)

Step 4: Corridor Intelligence

The system also displays:

• Historical Closure Rate
• Past Incident Count
• Most Common Incident Cause
• Peak Hour Incident Percentage

Step 5: Diversion Guidance

Based on the corridor profile and predicted impact, the system provides diversion recommendations.

FEATURES

• Road Closure Prediction
• Resource Recommendation
• Corridor Risk Profiling
• Diversion Planning
• Corridor Heatmap Visualization
• Model Performance Dashboard
• Event Outcome Logging

TECHNOLOGY STACK

Frontend:
• Streamlit

Backend:
• Python

Libraries:
• Pandas
• NumPy
• Scikit-Learn
• XGBoost
• Matplotlib
• Folium

EXAMPLE OUTPUT

Sample Event:

Cause: Procession
Corridor: Varthur Road
Priority: High

System Output:

Closure Probability: 76.3%
Manpower Required: 14
Barricade Sections: 18
Tow Trucks: 0

FUTURE IMPROVEMENTS

• Real-time traffic feeds
• CCTV integration
• Dynamic route recommendations
• Mobile application for field officers
• Smart city platform integration

LIVE DEPLOYMENT

ASTraM Dashboard:
https://astram-adaptive-smart-traffic-and-resource-management.streamlit.app

TEAM: Topic-Modellers

Project Name:
ASTraM – Adaptive Smart Traffic & Resource Management

Developed as part of the Flipkart GridLock 2.O Challenge.

TAGLINE

Transforming traffic event data into actionable recommendations for smarter and faster traffic management.
