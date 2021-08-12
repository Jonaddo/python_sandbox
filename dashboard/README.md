# Zürich population - Quick overview
Here we create a little dashboard thanks to the library called **dash** (installation instructions [here](https://dash.plotly.com/installation))    

## Input data
Our toy dataset have 34 attributes (gender, year, month, age, district,..) describing the population of Zürich, Switzerland (can be downloaded [her](https://data.stadt-zuerich.ch/dataset/bev_monat_bestand_quartier_geschl_ag_herkunft_od3250)). The time range starts from January 1998 to May 2021 (the dataset is updated monthly).  
To make it simple, we prepare two dataframes, one aggregation per gender overtime and another one for foreigners ratio per district overtime.

## The dashboard
The dashboard has three filters (*district* aka *Kreis*, *year* and *semester*). The filters *year* and *semester* update both graphs thanks to the callback dash decorators.  
The *Kreis* filter only updates the Bar chart on the left.

<img width="960" alt="dash_zurich_eda" src="https://user-images.githubusercontent.com/36447056/129057160-4e92a53c-ed31-4b91-a584-ebbbd53e1afc.png">

