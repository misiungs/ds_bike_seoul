# Bike demand in Seoul
Project about prediction of shared bikes in Seoul

<img src="https://github.com/misiungs/readme_images/blob/master/bike.jpg?raw=true" alt="drawing" width="500"/>

# Problem

The project consists of one year data (from 1.12.2017 to 30.11.2018) of count of rented bikes combined with weather information and time.
The main idea is to forecast the bike usage based on the past information.

This project is based on a dataset:
https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand
for which two articles can be identified:  
[1] Sathishkumar V E, Jangwoo Park, and Yongyun Cho. 'Using data mining techniques for bike sharing demand prediction in metropolitan city.' Computer Communications, Vol.153, pp.353-366, March, 2020  
[2] Sathishkumar V E and Yongyun Cho. 'A rule-based model for Seoul Bike sharing demand prediction using weather data' European Journal of Remote Sensing, pp. 1-18, Feb, 2020.  

Features:  
Date : year-month-day  
Rented Bike count - Count of bikes rented at each hour  
Hour - Hour of he day  
Temperature-Temperature in Celsius  
Humidity - %  
Windspeed - m/s  
Visibility - 10m  
Dew point temperature - Celsius  
Solar radiation - MJ/m2  
Rainfall - mm  
Snowfall - cm  
Seasons - Winter, Spring, Summer, Autumn  
Holiday - Holiday/No holiday  
Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)  

# Approach

After a thorough data exploration part of the dataset is chosen for time series regression.
First, the Holt-Winter method which does not take exogenous variables into account is used.
Next, different regression techniques with time series lags as additional variables are done and results are compared to the Holt-Winters approach.

# Conclusions
Results obtained for regression with lags yielded more accurate results but it has to be stated that variables such as temperature were assumed to be known.
In reality they would have to be predicted what would definetly affect accuracy of the model.









