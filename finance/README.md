# About  
This repository contains different exercises regarding quantitative finance analysis.

---
### Fund Technical Information
File: *fund_technical_info.ipynb*

Exercise to calculate common technical fund information such as TE, beta, correlation and volatility.

---
### Option Pricing
File: *call_option_pricing.ipynb*

Simple code to compute a call option price using the Monte-Carlo simulation in the Black-Scholes framework.

---
### Forecastinf using Monte-Carlo simulation
File: *forecasting_mc_sim.ipynb*  

This notebook shows an example of time series forecasting using Monte-Carlo simulation. In particular, we use the euler discretization of the geometric brownian motion (GBM).
We estimate the parameters (mean and sigma) on 1 year i.e. 12 data points. One could also use the Heston model. The main difference with the first example is that this time the volatility is also stochastic and thus is more realistic for simulating e.g. stock prices. However more tricky to setup since now we need the volatility of the volatility!  

![aapl_mc_sim_5months](https://user-images.githubusercontent.com/36447056/130253121-f0b3ac0e-b847-4f21-9e70-a7ea08779144.png)




