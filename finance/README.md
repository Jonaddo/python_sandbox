# About
This repository contains different exercises of quantitative finance analysis.

## Technical Fund Information
File: *portfolio_measures.ipynb*

Exercise to calculate common technical fund information such as TE, beta, correlation and volatility.\
Note: In the next update we will see how we can add the weight re-balancing in the logic.

Important reminder:
  1) Simple returns are additive in a portfolio but not log returns
  2) Simple returns are not additive over time
  3) Stock returns do NOT follow a normal distribution since in comparison they have fatter tails and are negatively skewed.
 
In the picture below, rp is the portfolio return in 2020 and ret_SP500 (S&P 500) is our benchmark.\
The excess return for year 2020 in this example is +7.19%.\
The constitutens of the portfolio are the SPY, TLT and GLD.  

![cumulative](https://user-images.githubusercontent.com/36447056/107700637-1199b380-6cb8-11eb-8a79-2804f520cfe0.png)


## Option Pricing
File: *call_option_pricing.ipynb*

Simple code to compute a call option price using the Monte-Carlo simulation in the Black-Scholes framework.


## Monte-Carlo Simulation
File: *MC_simulation_stock_price.ipynb*  

The notebook shows an example of time series forecasting using Monte-Carlo simulation. In particular, we use the euler discretization of the geometric brownian motion (GBM).
We estimate the parameters (mean and sigma) on 1 year i.e. 12 data points. One could also use the Heston model. The main difference with the first example is that this time the volatility is also stochastic and thus is more realistic for simulating e.g. stock prices. However more tricky to setup since now we need the volatility of the volatility!  

![aapl_mc_sim_5months](https://user-images.githubusercontent.com/36447056/130253121-f0b3ac0e-b847-4f21-9e70-a7ea08779144.png)




