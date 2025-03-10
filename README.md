# Algorithmic Trading Competition Platform

[![Coverage Status](https://coveralls.io/repos/github/ACquantclub/algo-trading-platform/badge.svg?branch=main)](https://coveralls.io/github/ACquantclub/algo-trading-platform?branch=main) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Overview
This repository contains the platform and resources for the Amherst College Quant Competition: Algorithmic Trading Case, where participants develop portfolio allocation strategies across a diverse set of global stocks.

## Competition Structure
- **Objective**: Develop algorithms for optimal portfolio allocation
- **Evaluation**: Performance ranked on multiple financial metrics including Sharpe Ratio, Sortino Ratio, Max Drawdown, etc.
- **Final Scoring**: Weighted sum of ranks across all metrics
- **More Information**: See case packet folder for comprehensive details on the case (**Note:** The case packet contains the only information given to participants; other information in this repository were not known to participants until after the competition)

## Bot Baseline Submissions
The competition includes baseline submissions generated from leading LLMs:
- ChatGPT-4o
- Claude 3.5 Sonnet
- DeepSeek R1
- Gemini 2.0 Flash
- Perplexity Deep Research
- And more...

These bots received identical information as participants (case packet, training data, example code) and serve as performance benchmarks.

## Gradescope Integration
The `strategy_evaluator` folder contains our code for evaluating participants' strategies and includes our custom API for integration with Gradescope. Gradescope was chosen as the platform to handle participant submissions and evaluation because it is a familiar platform that almost all college students have experience with.

## Stock Details

### Geographic Distribution (26 Stocks Total)
- US Exchanges: 18 stocks
- Chinese Markets: 3 stocks
- Indian Markets: 2 stocks
- Japanese Markets: 2 stocks
- European Markets: 1 stock

### Market Capitalization Categories
- Mega-cap (>$200B): 11 stocks
- Large-cap ($10B-$200B): 13 stocks
- Mid-cap ($2B-$10B): 2 stocks
- *Note: Minimum $2B cap to reduce volatility*

### Industry Diversity
- 22 distinct industries represented
- Sectors include:
  - Commodities
  - Internet Retail
  - Utilities
  - And more...

### Currency Standardization
All stock prices are converted to USD using IRS annual mean exchange rates.

## License
This project is licensed under the Apache License 2.0
