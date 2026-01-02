# NBA-over-under-model

## Overview
The main goal of this model is to predict the **Total Over/Under** points for NBA games alongside the **Point Spread**. 

The model combines two statistical approaches:
* **Rolling Averages:** To capture short-term streaks and immediate form.
* **Elo-Based Ratings:** To capture accurate, long-term team strength and efficiency, adjusted for the quality of opponents.

## How It Works
The system runs two distinct XGBoost models with different "personalities" to analyze the game:

*  The Conservative Model: Uses a lower learning rate and deeper historical data. Its goal is to capture long-term trends and provide stability.
*  The Chaos Model: Uses a high learning rate and only looks at data from the "modern scoring era" (post-Jan 2024). Its goal is to react aggressively to recent volatility and scoring explosions.

## Betting Logic
The strongest signals occur when **both models agree** on the outcome (e.g., both predict the Over, or both predict the Home team covers).

## Important Note
This model relies purely on statistical trends and does not take daily roster changes (injuries/resting) into account directly. Always double-check the injury report—especially if the Vegas line looks "too good to be true."

## Documentation
For a deep dive into the math behind the Elo calculations and model hyperparameters, please see [METHODOLOGY.md](METHODOLOGY.md).
