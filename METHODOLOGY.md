# Methodology & mathematical framework

## 1. Feature engineering

We rely on a small but dense set of features to capture team performance. Instead of using raw box-score averages (which are biased by opponent quality), we derive our own metrics.

### a. Elo-based efficiency ratings

We run three parallel Elo rating systems for every team. Unlike standard win/loss Elo, these track continuous performance metrics:

1. **Offensive rating:** points scored per 100 possessions.
2. **Defensive rating:** points allowed per 100 possessions.
3. **Pace rating:** possessions per 48 minutes.

**Math behind update rule**
For every game, we update a team's rating (R) based on the difference between their actual performance and expected performance.

```math
R_{new} = R_{old} + K \cdot (Actual - Expected)

```

* (k-factor): is set to `0.15`. This controls reactivity; a lower  makes the ratings more stable, while a higher  makes them react wildly to a single blowout.
* We use reversion to mean so with every update, we pull the rating `1%` back toward the league average. This prevents ratings from "drifting" too far into unrealistic territory over a long season.

**Calculating expectation:**
We adjust a team's offensive rating based on the opponent's defensive strength relative to the league average:

```math
Exp_{HomeOff} = Home_{OffRtg} + (Away_{DefRtg} - Avg_{LeagueDef})

```

* *Intuition:* if the home team has a rating of 115, but plays a defense that is 5 points better than average, we expect the home team to score only 110.

### b. Rolling stats 

While Elo captures "true quality," we also need to capture "current form" (hot/cold streaks). We use **5-game rolling averages** for:

* Points scored
* Pace
* Win percentage

*Note: all rolling stats are `shifted(1)` to ensure we only use data known before the game starts (preventing data leakage).*

### c. Rest days

We calculate the days elapsed since the last game for both teams, capped at 7. This serves as a proxy for fatigue (back-to-backs) or "rust" (long breaks).

## 2. Modeling strategy: the "two personalities" approach

To mitigate risk, we train two distinct XGBoost models with different "personalities" and hyperparameters in order 

### The conservative model 

* **Goal:** stability and minimizing variance.
* **Data:** trains on full history(2019–present). It assumes that long-term sample size is king.
* **Hyperparameters:**
* `learning_rate`: 0.02(slow, careful learning)
* `max_depth`: 4 (shallow trees to prevent overfitting)
* `min_child_weight`: 3 (requires meaningful sample sizes to make a split)



### The chaos model 

* **Goal:** capturing volatility and modern meta-shifts.
* **Data:** trains only on data since  1st January. It is an arbitrary date, however its main goal is to force the model to take into account higher paced era of basketball which happened in recent years.
* **Hyperparameters:**
* `learning_rate`: 0.05 (aggressive learning)
* `max_depth`: 8 (deep trees to model complex, specific interactions)
* `gamma`: 0 (no regularization penalties; it is allowed to chase outliers)



## 3. Prediction & betting logic

The system outputs a predicted score for home and away from both models.

```math
Prediction_{Total} = Pred_{Home} + Pred_{Away}

```

**Signal generation:**

* If **conservative** says over 220
* and **chaos** says over 225
* and Vegas line is 215 ->
* **Result:** strong buy signal on the over.

If the models disagree (e.g., conservative says under, chaos says over), the signal is considered "no bet" due to high uncertainty.
