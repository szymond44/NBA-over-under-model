from scipy.stats import norm

def quick_ev(prediction, line, odds, rmse):
    prob =  norm.cdf((line - prediction) / rmse) #remove 1 for under
    ev = prob * (odds - 1) - (1 - prob)
    return prob, ev

prediction = 235
rmse = 18.93

# Check each line
lines = [
    (237, 2.23),
    (238, 2.12),
    (239, 2.02),
    (240, 1.93),
    (241, 1.85),
    (242, 1.77),
    (243, 1.7),
    (244, 1.64),
]

for line, odds in lines:
    prob, ev = quick_ev(prediction, line, odds, rmse)
    print(f"U{line} @ {odds}: {prob:.1%} prob, EV = {ev:+.2f} ({ev*100:+.1f}%)")
