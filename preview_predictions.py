import pandas as pd

# Load predictions
preds = pd.read_csv("outputs/predictions.csv")

# Flag high-risk customers (predicted_class = 0)
high_risk_class = preds[preds['predicted_class'] == 0]

print(f"Number of high-risk customers: {len(high_risk_class)}")
print(high_risk_class.head(10))

# Optional: flag based on probability threshold (e.g., < 0.3 for bad)
if 'predicted_proba' in preds.columns:
    threshold = 0.3
    high_risk_prob = preds[preds['predicted_proba'] < threshold]
    print(f"\nNumber of high-risk customers (prob < {threshold}): {len(high_risk_prob)}")
    print(high_risk_prob.head(10))

# Save flagged customers
high_risk_class.to_csv("outputs/high_risk_customers.csv", index=False)
print("\nSaved high-risk customers to outputs/high_risk_customers.csv")
