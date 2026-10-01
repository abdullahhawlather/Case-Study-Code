"""Leakage-free supplier delivery-delay model (order level).

Paper link: Section 4 and Section 10.2, Table 12 (Random Forest, Gradient Boosting,
XGBoost versus a prevalence baseline, a leaky model that uses lead time, and the
percentile-threshold sensitivity check).
Output: results/delay_model.json
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier
from common import load_orders, save_json

orders = load_orders()
orders["od"] = pd.to_datetime(orders["Order_Date"])
orders["dd"] = pd.to_datetime(orders["Delivery_Date"])
orders["lead_time"] = (orders["dd"] - orders["od"]).dt.days
valid = orders[orders["lead_time"].between(0, 60)].sort_values("od").reset_index(drop=True)
threshold = valid["lead_time"].quantile(.75)
valid["delay"] = (valid["lead_time"] > threshold).astype(int)
valid["month"] = valid["od"].dt.month

# Inputs available at order placement only (no lead time, no defect information).
X = (pd.get_dummies(valid[["Supplier", "Item_Category"]])
       .join(valid[["Quantity", "Unit_Price", "Negotiated_Price", "month"]]).astype(float))
n = len(valid)
train_end, val_end = int(n * .7), int(n * .85)            # chronological 70 / 15 / 15
y_train, y_test = valid["delay"][:train_end], valid["delay"][val_end:]

def evaluate(model, features, label=valid["delay"]):
    model.fit(features[:train_end], label[:train_end])
    p = model.predict_proba(features[val_end:])[:, 1]
    pred = (p >= label[:train_end].mean()).astype(int)
    t = label[val_end:]
    return {"AUC": float(roc_auc_score(t, p)), "Brier": float(brier_score_loss(t, p)),
            "Accuracy": float(accuracy_score(t, pred)), "Precision": float(precision_score(t, pred, zero_division=0)),
            "Recall": float(recall_score(t, pred)), "F1": float(f1_score(t, pred))}

models = {"Random Forest": RandomForestClassifier(300, max_depth=4, random_state=0),
          "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0),
          "XGBoost": XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.1, random_state=0, eval_metric="logloss")}
R = {"valid_orders": n, "excluded_orders": int(len(orders) - n), "delay_threshold_days": float(threshold),
     "delay_prevalence": float(valid["delay"].mean()), "test_orders": len(y_test),
     "test_prevalence": float(y_test.mean()), "models": {k: evaluate(m, X) for k, m in models.items()}}
R["baseline_brier"] = float(brier_score_loss(y_test, [y_train.mean()] * len(y_test)))
leaky = X.copy(); leaky["lead_time"] = valid["lead_time"].astype(float)
R["leaky_model_with_lead_time"] = evaluate(RandomForestClassifier(300, max_depth=4, random_state=0), leaky)
R["threshold_sensitivity"] = {}
for q in (.6, .75, .9):
    th = valid["lead_time"].quantile(q)
    label = (valid["lead_time"] > th).astype(int)
    rf = RandomForestClassifier(300, max_depth=4, random_state=0).fit(X[:train_end], label[:train_end])
    R["threshold_sensitivity"][str(q)] = {"threshold_days": float(th), "prevalence": float(label.mean()),
                                          "AUC": float(roc_auc_score(label[val_end:], rf.predict_proba(X[val_end:])[:, 1]))}
save_json(R, "delay_model.json")
print({k: round(v["AUC"], 3) for k, v in R["models"].items()}, "baseline Brier", round(R["baseline_brier"], 3))
