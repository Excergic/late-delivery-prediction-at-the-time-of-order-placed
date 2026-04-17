# Problem Statement: Supply Chain Late Delivery Prediction

## Business Context
The company has no existing data-driven process for managing delivery risk.
Operations teams react to late deliveries after the fact. This model enables
proactive intervention — flagging at-risk shipments at order time so the ops
team can give them priority handling, reducing customer-facing late deliveries.

## ML Formulation
- **Problem type**: Binary classification
- **Target variable**: `Late_delivery_risk` (1 = late, 0 = on time)
- **Primary metric**: Recall — because a missed late shipment (false negative)
  causes customer impact; an unnecessary intervention (false positive) only
  costs ops time
- **Guardrail metrics**: Precision-Recall AUC, Precision (to keep alert volume
  manageable), F1
- **Current baseline**: No process exists — first model IS the baseline.
  Comparison point: naive "predict all late" achieves 100% recall, 54.8%
  precision — our model must be more precise while maintaining high recall

## Data Summary
- **Rows**: 180,519 shipments
- **Features**: 53 columns → ~35 usable after dropping leakage/PII/empty columns
- **Label availability**: Yes, fully labeled (`Late_delivery_risk`)
- **Class balance**: 54.8% late, 45.2% on time — manageable, no severe imbalance
- **Known leakage columns to drop**: `Delivery Status`, `Days for shipping (real)`
- **PII to drop**: `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`
- **High-null columns to drop**: `Product Description` (100% null), `Order Zipcode` (86% null)
- **Encoding**: latin-1 (non-UTF-8 characters present in source CSV)

## Constraints
- **Latency**: Batch — predictions needed at order creation time, not real-time API
- **Interpretability**: Moderate — ops team should understand why a shipment is flagged
- **Regulatory**: No specific compliance requirements noted

## Framework
- **Orchestration**: ZenML
- **Experiment tracking**: MLflow (via ZenML integration)

## Success Criteria
The model is production-ready when:
1. Recall ≥ 0.80 on held-out test set (catching 4 in 5 late shipments)
2. Precision ≥ 0.65 (flagged shipments are mostly actually at risk)
3. Pipeline is reproducible end-to-end via ZenML
4. Drift detection is operational
5. Ops team can consume predictions from a batch output file or dashboard
