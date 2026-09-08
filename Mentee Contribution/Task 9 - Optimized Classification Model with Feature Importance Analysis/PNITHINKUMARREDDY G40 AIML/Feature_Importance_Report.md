# Feature Importance Report

**Model interpreted:** Logistic Regression (optimized, selected model). Values below are model coefficients
(direction matters: positive = increases predicted purchase likelihood, negative = decreases it), ranked by
absolute magnitude. Full list in `outputs/feature_importance.csv`.

| Rank | Feature | Coefficient | Direction | Interpretation | Recommended Action |
|---|---|---:|---|---|---|
| 1 | ProductRelated_Duration | 1.092 | ↑ increases likelihood | Longer time spent on product pages is the single strongest signal of buying intent | Trigger a cart-reminder or live-chat prompt once a session crosses a high product-page-dwell-time threshold |
| 2 | PageValues | 0.432 | ↑ increases likelihood | Sessions that reach higher-value pages (Google Analytics conversion-value metric) convert far more often | Prioritize these sessions in real-time remarketing; surface related high-value products |
| 3 | ExitRates | -0.201 | ↓ decreases likelihood | Pages with a high exit rate signal visitors leaving rather than continuing to shop | Audit and redesign the highest-exit-rate pages to reduce drop-off |
| 4 | OperatingSystems_2 | -0.141 | ↓ decreases likelihood | One specific OS code is associated with lower conversion | Investigate whether this reflects a device/rendering issue on that platform |
| 5 | Browser_3 | -0.126 | ↓ decreases likelihood | One specific browser is associated with lower conversion | Check checkout-flow compatibility on this browser |
| 6 | Month_June | -0.116 | ↓ decreases likelihood | June sessions convert at a lower rate than the baseline month | Consider seasonal promotions to lift June conversion specifically |
| 7 | BounceRates | -0.111 | ↓ decreases likelihood | Sessions that bounce (leave after one page) rarely convert | Improve landing-page relevance/load speed to reduce bounce |
| 8 | TrafficType_9 | 0.103 | ↑ increases likelihood | One specific traffic source converts especially well | Increase marketing investment in this traffic channel |
| 9 | Browser_13 | 0.100 | ↑ increases likelihood | Another browser associated with higher conversion | Lower priority — informational only |
| 10 | Browser_10 | 0.092 | ↑ increases likelihood | Another browser associated with higher conversion | Lower priority — informational only |

## Important caveat on causation
These are model-estimated associations, not proven causes. In particular, `PageValues` is itself a Google
Analytics metric partly computed from pages visited *close to* a completed purchase — so part of its
predictive power may reflect proximity to conversion rather than a cause of it. `OperatingSystems`/`Browser`
categories are anonymized codes; their effects should be validated against real usage data before acting on
them operationally (e.g. confirm a genuine checkout bug rather than assuming causation from the coefficient
alone).
