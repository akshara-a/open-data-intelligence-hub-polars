# Customer Segment Report

**Method:** K-Means, k=4 (selected by silhouette score, 0.354 — see `Decision_Log.md`)
**Total customers:** 2,000

---

## Segment Summary

| Segment | Name | Customers | % of Base | Revenue Share | Avg. Recency (days) | Avg. Frequency | Avg. Spending | Avg. Rating |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cluster 3 | **High-Value Loyal Customers** | 443 | 22.2% | **59.9%** | 12.7 | 12.70 | $2,414.63 | 3.65 |
| Cluster 1 | **Growing / Promising Customers** | 662 | 33.1% | 20.0% | 26.3 | 2.74 | $538.77 | 3.55 |
| Cluster 2 | **Discount-Driven Customers** | 408 | 20.4% | 16.0% | 50.5 | 5.59 | $701.89 | 3.18 |
| Cluster 0 | **At-Risk / Low-Engagement Customers** | 487 | 24.4% | 4.0% | 173.9 | 0.00 | $148.22 | 2.75 |

*(Revenue share = each cluster's total spending ÷ total spending across all customers.)*

---

## Segment 1 (Cluster 3): High-Value Loyal Customers

**Characteristics:** Most recent purchases (~13 days), by far the highest purchase frequency (~12.7/year) and
spending (~$2,415/year), highest average order value ($238), low reliance on discounts (16% usage), and the
highest satisfaction rating (3.65). Only 22% of the customer base but **generates 60% of total revenue.**

**Business actions:**
- Offer loyalty rewards and a premium/VIP membership tier.
- Give early access to new products and limited releases.
- Avoid blanket discounting — this group already converts without it; discounting erodes margin unnecessarily.
- Prioritize retention here above all other segments — losing even a few of these customers has outsized
  revenue impact.

## Segment 2 (Cluster 1): Growing / Promising Customers

**Characteristics:** Good recency (~26 days), moderate frequency (~2.7/year), moderate spending (~$539/year),
highest website visit count (~29 visits) relative to purchases — engaged but not yet converting at the same
rate as loyal customers. Second-largest revenue contributor (20%).

**Business actions:**
- Send personalized product recommendations to convert browsing into purchases.
- Use onboarding/second-purchase incentives to build purchase habit.
- Nurture toward the loyal segment with milestone-based loyalty program entry points.
- Track this segment's migration rate into Cluster 3 as a leading indicator of program success.

## Segment 3 (Cluster 2): Discount-Driven Customers

**Characteristics:** By far the highest discount usage (73% of purchases), moderate recency (~50 days) and
frequency (~5.6/year), but the lowest average order value ($159) and below-average rating (3.18) — this group
buys mainly when there's a promotion.

**Business actions:**
- Run targeted, time-limited promotional campaigns rather than always-on discounts.
- Recommend bundled products to lift average order value without deepening the discount.
- Avoid extending discounts outside campaign windows — this segment already responds to promotions, so
  further discounting mostly sacrifices margin rather than driving incremental volume.
- Investigate the below-average rating — it may signal price-sensitivity is masking a satisfaction issue
  worth addressing separately.

## Segment 4 (Cluster 0): At-Risk / Low-Engagement Customers

**Characteristics:** Longest recency gap by far (~174 days since last purchase), essentially zero purchase
frequency, lowest spending ($148/year) and lowest satisfaction rating (2.75). Smallest revenue contribution
(4%) despite being the second-largest group by customer count (24%).

**Business actions:**
- Run low-cost, automated re-engagement email campaigns rather than expensive advertising.
- Send a "we miss you" comeback incentive to test win-back potential before investing further.
- Request feedback to understand the low rating — this may separate genuinely churned customers from
  recoverable ones.
- Given the low revenue share, treat this as a lower marketing-spend priority relative to the other three
  segments — resources are better spent nurturing Growing/Promising customers.

---

## Cross-Segment Takeaway

Revenue is heavily concentrated: the top 22% of customers (Cluster 3) generate ~60% of revenue, while the
bottom 24% (Cluster 0) generate only 4%. Marketing and retention budget should be weighted accordingly —
protect Cluster 3, invest in converting Cluster 1, manage Cluster 2's margin carefully, and treat Cluster 0
as a low-cost, low-priority win-back effort rather than a growth target.
