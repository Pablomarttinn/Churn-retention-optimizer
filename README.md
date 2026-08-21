# Retention Budget Optimizer

*Which customers should you call when you can't call everyone?*

This project solves a resource-allocation problem: given a **limited** retention budget, which customers should receive retention effort to **maximize the value retained** — not the number of churners caught.

The naïve approach ranks customers by churn probability and treats the highest-risk first. That wastes budget on customers who are likely to leave but cheap to lose. This project reframes the task as a **decision under constraint**: each customer has an expected value of being retained, each retention action has a cost, and the budget is finite. The result is a knapsack optimization over expected value, with a calibrated churn model feeding the probabilities and a business-grounded valuation turning them into euros.

**In one line:** on the IBM Telco dataset, reframing churn *prediction* as budget-constrained *decision* lets 7% of a mass-contact budget do the work of the whole thing — a targeted plan at 3.41× the efficiency of contacting everyone — while showing exactly when the optimizer helps and when it doesn't.

**[Live dashboard →](https://retention-budget-optimizer.streamlit.app/)**

## The problem

A churn model tells you *who is likely to leave*. But knowing who will leave is not the same as knowing *who to spend your retention budget on*. These are different questions, and the gap between them is where value is won or lost.

The standard approach ranks customers by churn probability and works down the list until the budget runs out. This has two blind spots:

- **It ignores value.** A customer who pays €20/month and one who pays €100/month count the same if their churn risk is equal — but saving the second is worth five times as much. Targeting by probability alone spends scarce budget on high-risk, low-value customers.
- **It ignores the cost of acting.** Reaching different customers can cost different amounts — a personal call, an email, an automated touch are not equally expensive, and not every customer is best reached the same way. When contact costs vary across customers, *"who to contact"* stops being a ranking and becomes a packing problem: the best set balances value against the cost of reaching it.

The right question is not *"who is most likely to churn?"* but *"which set of customers, contacted within my budget, retains the most value?"* That reframing — from **prediction** to **decision under constraint** — is what this project is built around.

## The approach

The project is built as a chain of three steps, each one addressing a question the previous step can't answer on its own.

**1. Prediction — who is likely to leave.**
A churn model estimates each customer's probability of leaving. The probability is only useful downstream if it's *calibrated*: when the model says 30%, roughly 30% of those customers should actually churn — otherwise multiplying it by euros produces meaningless numbers. This step ends with a calibrated probability per customer, not a yes/no label.

**2. Decision — who to spend the budget on.**
Each customer is assigned an *expected value of retention* (their probability and monthly value, turned into euros over a fixed horizon) and a *cost of contact*. Given a limited budget, choosing the set that maximizes total retained value minus cost is a **knapsack problem**. This is where prediction becomes decision: the model doesn't say "target this customer," the optimizer does — under an explicit budget constraint.

**3. Causality — would the action have worked.**
Retention value assumes the contact *succeeds*. But how persuadable a customer is (the campaign's effectiveness, or *uplift*) can't be read from historical data — it needs an experiment. The project treats this as an explicit, simulated dimension: you can vary effectiveness by segment and watch the optimal plan reorganize.

These three steps map onto the dashboard's views, and onto the three claims the project defends: **prioritizing by value beats prioritizing by risk; optimizing beats ranking when costs vary; and the "optimal" answer is sensitive to who is actually persuadable — an assumption the data can't provide.**

## How it works, part by part

### The churn model

The model is a gradient-boosted tree (**XGBoost**) trained on the IBM Telco dataset (7,043 customers, 26.5% churn rate). It's deliberately conservative — shallow trees (`max_depth=3`), slow learning rate (`lr=0.03`), row subsampling — which turns out to matter for what comes next.

**Performance** *(held-out test set, 80/20 stratified split)*: AUC 0.848 / AP 0.664. A logistic-regression baseline reaches AUC 0.846 / AP 0.657 — close enough that the choice of XGBoost isn't load-bearing; the project would stand on either model. XGBoost is preferred for capturing feature interactions without manual engineering, with aggregate ranking metrics marginally ahead. Cross-validated AUC on train (0.849) matches test AUC (0.848), so the hyperparameter search didn't overfit.

**Calibration — verified, not assumed.** Because probabilities get multiplied by euros downstream, they must be calibrated. Rather than applying a calibration wrapper by default, the project *checked whether one was needed*. On out-of-fold predictions, the raw model's Brier score (0.1338) was already as good as isotonic (0.1340) or sigmoid (0.1351) calibration, and the mean predicted probability matched the true churn rate to the sixth decimal (0.265368 vs 0.265370). More importantly, binned inside the band the campaign actually touches (p from 0.54 to 0.91), all five quantile bins fall within a 95% Wilson interval of the diagonal — calibration verified where it matters, not just in aggregate. The conservative training regime never developed the overconfidence that post-hoc calibration exists to fix. **No wrapper added, because measurement showed there was nothing to correct.**

### Turning probability into euros

A retained customer is worth money over time. The project values retention as a **fixed annuity**: the customer's monthly margin, summed over a fixed horizon of `H` months, discounted at a small monthly rate. The default horizon is 12 months (annuity factor A(12) = 11.2551 at a 1% monthly discount), with 6 and 24 months as sensitivity cases. The annuity assumes the customer, once retained, stays for `H` months — `H` is a business convention, not an estimate, which makes the valuation transparent and easy to defend.

The **gross expected value** of retaining a customer — what you recover if the action succeeds — is:

`expected_value = p · m · eff · mf · A(H)`

The **net value** subtracts the cost of contact:

`net_value = expected_value − cost`

where `p` is churn probability, `m` is monthly charge, `eff` is campaign effectiveness, `mf` is a margin factor (0.35, in line with telecom EBITDA margins), `A(H)` is the annuity factor, and `cost` is the contact cost. Effectiveness defaults to 0.30 — a working assumption, not measured here, and (as the dashboard shows) the single most consequential unverified parameter in the model. This gross/net split is what the optimizer works on: a customer can have positive gross value and negative net value — worth saving in principle, not worth the price of the phone call. Only `p · m` reorders customers; the rest is a valuation scalar that scales the euros without changing priorities, until costs or effectiveness vary per customer.

### The optimizer

With an expected value and a cost per customer, choosing the subset that maximizes total value within the budget is a **0/1 knapsack problem**, solved here with integer linear programming (PuLP + CBC).

The interesting part is *when the optimizer actually matters*. With **constant** cost and effectiveness, the knapsack has nothing to optimize: the choice reduces to taking the top-`k` by `p · m`, where `k` is what the budget buys. In this regime the optimizer and a simple ranking select the **same set** — verified: at a flat cost the knapsack and the top-N by `p · m` select identical sets, 500/500 at €10,000, and the same identity holds at every budget tested from €5,000 to €60,000, with net values matching to the cent. Even above the profitability elbow, where both stop spending before the budget is exhausted, they still agree. The knapsack isn't a fancier ranking; under homogeneity it *is* the ranking.

It reorders — and earns its place — only when cost or effectiveness are **heterogeneous**. Under channel costs, the optimizer retains €2,767 more than the ranking at a €10,000 budget (+23.9%), and it does so by picking customers that are individually *worse*: €37.82 of gross value per customer versus the ranking's €86.22. It wins by reaching 2.6× as many people with the same money. That trade — many cheap-but-decent over few expensive-but-excellent — is precisely what a ranking cannot see.

The dashboard lets you move between the regime where optimization collapses to ranking and the regime where it adds real value.

### The dashboard: three regimes, one optimizer

The [live dashboard](https://retention-budget-optimizer.streamlit.app/) is the same allocation engine seen under three cost/effectiveness regimes. Sliders (budget, contact cost, effectiveness, margin factor, horizon) update every view live.

**Flat cost** — the clean case. Every customer costs the same to contact, so the optimizer collapses to a ranking. This view isolates the first claim: prioritizing by **value** (`p · m`) beats prioritizing by **risk** (`p` alone). The advantage is largest when the budget is scarcest — exactly the regime a real campaign operates in.

**Cost by channel** — the realistic case. Contact cost varies by customer (a declared business policy mapped from contract type; the Telco dataset has no real contact-cost data). Now the optimizer *reorders*: it separates from the ranking and retains more for the same budget. This is where the knapsack earns its place.

**Simulated effectiveness** — the causal question. Campaign effectiveness can't be read from historical data, so it's an explicit, labeled **[SIMULATED]** assumption: three per-segment sliders let you set how persuadable each contract type is. This is the only control that *reorders the plan* rather than just scaling it — lower a segment's effectiveness and watch those customers drop out of the recommended list. It makes visible how much the "optimal" answer depends on an assumption nobody has measured.

## Key results

All figures below are under the **base-case assumptions**: effectiveness 0.30, margin factor 0.35, a 12-month horizon, and the declared channel-cost policy. They are illustrative of the mechanism, not measured outcomes — change any assumption in the dashboard and the euros move. Euro figures are net (expected revenue retained minus contact cost); ratios (ROI, efficiency) are gross returned per euro spent, so break-even sits at 1.0×, not 0.

**Prioritizing by value beats targeting by risk.** At a €10,000 budget with flat cost, allocating by `p · m` retains €2,120 more than targeting by churn probability alone — a 7.7% net gain over a baseline that is already sensible (it doesn't contact customers who cost more than they're worth). The gain is proportionally largest at small budgets (10.8% at €5k), where wasting scarce budget on low-value churners hurts most (the curve is noisy at this sample size, but the trend holds).

**Optimizing beats ranking under heterogeneous cost.** With channel-based costs, the knapsack retains €2,767 more than the value-ranking at €10,000 (+23.9% net) — by reaching 2.6× as many customers with the same money.

**Undirected spend destroys value — under this cost policy.** Contacting the entire base — no prioritization, no budget constraint — spends €182,180 to retain €163,780: it destroys €18,400 of value, net. Spending freely is worse than running no campaign at all.

This result is a direct consequence of the declared cost policy, not a universal law. Contacting the entire base returns €23.25 per customer on average, so the program is destructive whenever the average contact cost exceeds that. The channel policy averages €25.87 — dragged there by the 3,875 month-to-month customers on the €40 channel, 55% of the base — and destroys €18,400. Under a flat €20 the same mass program would be profitable (+€22,920). Note the distribution is irrelevant here: a flat €25.87 destroys exactly the same €18,400. When you contact everyone you pay everyone's cost, so only the mean matters. Heterogeneity matters elsewhere — it is what lets the optimizer *choose whom to pay for*, and it is why the knapsack beats the ranking above. The breakdown makes this concrete: contacted in bulk, two of the three segments lose money (month-to-month −€13,315, one year −€5,530) and the only profitable one clears just €445. Mass contact isn't dragged down by one expensive segment — it's underwater almost everywhere.

**What that buys, in context.** €10,000 is 7.1% of what contacting the entire base would cost (€140,860 at a flat rate). For that fraction of the spend, the targeted plan retains €29,685 in net value at 3.41× the efficiency of the mass program — and the crossover sits below €7,500, about 5% of what contacting everyone costs, past which a prioritized plan matches or beats undirected spend in net terms. Under the channel cost policy the comparison isn't close: the mass program destroys €18,400 while a €10,000 optimized plan retains €14,322. The point is not a marginal lift over a baseline — €10,000 retains €29,685 against €22,920 for contacting all 7,043 customers at twenty times the cost. A small, well-allocated budget comes out ahead of spending freely. That's a capital-allocation argument, not an incremental model improvement.

**The ROI, honestly.** At a margin factor of 0.35 (telecom-like), the value-ranking returns 3.97× per euro spent at €10,000. This decomposes as valuation scalar (1.18) × selection quality (3.36). The selection quality (3.36) — the part the model earns — is identical whether the margin factor is 0.35 or 1.0. Only the valuation assumption moves the ROI from 3.97× to 11.34×. The headline is 3.97×, not 11×.

## Limitations and findings

The project's value is as much in knowing where the model stops being trustworthy as in the results themselves. What follows are findings surfaced mostly by exploring the live dashboard.

**The optimizer adds nothing when the budget isn't binding.** Push a segment's assumptions far enough — month-to-month effectiveness to 0.10 (with one-year 0.30, two-year 0.30, €10,000, channel cost) — and its profitability threshold (`p · m > 101.5`) exceeds the most valuable customer in the entire dataset (`p · m = 91.9`): the segment becomes *mathematically impossible* to select, at any budget. What remains (897 profitable customers) fits within the budget, so the constraint goes inactive, the knapsack collapses to the ranking (+€0 advantage), and it stops spending at €7,755 of €10,000. This isn't a bug — it's the model correctly signaling that, absent scarcity, there is nothing to optimize. The dashboard shows this rather than hiding it.

The next three findings are three faces of one fact: the cheap two-year segment is the hinge of the whole channel-cost regime. Each looks at it from a different angle — what happens if they aren't persuadable, why the percentage metric misleads because of them, and why the ranking never sees them.

**The optimizer's advantage is fragile to an assumption nobody has measured.** Under uniform effectiveness (0.30), the knapsack beats the ranking by €2,767 at €10,000 — largely by exploiting cheap two-year customers. Simulate those customers as unpersuadable (two-year effectiveness 0.05, others held at 0.30) and the advantage collapses to €1,131: all 333 two-year customers drop out of the plan (643 → 340 selected). This confirms the base-case advantage came from their low contact cost, not from any measured persuadability. Effectiveness is the load-bearing assumption, and it isn't in the data.

**Percentage advantage is unreliable when the baseline collapses.** In simulation mode the relative advantage is shown in euros, not percent — deliberately. As the assumed effectiveness de-aligns `p · m` from net value, the ranking baseline can retain almost nothing, and dividing by a near-zero denominator makes the percentage explode: sweeping month-to-month effectiveness (with one-year 0.30, two-year 0.15), it swings 62% → 359% → 87% → 29% → 12% → 5% while the denominator moves by a factor of ten (€1,496 to €15,147). The euro figure is stable; the percentage is not. A general lesson: a percentage improvement over a baseline is treacherous when the baseline can approach zero.

**Ranking strategies don't just rank the cheap-but-valuable customers low — they don't see them at all.** Under channel costs, both ranking strategies hold *zero* two-year customers up to a €68,000 budget, while the optimizer holds 453 of them. The rankings only discover this block of 333 cheap (€3 contact), high-margin (~€101/month), low-risk (p ≈ 0.06) customers at €75,000 — as a visible upward step in the value curve — the point where budget finally stops being scarce. The optimizer selected them from a €10,000 budget onward. The rankings' blind spot isn't late prioritization; it's total omission until scarcity disappears.

**The entire causal dimension is simulated, not estimated.** Effectiveness — how persuadable each customer actually is — cannot be read from historical data, which shows who left, not who could have been retained. Every effectiveness figure in this project is a sensitivity assumption the user sets, never a measurement; activating simulation with all three sliders at 0.30 reproduces the real-mode result to €0.000000, confirming it is the same model under a chosen assumption, not a different one. Closing this gap would require a randomized experiment (an A/B test); the project is explicit that it does not, and treats the causal chapter as a discussion of what validation would look like, not a validated result.

## Tech & how to run

- **Stack:** Python 3.11, XGBoost, scikit-learn, PuLP (CBC solver), pandas, Streamlit, Plotly.
- **Model & analysis:** notebooks in `notebooks/` (foundation, calibration, optimization, results).
- **Core logic:** `src/retention_optimizer/` — valuation, allocator (knapsack), churn model, evaluation.
- **Dashboard:** `app/dashboard.py`.

**Install**

uv sync

**run the tests (16)**

uv run pytest

**run the dashboard**

uv run streamlit run app/dashboard.py


The deployed dashboard runs on a minimal dependency set (no modeling libraries at runtime) from a precomputed data file; `requirements.txt` holds the runtime dependencies, `requirements-dev.txt` the full environment.
