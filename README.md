# IAT 461 / 882 Final Project: FitnessPlanet Member Retention Strategy

**Derek Cheng** (`dca130`), Summer 2026

Client project for FitnessPlanet, a mid size gym chain that loses nearly half of every January
intake by the end of February. Leadership put three retention options on the table and wants to
know which one to fund.

## The two problems

**Problem 1.** Work out which of the three options the data actually supports, using class
attendance, visit frequency and distance from the gym.

- Option A: reward every new member for attending group fitness classes
- Option B: give every new member a punch card rewarding 3 visits a week
- Option C: check whether higher paying members leave more, and whether adjusting price would help

**Problem 2.** Assuming leadership proceeds with Option C, predict `churned_by_february` from the
monthly fee and other first month signals, then isolate how much the fee alone drives churn.

## Where this is

This is the **Checkpoint 2** draft, due Aug 9, with both problems modelled. The notebook runs from the
data documentation and EDA through the direction for each problem, then Problem 1 in section 4 and
Problem 2 in section 5.

## What the EDA found

- Churn is 45.12%, 22,560 of 50,000, which matches the "nearly half" in the client brief.
- The members who ghost were already showing up half as often in week 1, 1.85 visits against 3.59.
  Both groups then decay at the same rate, week 4 is 58% of week 1 for both. So the gap is there
  from day one and doesn't open up over the month.
- Only `avg_weekly_visits` really separates the two groups, with a KS of 0.421. Fee is 0.096 and
  distance is 0.091, which are shifts and not separations. That puts a ceiling on any model here.
- The raw fee gradient is about half contract mix. Month to month is both the most expensive plan
  and the least sticky, so churn by fee quintile mostly flattens once contract type is held constant.

## Direction

Both problems are supervised. Problem 1 is explanatory, since it asks which of three interventions
the data supports, so group comparisons, effect sizes and a permutation test rather than a fitted
model. Problem 2 is a logistic regression, picked over a tree ensemble because the client needs the
fee effect in dollars and not a feature ranking. Full reasoning in sections 3.1 and 3.2.

## Repository layout

```
IAT461_Final_DerekCheng_dca130.ipynb   the project notebook, runs top to bottom
app.py                                 Streamlit scenario tool for the client
requirements.txt                       dependencies
Report.pdf                             the progress report
data/gym_ghosts_master.csv             50,000 members x 24 columns
data/data_dictionary.csv               column dictionary shipped with the dataset
Milestone 1 - Client Proposal.pdf      the client brief
```

## Data

`gym_ghosts_master.csv`, from the Kaggle dataset linked in the client proposal
(`sergionefedov/gym-will-your-new-years-resolution-survive`). 50,000 member records, 24 columns,
zero missing values and no duplicates. All 50,000 rows are kept, see section 1.3 of the notebook
for why the distance and fee tails are treated as artifacts rather than errors.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
jupyter lab IAT461_Final_DerekCheng_dca130.ipynb
```

Runs top to bottom in about 90 seconds. `RANDOM_STATE = 42` throughout.

## The Streamlit app

```bash
streamlit run app.py
```

A scenario tool for the client, not a second analysis. The pricing tab moves the monthly fee and
shows predicted churn, members retained and year one revenue against the retained revenue curve.
The outreach tab moves the decision threshold and shows how many members get flagged and how many
leavers that catches. Pricing uses M3 from notebook section 5.5, flagging uses the model from 5.2,
so the numbers match the notebook.
