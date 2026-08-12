from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# config
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "gym_ghosts_master.csv"
RANDOM_STATE = 42

# Churners ghost by end of February so they bill ~2 months, stayers bill 12 (section 5.8)
MONTHS_IF_CHURN, MONTHS_IF_STAY = 2, 12

st.set_page_config(
    page_title="FitnessPlanet Retention Scenario Tool",
    layout="wide",
)

ACCENT, DARK, MUTED = "#e8925a", "#1e3a5f", "#7a9cc6"


@st.cache_data(show_spinner="Loading member data…")
def load_data():
    """Load the 50,000 members and rebuild the derived features from sections 2.3 and 5.1."""
    if not DATA_PATH.exists():
        st.error(
            f"Missing {DATA_PATH.name}. It ships with the repo in data/; if it is gone, "
            "re-download it from the Kaggle link in the client proposal."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)

    # reference: Week 5 slides, "Ratios & Interaction Features" (the four weekly counts
    # collapse into one rate; class_share is a ratio guarded against divide by zero)
    week_cols = [f"week{i}_visits" for i in range(1, 5)]
    df["total_visits_m1"] = df[week_cols].sum(axis=1)
    df["avg_weekly_visits"] = df["total_visits_m1"] / 4
    df["visit_trend"] = df["week4_visits"] - df["week1_visits"]
    df["class_share"] = np.where(
        df["total_visits_m1"] > 0,
        (df["avg_weekly_classes"] * 4) / df["total_visits_m1"],
        0.0,
    )
    return df


@st.cache_resource(show_spinner="Fitting the pricing model…")
def fit_pricing_model(df: pd.DataFrame):
    """M3 from section 5.5: fee, contract type, and the signup time controls.

    Contract type has to be in here. Without it the fee coefficient carries the
    commitment effect too and the answer comes out roughly twice as large.
    """
    # reference: Week 6 slides, "Why coefficients change when you add predictors"
    d = pd.get_dummies(df, columns=["contract_type", "primary_goal", "sex"], drop_first=True)
    cols = [
        "monthly_fee_usd", "contract_type_annual", "contract_type_month_to_month",
        "distance_km", "age", "prior_gym_experience", "joined_with_friend",
        "goal_aggressiveness", "signup_week_january", "promo_used",
    ]
    X = sm.add_constant(d[cols].astype(float))
    model = sm.Logit(d["churned_by_february"], X).fit(disp=0)
    return model, X


@st.cache_resource(show_spinner="Fitting the risk model…")
def fit_flagging_model(df: pd.DataFrame):
    """The predictive model from section 5.2, scored on the same held out 20%."""
    # reference: Week 7 slides, "The logistic function"; Week 4 "The Train / Test Split"
    numeric = [
        "monthly_fee_usd", "distance_km", "age", "goal_aggressiveness", "signup_week_january",
        "promo_used", "prior_gym_experience", "joined_with_friend",
        "avg_weekly_visits", "visit_trend", "avg_weekly_classes", "class_share",
        "avg_session_minutes", "guest_passes_used", "booked_induction", "personal_trainer",
        "app_installed", "locker_rented",
    ]
    cat = pd.get_dummies(df[["contract_type", "primary_goal", "sex"]], drop_first=True).astype(float)
    X = pd.concat([df[numeric], cat], axis=1)
    y = df["churned_by_february"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    ).fit(X_tr, y_tr)

    coefs = pd.Series(pipe[-1].coef_[0], index=X.columns).sort_values()
    return pipe, X_te, y_te, pipe.predict_proba(X_te)[:, 1], coefs


def scenario(model, X: pd.DataFrame, fees: pd.Series, cut: float):
    """Churn, retained revenue per member per month, and year one fee revenue at a fee change."""
    Xc = X.copy()
    Xc["monthly_fee_usd"] = Xc["monthly_fee_usd"] - cut
    p_churn = model.predict(Xc)
    fee_new = (fees - cut).clip(lower=0)
    retained_pm = (fee_new * (1 - p_churn)).mean()
    year_one = (fee_new * (p_churn * MONTHS_IF_CHURN + (1 - p_churn) * MONTHS_IF_STAY)).sum()
    return p_churn.mean(), retained_pm, year_one


def retained_revenue_curve(model, X: pd.DataFrame, lo: float, hi: float, step: float = 1.0):
    """Retained revenue if every member were put on the same flat fee, across the observed range."""
    grid = np.arange(np.ceil(lo), hi, step)
    out = []
    for f in grid:
        Xc = X.copy()
        Xc["monthly_fee_usd"] = f
        out.append(f * (1 - model.predict(Xc)).mean())
    return pd.DataFrame({"fee": grid, "retained": out})


def main():
    st.title("FitnessPlanet Retention Scenario Tool")
    st.markdown(
        """
**Problem 2: pricing and outreach.** 50,000 January members, **45.12%** of whom ghosted by the
end of February. Move the **fee** to see what it does to churn and revenue, or move the
**risk threshold** to size a retention campaign.

Every pricing figure is adjusted for contract type, because month to month is both the most
expensive plan and the least sticky. Built for IAT 461 Final Project · methods from Week 6
(coefficients and confounding) and Week 7 (logistic regression, odds, thresholds).
"""
    )

    df = load_data()
    m3, X3 = fit_pricing_model(df)
    _, X_te, y_te, proba, coefs = fit_flagging_model(df)
    fees = df["monthly_fee_usd"]

    # sidebar controls
    with st.sidebar:
        st.header("Controls")
        cut = st.slider(
            "Change to the monthly fee",
            min_value=-20.0, max_value=20.0, value=0.0, step=1.0,
            format="$%.0f",
            help="Applied to every member. Negative is a discount, positive is an increase.",
        )
        threshold = st.slider(
            "Flag a member above this churn risk",
            min_value=0.20, max_value=0.80, value=0.35, step=0.05,
            help="Lower catches more leavers but contacts more people who were staying anyway.",
        )
        st.divider()
        st.caption(
            f"Members: **{len(df):,}**  \n"
            f"Churn: **{df['churned_by_february'].mean()*100:.2f}%**  \n"
            f"Mean fee: **${fees.mean():.2f}**  \n"
            f"Fee range: **${fees.min():.2f} to ${fees.max():.2f}**  \n"
            f"Pricing model: **M3**, 10 predictors  \n"
            f"Risk model: **{X_te.shape[1]} features**, held out {len(y_te):,}"
        )
        st.divider()
        st.markdown(f"Data file:  \n`{DATA_PATH.name}`")

    base_churn, base_ret, base_year = scenario(m3, X3, fees, 0.0)
    churn, retained_pm, year_one = scenario(m3, X3, fees, cut)
    n = len(df)

    # metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean monthly fee", f"${(fees - cut).clip(lower=0).mean():.2f}",
              f"{-cut:+.0f}" if cut else None)
    c2.metric("Predicted churn", f"{churn*100:.2f}%",
              f"{(churn - base_churn)*100:+.2f} pts" if cut else None, delta_color="inverse")
    c3.metric("Members retained", f"{int(round((1 - churn) * n)):,}",
              f"{int(round((base_churn - churn) * n)):+,}" if cut else None)
    c4.metric("Year one fee revenue", f"${year_one/1e6:.2f}M",
              f"{(year_one - base_year)/1e6:+.2f}M" if cut else None)

    st.caption(
        f"Retained revenue is **${retained_pm:.2f}** per member per month "
        f"(fee times the probability they stay), against **${base_ret:.2f}** today."
    )

    # 5.7 pricing curve + 5.8 revenue impact
    left, right = st.columns(2)

    with left:
        st.subheader("5.7: Retained revenue across the fee range")
        curve = retained_revenue_curve(m3, X3, fees.min(), fees.max())
        fig = px.line(
            curve, x="fee", y="retained",
            labels={"fee": "Flat monthly fee ($)", "retained": "Retained revenue / member / month ($)"},
            title="No interior optimum: the curve rises the whole way",
        )
        fig.update_traces(line=dict(color=DARK, width=3), hovertemplate="$%{x:.0f} fee<br>$%{y:.2f} retained")
        fig.add_vline(
            x=float((fees - cut).clip(lower=0).mean()), line_dash="dash", line_color=ACCENT,
            annotation_text=f"your scenario  ${(fees - cut).clip(lower=0).mean():.0f}",
            annotation_position="top left",
        )
        fig.update_layout(height=440, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each $10 of fee costs about 1.9 points of stay probability and adds a full $10 to "
            "the bill, so the price response never bends the curve back down."
        )

    with right:
        st.subheader("5.8: What each size of cut costs")
        cuts = [0, 5, 10, 15, 20]
        deltas = [(scenario(m3, X3, fees, c)[2] - base_year) / 1e6 for c in cuts]
        bar = px.bar(
            x=[f"${c}" for c in cuts], y=deltas,
            labels={"x": "Monthly fee cut applied to every member", "y": "Change in year one revenue ($M)"},
            title="Every size of price cut loses money",
            text=[f"{d:+.2f}M" for d in deltas],
        )
        bar.update_traces(marker_color=[DARK] + [ACCENT] * 4, textposition="outside")
        bar.update_layout(height=440, showlegend=False, yaxis_range=[min(deltas) * 1.25, 1])
        st.plotly_chart(bar, width="stretch")
        st.caption(
            "A $10 cut needs churn to fall 15.6 points to break even. It delivers 1.9, "
            "so it is about 8.3 times short."
        )

    # 5.3 threshold
    st.subheader("5.3: Who to contact")
    pred = (proba >= threshold).astype(int)
    caught = int(((y_te == 1) & (pred == 1)).sum())
    missed = int(((y_te == 1) & (pred == 0)).sum())

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Members flagged", f"{pred.sum():,} of {len(y_te):,}", f"{pred.mean()*100:.1f}% of the base")
    f2.metric("Leavers caught", f"{caught:,}", f"{recall_score(y_te, pred)*100:.1f}% recall")
    f3.metric("Leavers missed", f"{missed:,}", delta_color="inverse")
    f4.metric("Flags that were right", f"{precision_score(y_te, pred)*100:.1f}%")

    sweep = pd.DataFrame([
        {
            "threshold": t,
            "Leavers caught": recall_score(y_te, (proba >= t).astype(int)),
            "Flags that were right": precision_score(y_te, (proba >= t).astype(int)),
        }
        for t in np.arange(0.20, 0.85, 0.05)
    ])
    tr = px.line(
        sweep, x="threshold", y=["Leavers caught", "Flags that were right"],
        labels={"threshold": "Risk threshold", "value": "Rate", "variable": ""},
        title="Catching more leavers means contacting more people who were staying",
        color_discrete_map={"Leavers caught": ACCENT, "Flags that were right": DARK},
    )
    tr.update_traces(mode="lines+markers")
    tr.add_vline(x=threshold, line_dash="dash", line_color="grey")
    tr.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(tr, width="stretch")

    # expanders
    with st.expander("Why the fee effect is smaller than it first looks (5.5)", expanded=False):
        st.markdown(
            """
Comparing fee against churn on its own says every $10 a month multiplies the odds of leaving by
**1.204**. Adding two variables for contract type drops that to **1.079**, so more than half of it
was never about price.

Month to month members pay the most (**$58.08**) and churn the most (**50.4%**). Annual members pay
the least (**$42.01**) and churn the least (**30.6%**). The simple comparison was reporting the
commitment effect and the price effect glued together.

Adding distance, age, prior experience, friend referral, goal aggressiveness, signup week and promo
leaves it at **1.084**, so contract type was the only thing distorting it. That is the number this
app uses.

*reference: Week 6 slides, "Why coefficients change when you add predictors"*
"""
        )

    with st.expander("Why the curve does not justify raising the price either", expanded=False):
        st.markdown(
            """
The retained revenue curve rises across the whole observed range, which looks like an argument for
charging more. It is not, and the reason is a limit of the data rather than the analysis.

**Every member in this file already agreed to the price.** Nobody in it saw the fee and walked away.
So the data can say how price relates to members *leaving*, but nothing about how price affects
whether people *join* in the first place. Acquisition is the thing that would actually bound a price
rise, and it is invisible here.

Testing an increase needs the randomised pilot in notebook section 5.11: next January's intake split
into a control arm, one arm about 10% above current pricing, and one about 10% below, measuring
**signup rate** first.
"""
        )

    with st.expander("What the risk model is reading", expanded=False):
        top = pd.concat([coefs.head(6), coefs.tail(6)]).sort_values()
        cf = px.bar(
            x=top.values, y=top.index.astype(str), orientation="h",
            labels={"x": "Coefficient on standardised features (log odds)", "y": ""},
            title="Strongest signals, both directions",
        )
        cf.update_traces(marker_color=[DARK if v < 0 else ACCENT for v in top.values])
        cf.update_layout(height=380, showlegend=False)
        st.plotly_chart(cf, width="stretch")
        st.markdown(
            """
The model is mostly reading behaviour. `avg_weekly_visits` and `avg_weekly_classes` are an order of
magnitude above everything else, then distance, contract type and goal aggressiveness. Fee is the
seventh largest effect, which is consistent with Problem 1.

**Where to set the threshold.** A missed leaver costs a membership, roughly $53.90 a month for as
long as they would have stayed. A false alarm costs one email. That asymmetry says go low, around
**0.35**, where F1 also peaks. If the intervention is expensive, say a free PT session, go to 0.55 or
higher and spend on a smaller, more certain group.

**The version that needs no model at all.** Members who visited once or less in week 1 are **33.3%**
of the intake, churn at **65.4%**, and contain **48.3%** of everyone who leaves. The front desk can
pull that list on the 8th of January.

*reference: Week 7 slides, "Precision and recall trade off", "A strict threshold", "A looser threshold"*
"""
        )

    st.divider()
    st.caption(
        "Pricing uses M3 from notebook section 5.5, flagging uses the model from section 5.2, so the "
        "numbers match the notebook exactly. Findings are correlation, not causation. No results are "
        "broken out by age or sex, per the client."
    )


if __name__ == "__main__":
    main()
