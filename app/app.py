"""Credit portfolio segmentation dashboard.

Run from the project root:   streamlit run app/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # lets us write "from src import ..." when Streamlit starts this file

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import appdata, stats
from src.config import ACCENT, FIGURE_DIR, SEGMENT_COLOURS, WARNING
from src.segments import SegmentModel, borrower_frame, category_options

st.set_page_config(page_title="Credit Portfolio Segmentation", layout="wide")


# ---------- Cached loaders (each runs once, then the result is reused) ----------
@st.cache_data
def load_table(name, **options):
    """Read one of the small files in app_data (None if it does not exist yet)."""
    return appdata.read_table(name, **options)


@st.cache_resource
def load_model():
    return SegmentModel()


def style(figure, height=380):
    figure.update_layout(
        height=height, template="plotly_white", margin={"l": 10, "r": 10, "t": 40, "b": 10}
    )
    return figure


def show_image(file_name, caption):
    path = FIGURE_DIR / file_name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{file_name} not found. Run the notebook that creates it.")


# ---------- Load the data ----------
profile = load_table("segment_profile.csv", index_col="Segment")
sample = load_table("loans_sample.csv")
if profile is None or sample is None:
    st.error("Dashboard data not found. Run notebook 10 (dashboard data) first.")
    st.stop()

names = profile["name"].to_dict()
label_of = {segment: f"{segment}: {name}" for segment, name in names.items()}
colour_of = {
    label_of[s]: SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)] for i, s in enumerate(sorted(names))
}
sample["SegmentName"] = sample["Segment"].map(label_of)
profile_view = profile.rename(index=label_of)

total_loans = int(profile["loans"].sum())
portfolio_rate = profile["defaults"].sum() / total_loans

st.title("Credit Portfolio Segmentation and Risk Patterns")
st.caption(
    "Segments are built from borrower characteristics only. Default is used afterwards, as an outside test."
)
st.info(
    "Segments describe groups of borrowers in a historical portfolio. They are not a credit decision "
    "about any individual, and default rates describe groups, not causes."
)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
st.sidebar.caption(
    "Filters apply to charts based on the 100,000-loan sample (Overview and the explorer)."
)
segment_choice = st.sidebar.multiselect(
    "Segment", sorted(names), default=sorted(names), format_func=lambda s: label_of[s]
)
year_low, year_high = int(sample["IssueYear"].min()), int(sample["IssueYear"].max())
years = st.sidebar.slider("Issue year", year_low, year_high, (year_low, year_high))
purpose_choice = st.sidebar.multiselect(
    "Loan purpose (empty = all)", sorted(sample["Purpose"].dropna().unique())
)
home_choice = st.sidebar.multiselect(
    "Home ownership (empty = all)", sorted(sample["HomeOwnership"].dropna().unique())
)
state_choice = st.sidebar.multiselect(
    "State (empty = all)", sorted(sample["State"].dropna().unique())
)

filtered = appdata.filter_sample(
    sample, segment_choice, years, purpose_choice, home_choice, state_choice
)
st.sidebar.write(f"{len(filtered):,} of {len(sample):,} sampled loans selected")

overview_tab, segments_tab, default_tab, maps_tab, borrower_tab, method_tab = st.tabs(
    [
        "Overview",
        "Segments",
        "Default Patterns",
        "Maps and Anomalies",
        "Place a Borrower",
        "Method and Limits",
    ]
)

# ---------- 1. Overview ----------
with overview_tab:
    if filtered.empty:
        st.warning("No sampled loans match the current filters.")
    else:
        cards = [
            ("Loans in view (sample)", f"{len(filtered):,}"),
            ("Full portfolio", f"{total_loans:,}"),
            ("Default rate in view", f"{filtered['Default'].mean():.1%}"),
            ("Median income", f"{filtered['Income'].median():,.0f}"),
            ("Median loan", f"{filtered['LoanAmount'].median():,.0f}"),
            ("Median credit score", f"{filtered['CreditScore'].median():.0f}"),
        ]
        for column, (label, value) in zip(st.columns(len(cards)), cards):
            with column.container(border=True):
                st.caption(label)
                st.subheader(value)

        left, right = st.columns(2)
        with left:
            st.subheader("Segment mix")
            counts = (
                filtered["SegmentName"]
                .value_counts()
                .rename_axis("SegmentName")
                .reset_index(name="loans")
            )
            st.plotly_chart(
                style(
                    px.pie(
                        counts,
                        names="SegmentName",
                        values="loans",
                        hole=0.5,
                        color="SegmentName",
                        color_discrete_map=colour_of,
                    ),
                    340,
                )
            )
        with right:
            st.subheader("Default rate by issue year")
            yearly = stats.rate_table(filtered, "IssueYear")
            line = px.line(
                yearly,
                x="IssueYear",
                y="default_rate",
                markers=True,
                color_discrete_sequence=[WARNING],
            )
            line.add_hline(y=portfolio_rate, line_dash="dash")
            st.plotly_chart(style(line, 340))
            st.caption(
                "Recent years look better partly because loans that will default later are not resolved yet."
            )

        st.subheader("Loans issued per year, by segment")
        per_year = filtered.groupby(["IssueYear", "SegmentName"]).size().reset_index(name="loans")
        st.plotly_chart(
            style(
                px.bar(
                    per_year,
                    x="IssueYear",
                    y="loans",
                    color="SegmentName",
                    color_discrete_map=colour_of,
                ),
                340,
            )
        )

    highest, lowest = profile["default_rate"].idxmax(), profile["default_rate"].idxmin()
    ratio = profile.loc[highest, "default_rate"] / profile.loc[lowest, "default_rate"]
    st.subheader("Key findings (full portfolio)")
    st.markdown(
        f"- Default rates run from {profile.loc[lowest, 'default_rate']:.1%} (*{names[lowest]}*) to "
        f"{profile.loc[highest, 'default_rate']:.1%} (*{names[highest]}*), about {ratio:.1f} times higher.\n"
        f"- The portfolio default rate is {portfolio_rate:.1%} across {total_loans:,} resolved loans.\n"
        f"- The largest segment holds {profile['share'].max():.0%} of loans and the smallest "
        f"{profile['share'].min():.0%}.\n"
        "- The segments were built without default, so this spread is an outside test that they capture "
        "something real about risk."
    )

# ---------- 2. Segments ----------
with segments_tab:
    st.subheader("Segment profiles (full portfolio, typical values)")
    display = profile_view[
        ["share", "Income", "LoanAmount", "DTI", "CreditScore", "EmploymentLength"]
    ].copy()
    display["share"] = (display["share"] * 100).round(1)
    display = display.rename(
        columns={
            "share": "Share of loans (%)",
            "LoanAmount": "Loan amount",
            "CreditScore": "Credit score",
            "EmploymentLength": "Years employed",
        }
    )
    st.dataframe(display)

    signature = load_table("segment_signature.csv", index_col="Segment")
    if signature is not None:
        st.subheader("Segment signature")
        st.caption(
            "Average standardised value of each variable. Red is above the portfolio average, blue below."
        )
        heat = px.imshow(
            signature.rename(index=label_of),
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-2,
            zmax=2,
            aspect="auto",
        )
        st.plotly_chart(style(heat, 360))

    st.subheader("What is each segment made of?")
    choice = st.selectbox(
        "Compare the make-up of the segments by", ["Purpose", "HomeOwnership", "State"]
    )
    files = {
        "Purpose": "mix_purpose.csv",
        "HomeOwnership": "mix_home.csv",
        "State": "mix_state.csv",
    }
    mix = load_table(files[choice], index_col="Segment")
    if mix is not None:
        overall = mix.sum() / mix.sum().sum()
        top = (
            overall.sort_values(ascending=False).head(5 if choice == "HomeOwnership" else 10).index
        )
        shares = mix.div(mix.sum(axis=1), axis=0)
        lift = (shares[top] / overall[top]).rename(index=label_of).round(2)
        st.plotly_chart(
            style(
                px.imshow(
                    lift,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=1,
                    aspect="auto",
                ),
                360,
            )
        )
        st.caption(
            "Lift of 2.00 means the category is twice as common in the segment as in the whole portfolio. "
            "Categories barely shaped the segments, so values stay close to 1."
        )
        if choice == "State":
            st.caption(
                "State is shown for description only. It can act as a proxy for protected characteristics "
                "and was never used to build or rank segments."
            )

# ---------- 3. Default patterns ----------
with default_tab:
    st.subheader("Default rate by segment (full portfolio)")
    ordered = profile_view.sort_values("default_rate")
    bars = go.Figure(
        go.Bar(
            x=ordered["default_rate"],
            y=list(ordered.index),
            orientation="h",
            marker_color=WARNING,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": ordered["ci_high"] - ordered["default_rate"],
                "arrayminus": ordered["default_rate"] - ordered["ci_low"],
            },
        )
    )
    bars.add_vline(x=portfolio_rate, line_dash="dash")
    bars.update_layout(
        xaxis_title="Default rate (bars show the 95% confidence interval; dashed line is the portfolio)"
    )
    st.plotly_chart(style(bars, 340))

    by_year = load_table("year_summary.csv")
    if by_year is not None:
        st.subheader("Does the ranking hold within each issue year?")
        grid = by_year[by_year["loans"] >= 500].pivot(
            index="Segment", columns="IssueYear", values="default_rate"
        )
        st.plotly_chart(
            style(
                px.imshow(
                    grid.rename(index=label_of),
                    text_auto=".2f",
                    color_continuous_scale="Reds",
                    aspect="auto",
                ),
                360,
            )
        )
    adjusted = load_table("default_adjusted_by_segment.csv", index_col=0)
    if adjusted is not None:
        st.subheader("Adjusting for issue year")
        st.caption(
            "Adjusted rates give every segment the same issue-year mix. Little change means the "
            "vintage mix does not distort the comparison."
        )
        st.dataframe(adjusted.round(4))

    st.subheader("Explore default rates (sample, uses the sidebar filters)")
    variable = st.selectbox(
        "Group loans by", ["CreditScore", "DTI", "Income", "LoanAmount", "EmploymentLength"]
    )
    if filtered.empty:
        st.warning("No sampled loans match the current filters.")
    else:
        bands = pd.qcut(filtered[variable], 8, duplicates="drop")
        band_rates = stats.rate_table(filtered.assign(Band=bands), "Band")
        band_rates = band_rates[band_rates["loans"] >= 300]
        band_rates["Band"] = band_rates["Band"].astype(str)
        explorer = go.Figure(
            go.Scatter(
                x=band_rates["Band"],
                y=band_rates["default_rate"],
                mode="lines+markers",
                line={"color": WARNING},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": band_rates["ci_high"] - band_rates["default_rate"],
                    "arrayminus": band_rates["default_rate"] - band_rates["ci_low"],
                },
            )
        )
        explorer.update_layout(
            xaxis_title=f"{variable} (equal-sized bands)", yaxis_title="Default rate"
        )
        st.plotly_chart(style(explorer, 340))

    st.subheader("Default rate by state (full portfolio, description only)")
    states = load_table("state_summary.csv")
    if states is not None:
        states = states[states["loans"] >= 2000].sort_values("default_rate", ascending=False)
        shown = pd.concat([states.head(8), states.tail(8)])
        state_chart = px.bar(
            shown,
            x="default_rate",
            y="State",
            orientation="h",
            color_discrete_sequence=[ACCENT],
            hover_data=["loans"],
        )
        state_chart.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style(state_chart, 440))
        st.caption(
            "The eight highest and eight lowest states with at least 2,000 loans. State was never used "
            "to build segments, and it can proxy protected characteristics."
        )

# ---------- 4. Maps and anomalies ----------
with maps_tab:
    umap_table = load_table("umap_sample.csv")
    if umap_table is None:
        st.info("Run notebook 10 to create the map data.")
    else:
        st.subheader("UMAP map of 20,000 borrowers")
        umap_table["SegmentName"] = umap_table["Segment"].map(label_of)
        umap_table["Unusual"] = umap_table["Anomaly"].map({True: "Unusual", False: "Typical"})
        umap_table["log10 income"] = np.log10(umap_table["Income"])
        options = {
            "Segment": "SegmentName",
            "Unusual profile": "Unusual",
            "Credit score": "CreditScore",
            "DTI": "DTI",
            "log10 income": "log10 income",
            "Loan amount": "LoanAmount",
        }
        picked = st.selectbox("Colour the map by", list(options))
        column = options[picked]
        if column == "SegmentName":
            scatter = px.scatter(
                umap_table,
                x="UMAP1",
                y="UMAP2",
                color=column,
                color_discrete_map=colour_of,
                opacity=0.6,
                render_mode="webgl",
            )
        elif column == "Unusual":
            scatter = px.scatter(
                umap_table,
                x="UMAP1",
                y="UMAP2",
                color=column,
                opacity=0.6,
                render_mode="webgl",
                color_discrete_map={"Typical": "#B7C9C2", "Unusual": WARNING},
            )
        else:
            scatter = px.scatter(
                umap_table,
                x="UMAP1",
                y="UMAP2",
                color=column,
                opacity=0.6,
                render_mode="webgl",
                color_continuous_scale="Viridis",
            )
        scatter.update_traces(marker={"size": 4})
        st.plotly_chart(style(scatter, 520))
        st.caption(
            "UMAP squeezes the data onto a flat map. Nearby borrowers are similar, but distances between "
            "distant regions are not meaningful. The data form a continuum, not separate islands."
        )

    st.subheader("Unusual borrower profiles (Isolation Forest, full portfolio)")
    anomaly_profile = load_table("anomaly_profile.csv", index_col="Anomaly")
    anomaly_by_segment = load_table("anomaly_by_segment.csv", index_col="Segment")
    if anomaly_profile is not None and anomaly_by_segment is not None:
        anomaly_profile.index = [
            "Unusual" if str(v) == "True" else "Typical" for v in anomaly_profile.index
        ]
        flagged = int(anomaly_by_segment["flagged"].sum())
        left, right = st.columns(2)
        with left.container(border=True):
            st.caption("Borrowers flagged as unusual")
            st.subheader(f"{flagged:,} ({flagged / total_loans:.1%})")
        with right.container(border=True):
            st.caption("Default rate: unusual vs typical")
            st.subheader(
                f"{anomaly_profile.loc['Unusual', 'default_rate']:.1%} vs "
                f"{anomaly_profile.loc['Typical', 'default_rate']:.1%}"
            )
        st.dataframe(
            anomaly_profile[
                [
                    "Income",
                    "LoanAmount",
                    "DTI",
                    "CreditScore",
                    "EmploymentLength",
                    "loans",
                    "default_rate",
                ]
            ]
        )
        share = anomaly_by_segment.rename(index=label_of).reset_index()
        share["flagged_share"] = share["flagged_share"] * 100
        st.plotly_chart(
            style(
                px.bar(
                    share,
                    x="Segment",
                    y="flagged_share",
                    color_discrete_sequence=[WARNING],
                    labels={"flagged_share": "Share of the segment flagged (%)"},
                ),
                320,
            )
        )
        st.caption(
            "Unusual means statistically rare in the combination of income, loan, DTI, credit score and "
            "tenure. It does not mean risky or suspicious. Loans with impossible DTI values are treated "
            "separately as data-quality problems."
        )
        st.dataframe(sample[sample["Anomaly"]].head(25), hide_index=True)

# ---------- 5. Place a borrower ----------
with borrower_tab:
    st.subheader("Place a borrower in a segment")
    st.write(
        "This applies the same saved preprocessing, PCA and K-Means used on the historical loans. "
        "It describes which group the profile resembles. It is not a credit decision."
    )
    try:
        model = load_model()
        options = category_options(model)
    except FileNotFoundError as error:
        st.error(f"Saved models not found: {error}. Run the notebooks first.")
        model = None

    if model is not None:
        purposes = options["Purpose"]
        default_purpose = (
            purposes.index("debt_consolidation") if "debt_consolidation" in purposes else 0
        )
        with st.form("borrower_form"):
            left, right = st.columns(2)
            with left:
                income = st.number_input(
                    "Annual income", min_value=1000, max_value=2_000_000, value=65000, step=1000
                )
                loan_amount = st.number_input(
                    "Loan amount", min_value=500, max_value=100_000, value=12000, step=500
                )
                dti = st.number_input(
                    "Debt-to-income ratio", min_value=0.0, max_value=100.0, value=18.0, step=0.5
                )
                credit_score = st.number_input(
                    "Credit score (FICO)", min_value=300, max_value=850, value=690, step=1
                )
            with right:
                employment_known = st.checkbox("Employment length known", value=True)
                employment = st.slider("Employment length (years, 10 = 10 or more)", 0, 10, 5)
                purpose = st.selectbox("Loan purpose", purposes, index=default_purpose)
                home = st.selectbox("Home ownership", options["HomeOwnership"])
            submitted = st.form_submit_button("Place borrower", type="primary")

        if submitted:
            borrower = borrower_frame(
                income,
                dti,
                loan_amount,
                credit_score,
                employment if employment_known else None,
                purpose,
                home,
            )
            result, distances = model.assign(borrower)
            row = result.iloc[0]
            segment = int(row["Segment"])

            first, second, third = st.columns(3)
            with first.container(border=True):
                st.caption("Closest segment")
                st.subheader(label_of.get(segment, str(segment)))
            with second.container(border=True):
                st.caption("Historical default rate of that segment")
                st.subheader(f"{profile.loc[segment, 'default_rate']:.1%}")
            with third.container(border=True):
                st.caption("Profile")
                st.subheader("Unusual" if row["IsAnomaly"] else "Typical")

            st.caption(
                f"The segment's default rate is a group figure with a 95% interval of "
                f"{profile.loc[segment, 'ci_low']:.1%} to {profile.loc[segment, 'ci_high']:.1%}. "
                "It says nothing certain about this borrower."
            )
            if row["ClosenessRatio"] > 0.9:
                st.warning(
                    "This borrower sits close to the border between two segments, so the assignment is "
                    "not clear-cut. Segments divide a continuum."
                )

            distance_table = pd.DataFrame(
                {
                    "Segment": [label_of[s] for s in sorted(names)],
                    "Distance": distances[0][: len(names)],
                }
            )
            st.plotly_chart(
                style(
                    px.bar(
                        distance_table,
                        x="Distance",
                        y="Segment",
                        orientation="h",
                        color_discrete_sequence=[ACCENT],
                    ),
                    300,
                )
            )
            st.caption(
                "Distance to each segment centre in the model's standardised space. Shorter means more similar."
            )

# ---------- 6. Method and limits ----------
with method_tab:
    comparison = load_table("algorithm_comparison.csv", index_col=0)
    if comparison is not None:
        st.subheader("Algorithm comparison")
        st.caption(
            "A rough comparison: the methods ran on different samples, and DBSCAN ignores its noise points."
        )
        st.dataframe(comparison)

    variance = load_table("pca_variance.csv")
    if variance is not None:
        st.subheader("PCA: variance explained")
        pca_chart = go.Figure()
        pca_chart.add_trace(
            go.Bar(x=variance["component"], y=variance["explained"], name="Each component")
        )
        pca_chart.add_trace(
            go.Scatter(x=variance["component"], y=variance["cumulative"], name="Cumulative")
        )
        pca_chart.update_layout(xaxis_title="Component", yaxis_title="Share of variance")
        st.plotly_chart(style(pca_chart, 320))

    left, right = st.columns(2)
    with left:
        show_image("13_kmeans_k_selection.png", "Choosing k for K-Means")
    with right:
        show_image("15_dendrogram.png", "Hierarchical clustering dendrogram")
