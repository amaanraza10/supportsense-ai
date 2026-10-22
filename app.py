import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SupportSense AI", layout="wide")
st.title("SupportSense AI")
st.caption("Multilingual (English / Hindi / Punjabi) customer support analytics. "
           "Demo data and ticket metadata are synthetic.")
st.info("This hosted demo shows results precomputed by the project's fine-tuned models "
        "(XLM-RoBERTa classifiers and LaBSE embeddings). Live analysis of a new message "
        "runs in the Colab / local version of the project. See the GitHub repository for details.")

LANG_NAME = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}


@st.cache_data
def load_data():
    d = pd.read_parquet("tickets_scored.parquet").reset_index(drop=True)
    d["created_at"] = pd.to_datetime(d["created_at"])
    e = np.load("embeddings.npy").astype("float32")
    return d, e


df, emb = load_data()
tab1, tab2, tab3, tab4 = st.tabs(["Explore a ticket", "Issue clusters", "Dashboard", "Search & export"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.write("Pick any ticket to see the model's predictions and its most similar tickets "
             "in all three languages.")
    c1, c2 = st.columns(2)
    lang = c1.selectbox("Language", sorted(df["language"].unique()), format_func=lambda x: LANG_NAME.get(x, x))
    cat = c2.selectbox("Predicted category", sorted(df["pred_category"].unique()))
    subset = df[(df["language"] == lang) & (df["pred_category"] == cat)]
    if subset.empty:
        st.warning("No tickets for this combination.")
    else:
        idx = st.selectbox("Ticket", subset.index.tolist(), format_func=lambda i: df.at[i, "text"][:100])
        row = df.loc[idx]
        st.markdown(f"**Message:** {row['text']}")
        m = st.columns(5)
        m[0].metric("Language", LANG_NAME.get(row["language"], row["language"]))
        m[1].metric("Category", row["pred_category"])
        m[2].metric("Sentiment", row["pred_sentiment"])
        m[3].metric("Urgency", row["pred_urgency"])
        m[4].metric("Route to", row["route_to"])

        sims = emb @ emb[idx]          # embeddings are normalized, so this is cosine similarity
        order = [i for i in np.argsort(-sims) if i != idx][:6]
        similar = df.loc[order, ["text", "language", "pred_category"]].copy()
        similar.insert(0, "similarity", np.round(sims[order], 2))
        st.subheader("Most similar tickets (any language)")
        st.dataframe(similar, hide_index=True)

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.write("Groups of tickets about the same issue, found with multilingual embeddings. "
             "A cluster with 3 languages shows the same problem reported in English, Hindi and Punjabi.")
    grouped = df[df["cluster"] >= 0].groupby("cluster")
    if len(grouped) == 0:
        st.warning("No clusters were found in the data.")
    else:
        summary = grouped.agg(tickets=("text", "size"),
                              languages=("language", "nunique"),
                              main_category=("pred_category", lambda s: s.mode().iat[0])).reset_index()
        only_multi = st.checkbox("Show only clusters that contain all 3 languages", value=True)
        if only_multi:
            summary = summary[summary["languages"] == 3]
        summary = summary.sort_values("tickets", ascending=False)
        st.write(f"{len(summary)} clusters")
        st.dataframe(summary, hide_index=True)
        if not summary.empty:
            pick = st.selectbox("Open a cluster", summary["cluster"].tolist())
            st.dataframe(df[df["cluster"] == pick][["text", "language", "pred_category", "pred_urgency"]].head(60),
                         hide_index=True)

# ---------------------------------------------------------------- Tab 3
with tab3:
    f1, f2 = st.columns(2)
    all_langs = sorted(df["language"].unique())
    all_cats = sorted(df["pred_category"].unique())
    langs = f1.multiselect("Language", all_langs, default=all_langs)
    cats = f2.multiselect("Category", all_cats, default=all_cats)
    d = df[df["language"].isin(langs) & df["pred_category"].isin(cats)]

    if d.empty:
        st.warning("No tickets match the selected filters.")
    else:
        k = st.columns(4)
        k[0].metric("Tickets", f"{len(d):,}")
        k[1].metric("High urgency", f"{(d['pred_urgency'] == 'High').mean():.0%}")
        k[2].metric("Negative sentiment", f"{(d['pred_sentiment'] == 'Negative').mean():.0%}")
        k[3].metric("Avg resolution (hours)", f"{d['resolution_hours'].mean():.0f}")

        st.plotly_chart(px.histogram(d, x="pred_category", color="language", barmode="group",
                                     title="Complaints by category and language"))

        d = d.assign(month=d["created_at"].dt.to_period("M").astype(str))
        trend = d.groupby(["month", "pred_category"]).size().reset_index(name="count")
        st.plotly_chart(px.line(trend, x="month", y="count", color="pred_category", markers=True,
                                title="Monthly complaint trend"))

        g1, g2 = st.columns(2)
        g1.plotly_chart(px.histogram(d, x="pred_category", color="pred_urgency",
                                     title="Urgency by category"))
        res = d.groupby("pred_category")["resolution_hours"].mean().reset_index()
        g2.plotly_chart(px.bar(res, x="pred_category", y="resolution_hours",
                               title="Average resolution time (hours)"))

        g3, g4 = st.columns(2)
        g3.plotly_chart(px.histogram(d, x="pred_sentiment", color="language", barmode="group",
                                     title="Sentiment by language"))
        by_region = d.groupby(["region", "pred_category"]).size().reset_index(name="count")
        g4.plotly_chart(px.bar(by_region, x="region", y="count", color="pred_category",
                               title="Complaints by region"))
        st.caption("Dates, regions, products and resolution times are simulated for demonstration.")

# ---------------------------------------------------------------- Tab 4
with tab4:
    kw = st.text_input("Search tickets (keyword in any language)")
    result = df[df["text"].str.contains(kw, case=False, na=False, regex=False)] if kw else df
    show = ["text", "language", "pred_category", "pred_sentiment", "pred_urgency", "route_to"]
    st.write(f"{len(result):,} tickets")
    st.dataframe(result[show], hide_index=True)
    st.download_button("Download report (CSV)", result[show].to_csv(index=False).encode("utf-8-sig"),
                       "supportsense_report.csv", "text/csv")
