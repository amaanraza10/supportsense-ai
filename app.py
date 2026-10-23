import gc
import threading

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SupportSense AI", layout="wide")
st.title("SupportSense AI")
st.caption("Multilingual (English / Hindi / Punjabi) customer support analytics. "
           "Demo data and ticket metadata are synthetic.")
st.info("The first tab runs the project's fine-tuned XLM-RoBERTa classifiers live on any message you type. "
        "The other tabs show results precomputed by the same models (plus LaBSE embeddings). "
        "See the [GitHub repository](https://github.com/amaanraza10/supportsense-ai) for details.")

LANG_NAME = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}
HF_USER = "AmaanRaza01"          # the three models live at AmaanRaza01/{category,urgency,sentiment}-model
TASKS = ["category", "urgency", "sentiment"]
MAX_CHARS = 1000


@st.cache_data
def load_data():
    d = pd.read_parquet("tickets_scored.parquet").reset_index(drop=True)
    d["created_at"] = pd.to_datetime(d["created_at"])
    e = np.load("embeddings.npy").astype("float32")
    return d, e


df, emb = load_data()

# Category -> team mapping and class lists, taken from the scored data so they always match the dashboard.
ROUTE = df.groupby("pred_category")["route_to"].agg(lambda s: s.mode().iat[0]).to_dict()
CLASSES = {
    "category": sorted(df["pred_category"].unique()),
    "urgency": sorted(df["pred_urgency"].unique()),
    "sentiment": sorted(df["pred_sentiment"].unique()),
}


# ------------------------------------------------------------------ live analysis helpers
@st.cache_resource
def _model_lock():
    """Only one analysis runs at a time, so two visitors cannot load models simultaneously
    and push the free-tier server over its memory limit."""
    return threading.Lock()


def _hf_token():
    try:
        return st.secrets.get("HF_TOKEN")      # optional, only needed if the model repos are private
    except Exception:
        return None


def detect_language(text):
    """Cheap script-based guess: Gurmukhi -> Punjabi, Devanagari -> Hindi, otherwise English."""
    gur = sum("\u0A00" <= ch <= "\u0A7F" for ch in text)
    dev = sum("\u0900" <= ch <= "\u097F" for ch in text)
    if gur == 0 and dev == 0:
        return "en"
    return "pa" if gur >= dev else "hi"


def _label_names(model, task):
    """Use the label names stored in the model config. If the config only has LABEL_0, LABEL_1, ...
    fall back to the alphabetical class list from the data (the usual LabelEncoder order)."""
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    generic = all(str(v).upper().startswith("LABEL_") for v in id2label.values())
    classes = CLASSES[task]
    if generic and len(classes) == len(id2label):
        return {i: classes[i] for i in range(len(classes))}, True
    return id2label, False


def _classify_one(task, text):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = f"{HF_USER}/{task}-model"
    token = _hf_token()
    tok = AutoTokenizer.from_pretrained(repo, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, token=token, low_cpu_mem_usage=True).eval()
    try:
        with torch.no_grad():
            enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        k = int(probs.argmax())
        names, guessed = _label_names(model, task)
        return names[k], float(probs[k]), guessed
    finally:
        del model, tok                      # free memory before the next model loads
        gc.collect()


@st.cache_data(show_spinner=False, max_entries=64)
def analyze(text):
    out = {}
    with _model_lock():
        for task in TASKS:
            out[task] = _classify_one(task, text)
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def embed_text(text):
    import torch
    from sentence_transformers import SentenceTransformer

    with _model_lock():
        try:
            m = SentenceTransformer("sentence-transformers/LaBSE", device="cpu",
                                    model_kwargs={"torch_dtype": torch.bfloat16})   # half the memory
        except Exception:
            m = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
        try:
            v = m.encode([text], convert_to_tensor=True, normalize_embeddings=True)
            v = v.float().cpu().numpy()[0]
        finally:
            del m
            gc.collect()
    v = v.astype("float32")
    return v / max(float(np.linalg.norm(v)), 1e-9)


EXAMPLES = {
    "(write my own)": "",
    "English - delivery": "My order was supposed to arrive last week and it still has not shown up. Please help.",
    "Hindi - delivery": "मेरा ऑर्डर पिछले हफ्ते आना था लेकिन अभी तक नहीं पहुँचा। कृपया मदद करें।",
    "Punjabi - delivery": "ਮੇਰਾ ਆਰਡਰ ਪਿਛਲੇ ਹਫ਼ਤੇ ਆਉਣਾ ਸੀ ਪਰ ਹੁਣ ਤੱਕ ਨਹੀਂ ਪਹੁੰਚਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਮਦਦ ਕਰੋ।",
    "English - payment failed": "Money was deducted from my account but the payment shows as failed.",
    "Hindi - account hacked": "मेरा अकाउंट हैक हो गया है और किसी ने मेरा पासवर्ड बदल दिया है।",
    "Punjabi - rude agent": "ਤੁਹਾਡੇ ਸਪੋਰਟ ਏਜੰਟ ਨੇ ਮੇਰੇ ਨਾਲ ਬਦਤਮੀਜ਼ੀ ਨਾਲ ਗੱਲ ਕੀਤੀ।",
}


def _fill_example():
    st.session_state["live_msg"] = EXAMPLES[st.session_state["live_example"]]


st.session_state.setdefault("live_msg", "")

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Analyze a message", "Explore a ticket", "Issue clusters", "Dashboard", "Search & export"])

# ---------------------------------------------------------------- Tab 0 (live analysis)
with tab0:
    st.write("Type a customer message in English, Hindi or Punjabi and the fine-tuned models will "
             "predict its category, urgency and sentiment, and suggest which team should handle it.")
    st.caption("The first analysis after the app wakes up downloads the models (a few minutes). "
               "After that, each analysis takes roughly 10 to 30 seconds.")
    st.selectbox("Try an example", list(EXAMPLES), key="live_example", on_change=_fill_example)
    msg = st.text_area("Customer message", key="live_msg", height=120, max_chars=MAX_CHARS)
    want_similar = st.checkbox("Also find the most similar tickets (loads LaBSE, slower)")

    if st.button("Analyze", type="primary"):
        text = msg.strip()
        if not text:
            st.warning("Enter a message first.")
        else:
            try:
                with st.spinner("Running the models... the first run can take a few minutes."):
                    res = analyze(text)
                sim_table = None
                if want_similar:
                    with st.spinner("Finding similar tickets..."):
                        v = embed_text(text)
                    sims = emb @ v
                    order = np.argsort(-sims)[:6]
                    sim_table = df.loc[order, ["text", "language", "pred_category"]].copy()
                    sim_table.insert(0, "similarity", np.round(sims[order], 2))
                st.session_state["live_result"] = {"text": text, "res": res, "sim": sim_table}
            except Exception as exc:                       # network, memory, or repo-name problems
                st.session_state.pop("live_result", None)
                st.error("Could not run the models. If this keeps happening, reboot the app from "
                         "'Manage app' and try again.")
                st.exception(exc)

    shown = st.session_state.get("live_result")
    if shown:
        res = shown["res"]
        cat, urg, sen = res["category"][0], res["urgency"][0], res["sentiment"][0]
        st.markdown(f"**Message:** {shown['text']}")
        m = st.columns(5)
        m[0].metric("Language", LANG_NAME[detect_language(shown["text"])])
        m[1].metric("Category", cat)
        m[1].caption(f"{res['category'][1]:.0%} confidence")
        m[2].metric("Sentiment", sen)
        m[2].caption(f"{res['sentiment'][1]:.0%} confidence")
        m[3].metric("Urgency", urg)
        m[3].caption(f"{res['urgency'][1]:.0%} confidence")
        m[4].metric("Route to", ROUTE.get(cat, "Support"))
        if any(res[t][2] for t in TASKS):
            st.caption("Label names were inferred from the training data's class order because the "
                       "model configs only store numeric labels.")
        if shown["sim"] is not None:
            st.subheader("Most similar tickets (any language)")
            st.dataframe(shown["sim"], hide_index=True)

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
        res_hours = d.groupby("pred_category")["resolution_hours"].mean().reset_index()
        g2.plotly_chart(px.bar(res_hours, x="pred_category", y="resolution_hours",
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
