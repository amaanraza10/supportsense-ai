import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SupportSense AI", layout="wide")
st.title("SupportSense AI")
st.caption("Multilingual (English / Hindi / Punjabi) customer support analytics. "
           "Demo data and ticket metadata are synthetic.")
with st.sidebar:
    st.header("SupportSense AI")
    st.write("Turns multilingual customer messages into a category, urgency, sentiment and a suggested "
             "team, and shows trends, clusters and alerts across all tickets.")
    st.markdown("**Languages:** English, Hindi, Punjabi  \n"
                "**Models:** 3 fine-tuned XLM-RoBERTa classifiers + LaBSE embeddings")
    st.markdown("[GitHub repository](https://github.com/amaanraza10/supportsense-ai)  \n"
                "[Models on Hugging Face](https://huggingface.co/AmaanRaza01)")
    with st.expander("How the live analysis works"):
        st.write("Each model runs in its own short-lived process so the free server never runs out of "
                 "memory. Dashboard and Search start from stored results for all tickets and switch to "
                 "live results when you run the models on a sample.")
    st.caption("All tickets and ticket metadata (dates, regions, resolution times) are synthetic.")

LANG_NAME = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}
HF_USER = "AmaanRaza01"          # models live at AmaanRaza01/{category,urgency,sentiment}-model
TASKS = ["category", "urgency", "sentiment"]
MAX_CHARS = 1000
LOW_CONFIDENCE = 0.60
URGENCY_COLORS = {"High": "#D64545", "Medium": "#F2A33A", "Low": "#3BA55D"}
BATCH = 16
WORKER = Path(__file__).with_name("worker.py")


@st.cache_data
def load_data():
    d = pd.read_parquet("tickets_scored.parquet").reset_index(drop=True)
    d["created_at"] = pd.to_datetime(d["created_at"])
    e = np.load("embeddings.npy").astype("float32")
    return d, e


df, emb = load_data()

# Category -> team mapping and class lists, taken from the scored data so they always match.
ROUTE = df.groupby("pred_category")["route_to"].agg(lambda s: s.mode().iat[0]).to_dict()
CLASSES = {
    "category": sorted(df["pred_category"].unique()),
    "urgency": sorted(df["pred_urgency"].unique()),
    "sentiment": sorted(df["pred_sentiment"].unique()),
}


# ================================================================== live inference helpers
@st.cache_resource
def _model_lock():
    """One model run at a time across all visitors, so two people cannot load models together
    and push the free-tier server over its memory limit."""
    return threading.Lock()


def _load_example_cache():
    """Predictions and embeddings for the built-in examples, computed once with the same models
    (see make_example_cache.py). They make the examples answer instantly; any other text is run live."""
    path = Path(__file__).with_name("example_cache.json")
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    preds, vecs = {}, {}
    for text, r in data.items():
        if all(t in r for t in TASKS):
            preds[text] = {t: (r[t][0], float(r[t][1]), bool(r[t][2])) for t in TASKS}
        if "vec" in r:
            vecs[text] = np.array(r["vec"], dtype="float32")
    return preds, vecs


@st.cache_resource
def _pred_cache():
    return _load_example_cache()[0]      # text -> {task: (label, confidence, label_names_were_guessed)}


@st.cache_resource
def _emb_cache():
    return _load_example_cache()[1]      # text -> normalized LaBSE vector


def _hf_token():
    try:
        return st.secrets.get("HF_TOKEN")      # optional; only needed if the model repos are private
    except Exception:
        return None


def detect_language(text):
    """Cheap script-based guess: Gurmukhi -> Punjabi, Devanagari -> Hindi, otherwise English."""
    gur = sum("\u0A00" <= ch <= "\u0A7F" for ch in text)
    dev = sum("\u0900" <= ch <= "\u097F" for ch in text)
    if gur == 0 and dev == 0:
        return "en"
    return "pa" if gur >= dev else "hi"


def _run_worker(job):
    """Run one model job in a fresh Python process. When it exits, ALL its memory returns to the
    server, so the app itself stays small and cannot be killed for using too much RAM."""
    with tempfile.TemporaryDirectory() as tmp:
        job_file, out_file = os.path.join(tmp, "job.json"), os.path.join(tmp, "out.json")
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job, f)
        env = dict(os.environ)
        token = _hf_token()
        if token:
            env["HF_TOKEN"] = token
        proc = subprocess.run([sys.executable, str(WORKER), job_file, out_file],
                              capture_output=True, text=True, env=env, timeout=1800)
        if proc.returncode != 0 or not os.path.exists(out_file):
            why = ("The model process was killed (the server most likely ran out of memory)."
                   if proc.returncode in (-9, 137) else "The model process failed.")
            raise RuntimeError(f"{why}\n{(proc.stderr or '')[-1500:]}")
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)


def _run_task(task, texts, out, progress, task_no):
    """Classify all texts with ONE model (in its own process)."""
    if progress:
        progress(task_no / len(TASKS),
                 f"Running the {task} model on {len(texts)} text(s)... (the first run downloads it)")
    res = _run_worker({"kind": "classify", "task": task, "repo": f"{HF_USER}/{task}-model",
                       "texts": texts, "classes": CLASSES[task], "batch": BATCH})
    for t, (label, conf) in zip(texts, res["labels"]):
        out[t][task] = (label, float(conf), bool(res["guessed"]))
    if progress:
        progress((task_no + 1) / len(TASKS), f"Finished the {task} model")


def predict(texts, progress=None):
    """Run the three classifiers live on `texts`. Results are cached per text, so a text is only
    ever analysed once while the server stays up."""
    cache = _pred_cache()
    if len(cache) > 3000:
        cache.clear()
    texts = list(texts)
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        with _model_lock():
            todo = [t for t in todo if t not in cache]      # another visitor may have just done it
            if todo:
                partial = {t: {} for t in todo}
                for n, task in enumerate(TASKS):
                    _run_task(task, todo, partial, progress, n)
                cache.update(partial)
    return {t: cache[t] for t in texts}


def embed_texts(texts):
    """LaBSE embeddings, computed live (in their own process) and cached per text."""
    cache = _emb_cache()
    if len(cache) > 3000:
        cache.clear()
    texts = list(texts)
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        with _model_lock():
            todo = [t for t in todo if t not in cache]
            if todo:
                v = np.array(_run_worker({"kind": "embed", "texts": todo, "batch": BATCH})["vectors"],
                             dtype="float32")
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                for t, vec in zip(todo, v):
                    cache[t] = vec
    return np.stack([cache[t] for t in texts])


def run_live(texts, what):
    """predict() with a progress bar."""
    bar = st.progress(0.0, text=f"Starting the models for {what}. The first run downloads them "
                                "(a few minutes).")
    try:
        return predict(texts, lambda f, msg: bar.progress(min(max(f, 0.0), 1.0), text=msg))
    finally:
        bar.empty()


@contextlib.contextmanager
def guard():
    """Show a friendly error instead of crashing the whole page when a live run fails."""
    try:
        yield
    except Exception as exc:                 # network, memory, or repo-name problems
        st.error("Could not run the models. Please try again in a minute. If it keeps happening, "
                 "reboot the app from 'Manage app'.")
        with st.expander("Technical details"):
            st.code(str(exc)[-1500:])


URGENCY_BANNER = {"High": st.error, "Medium": st.warning, "Low": st.success}


def show_prediction(text, res, lang_label=None, stored=None):
    """Render one live prediction: a priority banner, key metrics and confidence bars.
    `stored` = the precomputed row, for comparison."""
    cat, urg, sen = res["category"][0], res["urgency"][0], res["sentiment"][0]
    team = ROUTE.get(cat, "Support")
    URGENCY_BANNER.get(urg, st.info)(f"**{urg} priority**: route to **{team}** "
                                     f"({cat}, {sen.lower()} tone)")
    m = st.columns(5)
    m[0].metric("Language", lang_label or LANG_NAME[detect_language(text)])
    m[1].metric("Category", cat)
    m[2].metric("Sentiment", sen)
    m[3].metric("Urgency", urg)
    m[4].metric("Route to", team)

    st.caption("Model confidence (the model's prediction score, not a guaranteed probability)")
    for task in TASKS:
        label, conf = res[task][0], float(res[task][1])
        st.progress(min(max(conf, 0.0), 1.0), text=f"{task.capitalize()}: {label} ({conf:.0%})")
    if min(float(res[t][1]) for t in TASKS) < LOW_CONFIDENCE:
        st.warning("Low confidence on at least one prediction. In a real support system this ticket "
                   "would be sent to a human agent for review.")
    if any(res[t][2] for t in TASKS):
        st.caption("Label names were inferred from the training data's class order because the "
                   "model configs only store numeric labels.")
    if stored is not None:
        same = sum([cat == stored["pred_category"], urg == stored["pred_urgency"],
                    sen == stored["pred_sentiment"]])
        st.caption(f"Live predictions match the stored ones on {same} of 3 tasks.")


SIM_CONFIG = {"similarity": st.column_config.ProgressColumn(
    "Similarity", min_value=0.0, max_value=1.0, format="%.2f")}


def nearest_table(vec, exclude=None, k=6):
    sims = emb @ vec
    order = [i for i in np.argsort(-sims) if i != exclude][:k]
    out = df.loc[order, ["text", "language", "pred_category"]].copy()
    out.insert(0, "similarity", np.round(sims[order], 2))
    return out, sims


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


def _fill_cluster_example():
    st.session_state["cluster_msg"] = EXAMPLES[st.session_state["cluster_example"]]


st.session_state.setdefault("live_msg", "")
st.session_state.setdefault("cluster_msg", "")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Analyze a message", "Explore a ticket", "Issue clusters", "Dashboard", "Search & export", "Alerts"])

# ---------------------------------------------------------------- Tab 0: analyze any message live
with tab0:
    st.write("Type a customer message in English, Hindi or Punjabi and the fine-tuned models will "
             "predict its category, urgency and sentiment, and suggest which team should handle it.")
    st.caption("The built-in examples are precomputed, so they answer instantly. Your own text runs the "
               "models live: the first analysis after the app wakes up downloads them (a few minutes), "
               "then each analysis takes roughly 10 to 30 seconds.")
    st.selectbox("Try an example", list(EXAMPLES), key="live_example", on_change=_fill_example)
    msg = st.text_area("Customer message", key="live_msg", height=120, max_chars=MAX_CHARS)
    want_similar = st.checkbox("Also find the most similar tickets (loads LaBSE, slower)")

    if st.button("Analyze", type="primary", key="analyze_btn"):
        text = msg.strip()
        if not text:
            st.warning("Enter a message first.")
        else:
            with guard():
                res = run_live([text], "your message")[text]
                sim_table = None
                if want_similar:
                    with st.spinner("Finding similar tickets..."):
                        sim_table, _ = nearest_table(embed_texts([text])[0])
                st.session_state["live_result"] = {"text": text, "res": res, "sim": sim_table}

    shown = st.session_state.get("live_result")
    if shown:
        st.markdown(f"**Message:** {shown['text']}")
        show_prediction(shown["text"], shown["res"])
        if shown["sim"] is not None:
            st.subheader("Most similar tickets (any language)")
            st.dataframe(shown["sim"], hide_index=True, column_config=SIM_CONFIG)

# ---------------------------------------------------------------- Tab 1: explore a ticket (live)
with tab1:
    st.write("Pick any ticket and run the models on it live: predictions, routing, and its most "
             "similar tickets in all three languages (found with the ticket's LaBSE embedding).")
    c1, c2 = st.columns(2)
    lang = c1.selectbox("Language", sorted(df["language"].unique()), format_func=lambda x: LANG_NAME.get(x, x))
    cat_filter = c2.selectbox("Browse by category", sorted(df["pred_category"].unique()))
    subset = df[(df["language"] == lang) & (df["pred_category"] == cat_filter)]
    if subset.empty:
        st.warning("No tickets for this combination.")
    else:
        idx = st.selectbox("Ticket", subset.index.tolist(), format_func=lambda i: df.at[i, "text"][:100])
        row = df.loc[idx]
        st.markdown(f"**Message:** {row['text']}")
        if st.button("Analyze this ticket live", type="primary", key="explore_btn"):
            with guard():
                text = row["text"]
                res = run_live([text], "this ticket")[text]
                sim_table, _ = nearest_table(emb[idx], exclude=idx)   # the ticket's own LaBSE vector
                st.session_state["explore_live"] = {"idx": idx, "res": res, "sim": sim_table}
        got = st.session_state.get("explore_live")
        if got and got["idx"] == idx:
            show_prediction(row["text"], got["res"], lang_label=LANG_NAME.get(row["language"]), stored=row)
            st.subheader("Most similar tickets (any language)")
            st.dataframe(got["sim"], hide_index=True, column_config=SIM_CONFIG)
        else:
            st.caption("Press the button to run the models on this ticket.")

# ---------------------------------------------------------------- Tab 2: clusters (live assignment)
with tab2:
    st.write("Groups of tickets about the same issue, found with multilingual embeddings. "
             "A cluster with 3 languages shows the same problem reported in English, Hindi and Punjabi.")

    st.subheader("Find the cluster of a new message (live)")
    st.selectbox("Try an example", list(EXAMPLES), key="cluster_example", on_change=_fill_cluster_example)
    cmsg = st.text_input("Message", key="cluster_msg", max_chars=MAX_CHARS)
    if st.button("Find its cluster live", key="cluster_btn"):
        text = cmsg.strip()
        if not text:
            st.warning("Enter a message first.")
        else:
            with guard():
                with st.spinner("Embedding the message with LaBSE..."):
                    vec = embed_texts([text])[0]
                sims = emb @ vec
                top = np.argsort(-sims)[:10]
                votes = df.loc[top[df.loc[top, "cluster"].values >= 0], "cluster"]
                best = int(votes.mode().iat[0]) if len(votes) else None
                st.session_state["cluster_live"] = {
                    "text": text, "best": best,
                    "top_sim": float(sims[top[0]]),
                    "neighbors": df.loc[top[:6], ["text", "language", "pred_category", "cluster"]]
                                   .assign(similarity=np.round(sims[top[:6]], 2))}
    cl = st.session_state.get("cluster_live")
    if cl:
        if cl["best"] is None:
            st.warning("Its nearest tickets are not in any cluster.")
        else:
            members = df[df["cluster"] == cl["best"]]
            st.success(f"Closest cluster: #{cl['best']} ({len(members)} tickets, "
                       f"{members['language'].nunique()} language(s), mostly "
                       f"{members['pred_category'].mode().iat[0]}). "
                       f"Best match similarity {cl['top_sim']:.2f}.")
            st.dataframe(members[["text", "language", "pred_category", "pred_urgency"]].head(30),
                         hide_index=True)
        st.caption("Nearest tickets used for the vote:")
        st.dataframe(cl["neighbors"][["similarity", "text", "language", "pred_category", "cluster"]],
                     hide_index=True, column_config=SIM_CONFIG)

    st.divider()
    st.subheader("All clusters in the ticket collection")
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

# ---------------------------------------------------------------- Tab 3: dashboard (live sample)
with tab3:
    st.write("Run the models live on a random sample of tickets and the dashboard recomputes from the "
             "live predictions. Without a live run it shows the stored results for all tickets.")
    s1, s2 = st.columns([3, 1])
    n_sample = s1.slider("Sample size", 20, 300, 60, step=10,
                         help="Larger samples take longer on the free server (roughly 1 minute per 100 tickets).")
    if s2.button("Run live on a random sample", type="primary", key="dash_btn"):
        with guard():
            sample = df.sample(n_sample).reset_index(drop=True)
            res = run_live(sample["text"].tolist(), f"{n_sample} tickets")
            live = sample.copy()
            for task in TASKS:
                live[f"live_{task}"] = [res[t][task][0] for t in sample["text"]]
            agree = {t: float((live[f"live_{t}"] == sample[f"pred_{t}"]).mean()) for t in TASKS}
            live["pred_category"] = live["live_category"]
            live["pred_urgency"] = live["live_urgency"]
            live["pred_sentiment"] = live["live_sentiment"]
            live["route_to"] = live["pred_category"].map(ROUTE)
            st.session_state["live_dash"] = {"df": live, "agree": agree, "n": n_sample}

    ld = st.session_state.get("live_dash")
    base, source_label = df, "stored results for all tickets"
    if ld:
        choice = st.radio("Data shown", [f"Live sample ({ld['n']} tickets)", "Stored results (all tickets)"],
                          horizontal=True)
        if choice.startswith("Live"):
            base, source_label = ld["df"], f"live predictions on a random sample of {ld['n']} tickets"
            a = ld["agree"]
            st.caption(f"Agreement between live and stored predictions on this sample: category "
                       f"{a['category']:.0%}, urgency {a['urgency']:.0%}, sentiment {a['sentiment']:.0%}.")
    st.markdown(f"**Showing:** {source_label}")

    f1, f2 = st.columns(2)
    all_langs = sorted(df["language"].unique())
    all_cats = sorted(df["pred_category"].unique())
    langs = f1.multiselect("Language", all_langs, default=all_langs)
    cats = f2.multiselect("Category", all_cats, default=all_cats)
    d = base[base["language"].isin(langs) & base["pred_category"].isin(cats)]

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
                                     color_discrete_map=URGENCY_COLORS,
                                     category_orders={"pred_urgency": ["High", "Medium", "Low"]},
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

# ---------------------------------------------------------------- Tab 4: search & export (live columns)
with tab4:
    kw = st.text_input("Search tickets (keyword in any language)")
    result = df[df["text"].str.contains(kw, case=False, na=False, regex=False)] if kw else df
    show = ["text", "language", "pred_category", "pred_sentiment", "pred_urgency", "route_to"]
    st.write(f"{len(result):,} tickets")
    st.dataframe(result[show], hide_index=True)
    st.download_button("Download report (CSV)", result[show].to_csv(index=False).encode("utf-8-sig"),
                       "supportsense_report.csv", "text/csv")

    st.divider()
    st.subheader("Live analysis of the matches")
    st.caption("The table above shows stored predictions. Run the models live on the first few matches "
               "to get fresh predictions you can download.")
    n_live = st.slider("Matches to analyze live", 5, 40, 10, key="search_n")
    if st.button("Analyze the first matches live", key="search_btn"):
        if result.empty:
            st.warning("No tickets match the search.")
        else:
            with guard():
                sub = result.head(n_live)
                res = run_live(sub["text"].tolist(), f"{len(sub)} tickets")
                out = sub[["text", "language"]].copy()
                out["live_category"] = [res[t]["category"][0] for t in sub["text"]]
                out["live_sentiment"] = [res[t]["sentiment"][0] for t in sub["text"]]
                out["live_urgency"] = [res[t]["urgency"][0] for t in sub["text"]]
                out["live_route_to"] = out["live_category"].map(ROUTE)
                st.session_state["live_search"] = {"kw": kw, "df": out}
    ls = st.session_state.get("live_search")
    if ls and ls["kw"] == kw:
        st.dataframe(ls["df"], hide_index=True)
        st.download_button("Download live results (CSV)", ls["df"].to_csv(index=False).encode("utf-8-sig"),
                           "supportsense_live_report.csv", "text/csv", key="live_csv")

# ---------------------------------------------------------------- Tab 5: emerging-issue alerts
with tab5:
    st.write("Flags weeks in which a complaint category was unusually busy compared with its own "
             "normal level (a simple z-score alert). Uses the stored results for all tickets.")
    weekly = df.assign(week=df["created_at"].dt.to_period("W").dt.start_time)
    wk = weekly.groupby(["week", "pred_category"]).size().unstack(fill_value=0)
    if len(wk) > 4:
        wk = wk.iloc[1:-1]                      # drop the partial first and last weeks
    thr = st.slider("Alert threshold (standard deviations above normal)", 1.5, 4.5, 3.0, step=0.5)
    mean, std = wk.mean(), wk.std().replace(0, np.nan)
    z = (wk - mean) / std
    alerts = [{"category": c, "week starting": w.date(), "tickets": int(wk.at[w, c]),
               "normal per week": round(float(mean[c]), 1), "z-score": round(float(z.at[w, c]), 1)}
              for c in wk.columns for w in wk.index if z.at[w, c] >= thr]
    a1, a2 = st.columns(2)
    a1.metric("Alerts found", len(alerts))
    a2.metric("Categories affected", len({a["category"] for a in alerts}))
    if alerts:
        st.dataframe(pd.DataFrame(alerts).sort_values("z-score", ascending=False), hide_index=True)
    else:
        st.success("No unusual weeks at this threshold.")
    pick_cat = st.selectbox("Show the weekly trend for", list(wk.columns), key="alert_cat")
    fig = px.line(wk.reset_index(), x="week", y=pick_cat, markers=True,
                  title=f"Weekly tickets: {pick_cat}")
    fig.add_hline(y=float(mean[pick_cat] + thr * (0 if np.isnan(std[pick_cat]) else std[pick_cat])),
                  line_dash="dash", annotation_text="alert level")
    st.plotly_chart(fig)
    st.caption("Ticket dates are simulated, with a built-in delivery-complaint spike in Oct-Nov, "
               "so the alert has something to find.")
