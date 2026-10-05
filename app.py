"""
Student Placement Predictor  -  Streamlit app
Decision Tree model (same pipeline as the notebook, with the branch-cleaning bug fixed).
Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "student_performance_dataset.csv"
BRANCHES = ["CIVIL", "COMMERCE", "CSE", "ECE", "IT", "MECHANICAL"]
NUMERIC = ["study_hours", "attendance", "sleep_hours", "internet_usage",
           "assignments_completed", "previous_score", "exam_score"]
NICE = {
    "gender": "Gender", "study_hours": "Study hours", "attendance": "Attendance",
    "sleep_hours": "Sleep hours", "internet_usage": "Internet usage",
    "assignments_completed": "Assignments done", "previous_score": "Previous score",
    "extracurricular": "Extracurricular", "exam_score": "Exam score",
}

# ----------------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------------
pio.templates.default = "plotly_dark"
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');
:root{
  --ink:#e8ecf4; --paper:#0b1020; --card:#161c2e; --line:#2a3350;
  --pass:#5fd39a; --pass-bg:#0f2a1f; --fail:#ff8a75; --fail-bg:#301613; --accent:#7fa2ff;
}
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif; color:var(--ink);}
.stApp {background:var(--paper);}
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:2rem; max-width:1150px;}
h1,h2,h3 {font-family:'Fraunces',serif !important; color:var(--ink); letter-spacing:-.01em;}
.hero {background:#1b2748; color:#fff; border-radius:18px; padding:30px 36px; margin-bottom:22px;
       border-bottom:6px solid #e0b84c;}
.hero h1 {color:#fff; margin:0 0 6px 0; font-size:2.2rem;}
.hero p {color:#cbd3e6; margin:0; font-size:1.02rem; max-width:640px;}
.panel {background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 22px;}
.verdict {border-radius:16px; padding:24px 26px; border:2px solid;}
.verdict.pass {background:var(--pass-bg); border-color:var(--pass); color:var(--pass);}
.verdict.fail {background:var(--fail-bg); border-color:var(--fail); color:var(--fail);}
.verdict .big {font-family:'Fraunces',serif; font-size:2.3rem; font-weight:700; line-height:1.1;}
.verdict .sub {color:var(--ink); opacity:.8; margin-top:6px;}
.rule {border-left:3px solid var(--accent); padding:4px 12px; margin:6px 0; background:#1a2340;
       border-radius:0 8px 8px 0; font-size:.95rem;}
.metric {background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px;}
.metric .v {font-family:'Fraunces',serif; font-size:1.7rem; font-weight:700;}
.metric .l {font-size:.85rem; opacity:.7;}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  background:#2b59c3; color:#fff; border:none; border-radius:10px; padding:.7rem 1rem;
  font-weight:600; width:100%;}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {background:#4a78e6; color:#fff;}
button[data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data + model (cached, runs once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the decision tree on your dataset...")
def build_model():
    df = pd.read_csv(DATA_PATH)
    d = df.drop_duplicates().dropna().drop(columns=["student_id"]).copy()

    # cleaning (same as notebook, but branch is now cleaned on the right dataframe)
    d["gender"] = (d["gender"].str.strip().str.lower()
                   .map({"male": 1, "female": 0, "m": 1, "f": 0}))
    d["branch"] = d["branch"].str.strip().str.upper()
    d["study_hours"] = d["study_hours"].clip(0, 12)
    d["sleep_hours"] = d["sleep_hours"].clip(3, 12)
    d["internet_usage"] = d["internet_usage"].clip(0, 12)
    d["extracurricular"] = d["extracurricular"].map({"No": 0, "Yes": 1})
    d["placement_status"] = d["placement_status"].map({"Not Placed": 0, "Placed": 1})
    d = pd.get_dummies(d, columns=["branch"], dtype=int)
    for b in BRANCHES:                      # make sure every branch column exists
        if f"branch_{b}" not in d.columns:
            d[f"branch_{b}"] = 0

    X = d.drop(columns=["placement_status"])
    X = X[sorted(X.columns)]
    y = d["placement_status"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [1, 2, 5, 10],
         "criterion": ["gini", "entropy"]},
        n_iter=10, cv=5, scoring="accuracy", random_state=42)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_

    pred = model.predict(X_te)
    metrics = {
        "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred), "F1-score": f1_score(y_te, pred),
    }
    return {
        "model": model, "columns": list(X.columns), "metrics": metrics,
        "cm": confusion_matrix(y_te, pred), "params": search.best_params_,
        "n_train": len(X_tr), "n_test": len(X_te), "clean": d,
    }


def make_row(inp, columns):
    row = {c: 0 for c in columns}
    row.update({
        "gender": 1 if inp["gender"] == "Male" else 0,
        "extracurricular": 1 if inp["extracurricular"] == "Yes" else 0,
        **{k: inp[k] for k in NUMERIC},
    })
    row[f"branch_{inp['branch']}"] = 1
    return pd.DataFrame([row])[columns]


def decision_path_rules(model, row, columns):
    """Turn the tree path for one student into plain-English rules."""
    t = model.tree_
    node_ids = model.decision_path(row).indices
    rules = []
    for n in node_ids:
        if t.children_left[n] == t.children_right[n]:      # leaf
            continue
        f, thr = columns[t.feature[n]], t.threshold[n]
        val = row.iloc[0][f]
        name = f.replace("branch_", "Branch: ").replace("_", " ")
        if f.startswith("branch_") or f in ("gender", "extracurricular"):
            rules.append(f"**{name}** {'is' if val > thr else 'is not'} selected")
        else:
            op = "≤" if val <= thr else ">"
            rules.append(f"**{name}** = {val:g}  ({op} {thr:.1f})")
    return rules


art = build_model()
model, cols = art["model"], art["columns"]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🎓 Student Placement Predictor</h1>
  <p>Enter a student's profile and a decision tree trained on 8,000+ student records
     estimates whether they are likely to be placed, and shows the exact rules behind the answer.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_insights, tab_model = st.tabs(["Predict", "Data insights", "Model report"])

# ----------------------------------------------------------------------------
# TAB 1: Predict
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("Student profile")
        with st.form("student_form"):
            c1, c2, c3 = st.columns(3)
            gender = c1.selectbox("Gender", ["Male", "Female"])
            branch = c2.selectbox("Branch", BRANCHES, index=2)
            extra = c3.selectbox("Extracurricular", ["Yes", "No"])

            a1, a2 = st.columns(2)
            study = a1.slider("Study hours / day", 0.0, 12.0, 3.5, 0.1)
            sleep = a2.slider("Sleep hours / day", 3.0, 12.0, 7.0, 0.1)
            net = a1.slider("Internet usage (hrs / day)", 0.0, 12.0, 4.5, 0.1)
            attend = a2.slider("Attendance (%)", 30.0, 100.0, 76.0, 0.5)
            assign = a1.slider("Assignments completed", 0, 30, 13)
            prev = a2.slider("Previous score", 0.0, 100.0, 62.0, 0.5)
            exam = st.slider("Latest exam score", 0.0, 100.0, 60.0, 0.5)

            go_btn = st.form_submit_button("Predict placement")

    inp = dict(gender=gender, branch=branch, extracurricular=extra, study_hours=study,
               sleep_hours=sleep, internet_usage=net, attendance=attend,
               assignments_completed=assign, previous_score=prev, exam_score=exam)

    with right:
        st.subheader("Result")
        if not go_btn and "last" not in st.session_state:
            st.markdown('<div class="panel">Fill in the profile and press '
                        '<b>Predict placement</b> to see the result here.</div>',
                        unsafe_allow_html=True)
        else:
            if go_btn:
                st.session_state["last"] = inp
            inp = st.session_state["last"]
            row = make_row(inp, cols)
            prob = float(model.predict_proba(row)[0][1])
            placed = prob >= 0.5

            cls, title = ("pass", "Likely to be Placed") if placed else ("fail", "Likely Not Placed")
            conf = prob if placed else 1 - prob
            st.markdown(f"""
            <div class="verdict {cls}">
              <div class="big">{title}</div>
              <div class="sub">Model confidence: <b>{conf*100:.0f}%</b></div>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%", "font": {"family": "Fraunces", "size": 38}},
                title={"text": "Probability of placement", "font": {"size": 14}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e8ecf4"},
                       "steps": [{"range": [0, 50], "color": "#3a1d19"},
                                 {"range": [50, 100], "color": "#14382a"}],
                       "threshold": {"line": {"color": "#ff8a75", "width": 3},
                                     "thickness": .8, "value": 50}}))
            fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

    # explanation row
    if "last" in st.session_state:
        row = make_row(st.session_state["last"], cols)
        e1, e2 = st.columns(2, gap="large")
        with e1:
            st.subheader("Why this answer?")
            st.caption("The questions the tree asked about this student, in order:")
            for r in decision_path_rules(model, row, cols):
                st.markdown(f'<div class="rule">{r}</div>', unsafe_allow_html=True)
        with e2:
            st.subheader("Student vs. placed students")
            clean = art["clean"]
            placed_avg = clean[clean.placement_status == 1][NUMERIC].mean()
            s = st.session_state["last"]
            labels = [NICE[k] for k in NUMERIC]
            fig2 = go.Figure()
            fig2.add_bar(name="This student", x=labels, y=[s[k] for k in NUMERIC], marker_color="#7fa2ff")
            fig2.add_bar(name="Avg. placed student", x=labels, y=[placed_avg[k] for k in NUMERIC],
                         marker_color="#e0b84c")
            fig2.update_layout(barmode="group", height=330, margin=dict(l=0, r=0, t=10, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig2, width="stretch")

            gaps = [(NICE[k], placed_avg[k] - s[k]) for k in
                    ["study_hours", "attendance", "assignments_completed", "previous_score", "exam_score"]
                    if placed_avg[k] - s[k] > 0]
            if gaps:
                gaps.sort(key=lambda x: -x[1])
                st.info("Biggest gaps vs. placed students: " +
                        ", ".join(f"**{n}** (−{g:.1f})" for n, g in gaps[:3]))

# ----------------------------------------------------------------------------
# TAB 2: Data insights
# ----------------------------------------------------------------------------
with tab_insights:
    clean = art["clean"].copy()
    clean["Status"] = clean["placement_status"].map({1: "Placed", 0: "Not Placed"})
    colors = {"Placed": "#3fbf85", "Not Placed": "#ff7a62"}

    i1, i2 = st.columns(2, gap="large")
    with i1:
        st.subheader("What matters most to the model")
        imp = (pd.Series(model.feature_importances_, index=cols)
               .rename(lambda c: c.replace("branch_", "Branch: ").replace("_", " ").capitalize())
               .sort_values().tail(8))
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#7fa2ff"))
        f.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(f, width="stretch")
    with i2:
        st.subheader("Placed vs. Not Placed")
        counts = clean["Status"].value_counts()
        f = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.55,
                             marker_colors=[colors[x] for x in counts.index]))
        f.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(f, width="stretch")

    st.subheader("Explore a feature")
    feat = st.selectbox("Pick a feature", NUMERIC, format_func=lambda k: NICE[k])
    f = go.Figure()
    for s_name in ["Placed", "Not Placed"]:
        f.add_trace(go.Box(y=clean.loc[clean.Status == s_name, feat], name=s_name,
                           marker_color=colors[s_name]))
    f.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f, width="stretch")

# ----------------------------------------------------------------------------
# TAB 3: Model report
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("How well does the model do on unseen students?")
    m = art["metrics"]
    mc = st.columns(4)
    for col, (k, v) in zip(mc, m.items()):
        col.markdown(f'<div class="metric"><div class="v">{v*100:.1f}%</div>'
                     f'<div class="l">{k}</div></div>', unsafe_allow_html=True)

    st.write("")
    r1, r2 = st.columns([1, 1], gap="large")
    with r1:
        st.markdown("**Confusion matrix** (test set)")
        cm = art["cm"]
        f = go.Figure(go.Heatmap(z=cm, x=["Pred: Not Placed", "Pred: Placed"],
                                 y=["Actual: Not Placed", "Actual: Placed"],
                                 text=cm, texttemplate="%{text}", colorscale="Blues",
                                 showscale=False))
        f.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                        yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(f, width="stretch")
    with r2:
        st.markdown("**Model details**")
        p = art["params"]
        st.markdown(f"""
- Algorithm: **Decision Tree Classifier**
- Best settings (RandomizedSearchCV, 5-fold): criterion = `{p['criterion']}`,
  max_depth = `{p['max_depth']}`, min_samples_leaf = `{p['min_samples_leaf']}`
- Training students: **{art['n_train']:,}**  |  Test students: **{art['n_test']:,}**
- Cleaning: duplicates and missing rows removed, gender and branch standardised,
  study / sleep / internet hours clipped to realistic ranges
- Encoding: branch one-hot encoded, gender and extracurricular as 0/1
        """)
        st.caption("This is a learning project. Predictions are statistical estimates, "
                   "not a guarantee of any real placement outcome.")
