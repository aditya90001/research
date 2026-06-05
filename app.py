import streamlit as st
import time

from agent import build_graph

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# HERO UI (your design kept intact)
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0;">
    <h1 style="font-size:3rem;">🔬 ResearchMind</h1>
    <p style="color:gray;">LangGraph Multi-Agent Autonomous Research System</p>
</div>
<hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# INPUT UI
# ─────────────────────────────────────────────
topic = st.text_input("Enter Research Topic")

run = st.button("Run Research Pipeline 🚀")

# ─────────────────────────────────────────────
# RUN PIPELINE (LANGGRAPH ONLY)
# ─────────────────────────────────────────────
if run:
    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()

    with st.spinner("Running LangGraph Research Pipeline..."):

        app = build_graph()

        result = app.invoke({
            "topic": topic,
            "search_results": "",
            "scraped_content": "",
            "report": "",
            "feedback": ""
        })

        st.session_state.results = result

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
r = st.session_state.results

if r:

    st.markdown("## 🔍 Search Results")
    st.code(r.get("search_results", ""), language="text")

    st.markdown("## 📄 Scraped Content")
    st.code(r.get("scraped_content", ""), language="text")

    st.markdown("## 📝 Final Report")
    st.markdown(r.get("report", ""))

    st.download_button(
        label="⬇ Download Report",
        data=r.get("report", ""),
        file_name=f"report_{int(time.time())}.md",
        mime="text/markdown"
    )

    st.markdown("## 🧠 Critic Feedback")
    st.markdown(r.get("feedback", ""))