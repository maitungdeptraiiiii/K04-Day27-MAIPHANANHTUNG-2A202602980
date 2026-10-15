"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")

RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}


async def fetch_recent_sessions() -> list[dict]:
    """Same query as audit.replay.list_threads, but returns rows instead of printing.

    worst_risk ranks high > med > low explicitly — a plain MAX(risk_level) compares
    strings alphabetically and would rank 'med' above 'high'.
    """
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp)  AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*)        AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def load_session(thread_id: str) -> tuple[dict | None, dict | None]:
    """Restore (interrupt_payload, final) for a thread from the SQLite checkpointer."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value, None
    return None, (snapshot.values or None)


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    sessions = asyncio.run(fetch_recent_sessions())
    if not sessions:
        st.caption("No sessions yet — run a review to populate audit_events.")
    else:
        st.dataframe(
            [
                {
                    "thread_id": s["thread_id"][:8],
                    "pr": s["pr_url"].rsplit("/", 1)[-1],
                    "worst_risk": s["worst_risk"],
                    "last_event": s["last_event"][:19].replace("T", " "),
                }
                for s in sessions
            ],
            hide_index=True,
            width="stretch",
        )
        labels = {
            s["thread_id"]: f"{RISK_ICON.get(s['worst_risk'], '⚪')} {s['thread_id'][:8]} · "
                            f"PR #{s['pr_url'].rsplit('/', 1)[-1]}"
            for s in sessions
        }
        chosen = st.selectbox(
            "Open session", options=list(labels), format_func=labels.get,
            index=None, placeholder="Pick a thread…",
        )
        if chosen and chosen != st.session_state.thread_id:
            st.session_state.thread_id = chosen
            st.session_state.pr_url = next(s["pr_url"] for s in sessions if s["thread_id"] == chosen)
            st.session_state.interrupt_payload, st.session_state.final = asyncio.run(load_session(chosen))
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", width="stretch"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        st.markdown("**The agent needs answers to these questions before it can review:**")
        answers: dict[str, str] = {
            q: st.text_input(q, key=f"escalation_q_{i}")
            for i, q in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers", type="primary"):
            if not all(a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action") or "?"
    analysis = final.get("analysis")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        if analysis is not None:
            if action.startswith("auto"):
                st.subheader(f"Auto-approved — confidence {analysis.confidence:.0%}")
                st.caption("High confidence: the agent posted the review without a human in the loop.")
            st.markdown(analysis.summary)
        st.link_button("View comment on GitHub", st.session_state.pr_url)
    elif action == "commit_failed" or action.endswith("commit_failed"):
        st.error("Posting the comment to GitHub failed — check GITHUB_TOKEN scope.")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
