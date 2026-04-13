"""모바일·브라우저 전용: 보유 포트폴리오 종목의 기관용 터미널 요약만 표시.

보유 목록·정렬·분석은 PC `app.py`의 `build_portfolio_snapshot`과 동일합니다.
배포(Streamlit Cloud 등) 시 엔트리로 `mobile_app.py`를 지정할 수 있습니다.

로컬 실행: `streamlit run mobile_app.py`
"""

from __future__ import annotations

import datetime as dt
import html
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def _market_last_us_date() -> dt.date:
    now_kst = dt.datetime.now(ZoneInfo("Asia/Seoul"))
    now_us = now_kst.astimezone(ZoneInfo("America/New_York")).date()
    if now_us.weekday() >= 5:
        days_back = now_us.weekday() - 4
        return now_us - dt.timedelta(days=days_back)
    return now_us


def main() -> None:
    st.set_page_config(
        page_title="포트폴리오 · 기관 터미널",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    import app as core

    base_font_px = 13
    if st.session_state.get("_mobile_term_css") is not True:
        st.session_state["_mobile_term_css"] = True
        st.markdown(
            f"""
<style>
  html {{ -webkit-text-size-adjust: 100%; }}
  .block-container {{
    padding-top: max(0.75rem, env(safe-area-inset-top)) !important;
    padding-left: max(0.75rem, env(safe-area-inset-left)) !important;
    padding-right: max(0.75rem, env(safe-area-inset-right)) !important;
    padding-bottom: max(0.5rem, env(safe-area-inset-bottom)) !important;
    max-width: min(560px, 100%) !important;
  }}
  h1 {{ font-size: 1.2rem !important; margin-bottom: 0.35rem !important; }}
  h2, h3 {{ font-size: 1.02rem !important; }}
  .quant-terminal {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
    padding: 0.75rem 0.85rem;
    border-radius: 6px;
    border: 1px solid #30363d;
    margin: 0.35rem 0 0.6rem 0;
    font-size: {base_font_px}px !important;
    line-height: 1.42 !important;
  }}
  .quant-terminal .qt-row {{ margin: 0.18rem 0; }}
  .quant-terminal .qt-k {{ color: #8b949e; }}
  .quant-terminal .qt-v {{ color: #58a6ff; font-weight: 600; }}
  .quant-terminal .qt-ok {{ color: #3fb950; }}
  .quant-terminal .qt-warn {{ color: #f85149; }}
  .quant-terminal .qt-muted {{ color: #6e7681; font-size: 0.92em; }}
  .quant-terminal .qt-section {{
    color: #d29922;
    font-weight: 700;
    margin-top: 0.45rem;
    margin-bottom: 0.12rem;
    padding-bottom: 0.12rem;
    border-bottom: 1px solid #30363d;
    letter-spacing: 0.02em;
  }}
  .sig-first-hint {{
    font-size: 0.88rem;
    color: #57606a;
    margin: 0 0 0.5rem 0;
    line-height: 1.4;
  }}
</style>
""",
            unsafe_allow_html=True,
        )

    st.title("📱 포트폴리오 · 기관 터미널")
    st.caption(
        "PC 대시보드 「내 포트폴리오 요약」과 **동일 보유·동일 정렬·동일 분석**으로 터미널만 보여 줍니다."
    )

    # PC(`app.py`)가 저장하는 `portfolio_holdings.json`을 매 실행마다 읽음 — 세션에 묶이지 않음
    holdings = core.load_portfolio_holdings()
    if not holdings:
        st.info(
            "보유 종목이 없습니다. PC 앱에서 포트폴리오를 편집하거나, "
            "배포 환경에 `portfolio_holdings.json`을 두세요."
        )
        return

    end_d = _market_last_us_date()
    st.caption(f"기준일(미국 최근 거래일): **{end_d}** · 평가금액 높은 순 = PC 스냅샷과 동일")

    with st.spinner("포트폴리오 종목 데이터를 불러오는 중…"):
        snap = core.build_portfolio_snapshot(as_of=end_d, holdings=holdings)
    if snap.empty:
        st.info("포트폴리오 요약을 계산할 수 없습니다. 티커·데이터 소스를 확인해 주세요.")
        return

    rows: list[dict] = []
    with st.spinner("종목별 터미널을 구성하는 중…"):
        for _, row in snap.iterrows():
            tkr = str(row["티커"])
            qty = int(row["보유수량"])
            v_eval = row["평가금액(USD)"]
            if v_eval == "" or pd.isna(v_eval):
                rows.append(
                    {
                        "ok": False,
                        "tkr": tkr,
                        "qty": qty,
                        "err": str(row.get("한줄 의사결정") or "일봉 데이터를 가져오지 못했습니다."),
                    }
                )
                continue

            block = core.mobile_portfolio_expander_content(tkr, qty, end_d)
            if not block.get("ok"):
                rows.append(
                    {
                        "ok": False,
                        "tkr": tkr,
                        "qty": qty,
                        "err": str(block.get("err") or "분석 실패"),
                    }
                )
                continue

            rows.append(
                {
                    "ok": True,
                    "tkr": block["tkr"],
                    "qty": block["qty"],
                    "title": block["title"],
                    "term_html": block["term_html"],
                    "sig_bucket": block["sig_bucket"],
                    "sig_first_d": block["sig_first_d"],
                    "sig_first_px": block["sig_first_px"],
                }
            )

    for i, r in enumerate(rows):
        if not r["ok"]:
            title = f"{r['tkr']} · 보유 {r['qty']}주"
            with st.expander(title, expanded=(i == 0)):
                st.warning(r["err"])
            continue

        with st.expander(r["title"], expanded=(i == 0)):
            if r["sig_bucket"] in ("사라", "팔라"):
                d_s = html.escape(r["sig_first_d"]) if r["sig_first_d"] else "—"
                p_s = html.escape(r["sig_first_px"]) if r["sig_first_px"] else "—"
                b_s = html.escape(r["sig_bucket"])
                st.markdown(
                    f'<p class="sig-first-hint">«{b_s}» 신호최초일: <strong>{d_s}</strong> · '
                    f'신호당시종가(USD): <strong>{p_s}</strong></p>',
                    unsafe_allow_html=True,
                )
            st.markdown(r["term_html"], unsafe_allow_html=True)


main()
