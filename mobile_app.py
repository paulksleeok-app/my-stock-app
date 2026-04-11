"""휴대폰·브라우저 전용 경량 요약.

차트 없이 숫자·문구만 표시해 데이터·렌더 비용을 줄였습니다. Streamlit Cloud 등
배포 URL만 있으면 Wi-Fi/셀룰러 등 인터넷 되는 곳 어디서나 같은 화면을 볼 수 있습니다.
로컬 PC 전용이 아닙니다 — `streamlit run mobile_app.py` 는 개발용입니다.
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


def _fmt_num(x, *, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except TypeError:
        pass
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    return f"{v:,.{digits}f}"


def main() -> None:
    # 반드시 첫 Streamlit 호출이어야 함. app은 이후에 지연 import (캐시·무거운 모듈 로드 분리).
    st.set_page_config(
        page_title="주식 요약 (모바일)",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    import app as core

    # 터치·가독성 + 불필요한 리소스 최소화 (차트 라이브러리 미사용)
    if st.session_state.get("_mobile_css") is not True:
        st.session_state["_mobile_css"] = True
        st.markdown(
            """
<style>
  html { -webkit-text-size-adjust: 100%; }
  .block-container {
    padding-top: max(0.75rem, env(safe-area-inset-top)) !important;
    padding-left: max(0.75rem, env(safe-area-inset-left)) !important;
    padding-right: max(0.75rem, env(safe-area-inset-right)) !important;
    padding-bottom: max(0.5rem, env(safe-area-inset-bottom)) !important;
    max-width: min(560px, 100%) !important;
  }
  h1 { font-size: 1.25rem !important; margin-bottom: 0.35rem !important; }
  h2, h3 { font-size: 1.05rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
  button[kind="header"] { min-height: 44px; }
  .mobile-section {
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
    margin: 0.4rem 0 0.75rem 0;
  }
  .mobile-row { font-size: 0.95rem; line-height: 1.45; padding: 0.2rem 0; border-bottom: 1px solid #eaecef; }
  .mobile-row:last-child { border-bottom: none; }
  .mobile-k { color: #57606a; }
  .mobile-v { font-weight: 600; color: #24292f; }
</style>
""",
            unsafe_allow_html=True,
        )

    st.title("📱 미국 주식 · 요약")
    st.caption(
        "인터넷만 연결된 브라우저에서 열면 됩니다. "
        "배포 주소(Streamlit Cloud 등)를 즐겨찾기해 두면 어디서든 동일하게 볼 수 있어요."
    )

    holdings = list(core.PORTFOLIO_HOLDINGS.keys())
    default_t = holdings[0] if holdings else "AAPL"

    with st.sidebar:
        st.caption("설정")
        if holdings:
            pick = st.selectbox("보유 종목", options=holdings, index=0)
        else:
            pick = "AAPL"
        ticker = st.text_input("티커", value=pick).strip().upper() or default_t

        m_last = _market_last_us_date()
        days_back = st.slider(
            "조회 일수 (짧을수록 빠름·데이터 적음)",
            min_value=60,
            max_value=365,
            value=90,
            step=10,
            help="이평·지표에 필요한 최소 길이만 넘기면 됩니다. 짧을수록 네트워크·계산이 가벼워집니다.",
        )
        short_w = st.number_input("단기 이평", 3, 120, 20)
        long_w = st.number_input("장기 이평", 10, 365, 60)
        if short_w >= long_w:
            st.warning("단기 이평은 장기보다 작아야 합니다.")
        st.caption("티커·기간·이평을 바꾸면 자동으로 다시 계산됩니다.")

    if short_w >= long_w:
        st.error("단기 이평 < 장기 이평으로 맞춰주세요.")
        return

    start_d = m_last - dt.timedelta(days=int(days_back))
    end_d = m_last

    with st.spinner("데이터 로드 중…"):
        df, err = core.load_price_data(ticker, start_d, end_d)

    if df.empty:
        st.error(err or "데이터를 가져오지 못했습니다.")
        return

    df = core.calculate_cross_signals(df, short_w, long_w)
    df = core.add_institutional_indicators(df)
    headline, inst = core.institutional_signal_summary(
        df,
        short_w,
        long_w,
        volume_filter=True,
        atr_stop_mult=2.0,
        atr_take_mult=3.0,
    )
    sig_text, sig_date = core.get_latest_signal(df)
    vol_label, _ = core.volume_change_summary(df)

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    close = float(last["close"]) if pd.notna(last.get("close")) else None
    chg_pct = None
    if prev is not None and pd.notna(prev.get("close")) and close and float(prev["close"]) != 0:
        chg_pct = (close / float(prev["close"]) - 1.0) * 100.0

    fac = inst.get("factors") or {}
    reasons = inst.get("reasons") or []
    vol_reasons = [r for r in reasons if "수급" in r or "거래량" in r][:5]
    if not vol_reasons:
        vol_reasons = [r for r in reasons if r][:4]

    # —— 기관형 수급(거래량·정합) —— #
    st.subheader("기관 수급 정보")
    vq = html.escape(str(inst.get("volume_quality") or "—"))
    v_score = html.escape(str(fac.get("volume", "")))
    vol_safe = html.escape(str(vol_label))
    st.markdown(
        f'<div class="mobile-section">'
        f'<div class="mobile-row"><span class="mobile-k">거래량 신호</span> '
        f'<span class="mobile-v">{vq}</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">수급 팩터 (−1~1)</span> '
        f'<span class="mobile-v">{v_score}</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">거래량 추이 요약</span> '
        f'<span class="mobile-v">{vol_safe}</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    if vol_reasons:
        st.caption("근거 (수급·거래량 관련)")
        for line in vol_reasons:
            st.write(f"· {line}")
    else:
        st.caption("수급 관련 상세 근거가 없습니다.")

    # —— 핵심 지표 —— #
    st.subheader("핵심 지표")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("종가 (USD)", _fmt_num(close, digits=2) if close is not None else "—")
    with c2:
        st.metric("전일 대비", f"{chg_pct:+.2f}%" if chg_pct is not None else "—")

    c3, c4 = st.columns(2)
    with c3:
        st.metric("종합 점수 (0~100)", str(inst.get("composite_100", "—")))
    with c4:
        h = str(headline)
        st.metric("퀀트 판단", h[:28] + "…" if len(h) > 28 else h)

    rsi = last.get("rsi")
    macd_h = last.get("macd_hist")
    atr_v = last.get("atr")
    st.markdown(
        f'<div class="mobile-section">'
        f'<div class="mobile-row"><span class="mobile-k">RSI</span> '
        f'<span class="mobile-v">{_fmt_num(rsi, digits=1)}</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">MACD 히스토그램</span> '
        f'<span class="mobile-v">{_fmt_num(macd_h, digits=4)}</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">ATR</span> '
        f'<span class="mobile-v">{_fmt_num(atr_v, digits=3)}</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">추세/모멘텀/변동성 팩터</span> '
        f'<span class="mobile-v">'
        f'{fac.get("trend", "—")} / {fac.get("momentum", "—")} / {fac.get("volatility", "—")}'
        f"</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    sl = inst.get("stop_loss")
    tp = inst.get("take_profit")
    rr = inst.get("rr_ratio")
    st.markdown(
        f'<div class="mobile-section">'
        f'<div class="mobile-row"><span class="mobile-k">ATR 손절가</span> '
        f'<span class="mobile-v">{_fmt_num(sl, digits=2)} USD</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">ATR 목표가</span> '
        f'<span class="mobile-v">{_fmt_num(tp, digits=2)} USD</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">손익비 (R)</span> '
        f'<span class="mobile-v">{_fmt_num(rr, digits=2)}</span></div>'
        f'<div class="mobile-row"><span class="mobile-k">한줄 의사결정</span> '
        f'<span class="mobile-v">{html.escape(str(inst.get("action", "—")))}</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"최근 크로스: {sig_text}" + (f" ({sig_date.date()})" if sig_date is not None else ""))
    st.caption(f"기간: {start_d} ~ {end_d} · 이평 {short_w}/{long_w}")


# Streamlit Cloud는 스크립트를 엔트리포인트로 실행합니다. __name__ 가드 없이 호출해 항상 부팅되게 합니다.
main()
