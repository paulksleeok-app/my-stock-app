"""모바일·브라우저 전용: 보유 포트폴리오 종목의 기관용 터미널 요약만 표시.

`app.py`와 동일한 지표·기간·이평 설정으로 `institutional_terminal_html`을 렌더합니다.
배포(Streamlit Cloud 등) 시 엔트리로 `mobile_app.py`를 지정할 수 있습니다.

로컬 실행: `streamlit run mobile_app.py`
"""

from __future__ import annotations

import datetime as dt
import html
from zoneinfo import ZoneInfo

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
        "PC 대시보드(`app.py`)와 같은 멀티팩터·이평·ATR 설정으로 보유 종목별 터미널만 보여 줍니다."
    )

    holdings = core.load_portfolio_holdings()
    if not holdings:
        st.info(
            "보유 종목이 없습니다. PC 앱에서 포트폴리오를 편집하거나, "
            "배포 환경에 `portfolio_holdings.json`을 두세요."
        )
        return

    end_d = _market_last_us_date()
    start_d = end_d - dt.timedelta(days=365)

    short_w, mid_w, long_w = 20, 50, 60
    rsi_w, atr_w = 14, 14
    bb_w, bb_k = 20, 2.0
    atr_stop_m, atr_take_m = 2.0, 3.0
    vol_f = True

    st.caption(f"기간: {start_d} ~ {end_d} · 이평 {short_w}/{mid_w}/{long_w} · RSI {rsi_w} · ATR {atr_w}")

    with st.spinner("포트폴리오 종목 데이터를 불러오는 중…"):
        items = sorted(holdings.items(), key=lambda x: x[0])
        for i, (tkr, qty) in enumerate(items):
            df, err = core.load_price_data(tkr, start_d, end_d)
            title = f"{tkr} · 보유 {qty}주"
            if df.empty:
                with st.expander(title, expanded=(i == 0)):
                    st.warning(err or "일봉 데이터를 가져오지 못했습니다.")
                continue

            df = core.trim_df_to_last_valid_close(df)
            if df.empty:
                with st.expander(title, expanded=(i == 0)):
                    st.warning("유효한 종가가 없습니다.")
                continue

            df = core.calculate_cross_signals(df, short_w, long_w)
            df[f"ma_{int(mid_w)}"] = df["close"].rolling(
                window=int(mid_w), min_periods=int(mid_w)
            ).mean()
            df = core.add_institutional_indicators(
                df,
                rsi_window=int(rsi_w),
                atr_window=int(atr_w),
                bb_window=int(bb_w),
                bb_k=float(bb_k),
            )
            inst_headline, inst_details = core.institutional_signal_summary(
                df,
                short_window=short_w,
                long_window=long_w,
                volume_filter=vol_f,
                atr_stop_mult=atr_stop_m,
                atr_take_mult=atr_take_m,
            )
            price_row, _, _ = core.last_valid_close_snapshot(df)
            last = price_row if price_row is not None else df.iloc[-1]

            term_html = core.institutional_terminal_html(
                tkr,
                last,
                short_window=int(short_w),
                mid_window=int(mid_w),
                long_window=int(long_w),
                rsi_window=int(rsi_w),
                atr_window=int(atr_w),
                atr_stop_mult=float(atr_stop_m),
                atr_take_mult=float(atr_take_m),
                inst_headline=inst_headline,
                inst_details=inst_details,
            )
            sig_bucket = core._composite_to_signal_bucket(inst_details.get("composite_100"))
            sig_first_d, sig_first_px = core.first_sara_pala_signal_date_price(df, sig_bucket)

            with st.expander(title, expanded=(i == 0)):
                if sig_bucket in ("사라", "팔라"):
                    d_s = html.escape(sig_first_d) if sig_first_d else "—"
                    p_s = html.escape(sig_first_px) if sig_first_px else "—"
                    b_s = html.escape(sig_bucket)
                    st.markdown(
                        f'<p class="sig-first-hint">«{b_s}» 최초 안내일: <strong>{d_s}</strong> · '
                        f'당시 종가: <strong>{p_s} USD</strong></p>',
                        unsafe_allow_html=True,
                    )
                st.markdown(term_html, unsafe_allow_html=True)


main()
