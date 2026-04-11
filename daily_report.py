import datetime as dt
import os
from zoneinfo import ZoneInfo

import smtplib
from email.message import EmailMessage

import pandas as pd

import app  # 같은 폴더의 app.py 재사용


def build_portfolio_daily_report(as_of: dt.date) -> str:
    """보유 종목 전체에 대한 일일 분석 리포트 텍스트 생성."""
    lines: list[str] = []
    lines.append(f"[일일 미국 주식 포트폴리오 리포트] 기준일: {as_of}")
    lines.append("")

    snap = app.build_portfolio_snapshot(as_of=as_of)
    if snap.empty:
        lines.append("포트폴리오 요약을 계산할 수 없습니다. (데이터 소스/티커 확인 필요)")
        return "\n".join(lines)

    total_usd = float(pd.to_numeric(snap["평가금액(USD)"], errors="coerce").fillna(0).sum())
    lines.append(f"- 포트폴리오 총 평가금액(USD): {total_usd:,.2f}")
    lines.append("")

    for _, row in snap.iterrows():
        ticker = str(row["티커"])
        qty = int(row["보유수량"])
        lines.append(f"===== {ticker} (보유 {qty}주) =====")

        # 개별 종목 상세 분석 (최근 6개월)
        start = as_of - dt.timedelta(days=180)
        df, _ = app.load_price_data(ticker, start, as_of)
        if df.empty:
            lines.append("  - 가격 데이터를 가져오지 못했습니다.")
            lines.append("")
            continue

        df = app.calculate_cross_signals(df, short_window=20, long_window=60)
        df = app.add_institutional_indicators(df)

        latest_signal_text, latest_date = app.get_latest_signal(df)
        inst_headline, inst_details = app.institutional_signal_summary(
            df,
            short_window=20,
            long_window=60,
            volume_filter=True,
            atr_stop_mult=2.0,
            atr_take_mult=3.0,
        )
        week_outlook_label, week_outlook_detail = app.one_week_outlook(
            df, short_window=20, long_window=60
        )
        week_price_label, week_price_detail = app.one_week_price_projection(df)
        vol_label, vol_detail = app.volume_change_summary(df)

        last = df.iloc[-1]
        close = float(last["close"])

        lines.append(f"- 현재가: {close:,.2f} USD")
        lines.append(f"- 최근 크로스/매매 의견: {latest_signal_text}")
        if latest_date is not None:
            lines.append(f"- 최근 크로스 일자: {latest_date.strftime('%Y-%m-%d')}")
        fac = inst_details.get("factors") or {}
        lines.append(
            f"- 퀀트 멀티팩터: {inst_headline} / 한줄: {inst_details.get('action', '')} "
            f"/ 종합점수(0~100): {inst_details.get('composite_100', '')}"
        )
        lines.append(
            f"  · 팩터(−1~1) 추세 {fac.get('trend', '')} · 모멘텀 {fac.get('momentum', '')} · "
            f"변동성 {fac.get('volatility', '')} · 거래량 {fac.get('volume', '')}"
        )
        lines.append(f"  · 거래량 신호: {inst_details.get('volume_quality', '')}")
        sl = inst_details.get("stop_loss")
        tp = inst_details.get("take_profit")
        rr = inst_details.get("rr_ratio")
        sl_s = f"{float(sl):,.2f}" if sl is not None and pd.notna(sl) else "N/A"
        tp_s = f"{float(tp):,.2f}" if tp is not None and pd.notna(tp) else "N/A"
        rr_s = f"{float(rr):.2f}:1" if rr is not None and pd.notna(rr) else "N/A"
        lines.append(f"  · ATR 손절/목표(롱): {sl_s} / {tp_s} USD, 손익비 R≈{rr_s}")
        lines.append(f"- 단기(1주일) 방향성 전망: {week_outlook_label}")
        lines.append(f"- 1주일 후 가격 예측치: {week_price_label}")
        lines.append(f"- 거래량 변화: {vol_label}")
        lines.append("")
        lines.append("[1주일 방향성 상세]")
        lines.append(week_outlook_detail)
        lines.append("")
        lines.append("[1주일 가격 예측 상세]")
        lines.append(week_price_detail)
        lines.append("")
        lines.append("[거래량 변화 상세]")
        lines.append(vol_detail)
        lines.append("")

    return "\n".join(lines)


def send_email_report(subject: str, body: str, to_email: str) -> None:
    """SMTP를 사용해 이메일 발송.

    환경변수에서 설정을 읽습니다:
    - STOCK_APP_SMTP_HOST
    - STOCK_APP_SMTP_PORT
    - STOCK_APP_SMTP_USER
    - STOCK_APP_SMTP_PASSWORD
    - STOCK_APP_FROM_EMAIL (없으면 USER를 From 으로 사용)
    """
    host = os.environ.get("STOCK_APP_SMTP_HOST", "")
    port = int(os.environ.get("STOCK_APP_SMTP_PORT", "587"))
    user = os.environ.get("STOCK_APP_SMTP_USER", "")
    password = os.environ.get("STOCK_APP_SMTP_PASSWORD", "")
    from_email = os.environ.get("STOCK_APP_FROM_EMAIL", user)

    if not (host and user and password and from_email):
        raise RuntimeError(
            "SMTP 환경변수가 설정되지 않았습니다. "
            "STOCK_APP_SMTP_HOST / STOCK_APP_SMTP_PORT / STOCK_APP_SMTP_USER / "
            "STOCK_APP_SMTP_PASSWORD / STOCK_APP_FROM_EMAIL 을 설정해주세요."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def main() -> None:
    # 기준 날짜: 미국 동부 시간 오늘 날짜
    now_kst = dt.datetime.now(ZoneInfo("Asia/Seoul"))
    now_us = now_kst.astimezone(ZoneInfo("America/New_York")).date()

    report_body = build_portfolio_daily_report(as_of=now_us)

    to_email = os.environ.get("STOCK_APP_REPORT_TO_EMAIL", "")
    if not to_email:
        raise RuntimeError(
            "받는 사람 이메일이 설정되지 않았습니다. "
            "환경변수 STOCK_APP_REPORT_TO_EMAIL 에 본인 이메일을 지정해주세요."
        )

    subject = f"[미국 주식 일일 리포트] {now_us}"
    send_email_report(subject, report_body, to_email=to_email)


if __name__ == "__main__":
    main()

