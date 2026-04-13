from __future__ import annotations

import datetime as dt
import json
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import io
import math

import requests

# yfinance·HTTP는 동시 호출 시 간헐 예외가 나기 쉬워 직렬화
_price_fetch_lock = threading.Lock()


def _normalize_ohlc_df(data: pd.DataFrame) -> pd.DataFrame | None:
    if data is None or data.empty:
        return None
    out = data.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower() for c in out.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(out.columns)):
        return None
    out.index = pd.to_datetime(out.index)
    if "volume" not in out.columns:
        out["volume"] = pd.NA
    return out


def _load_yfinance(ticker: str, start: dt.date, end: dt.date) -> Tuple[pd.DataFrame, str | None]:
    import yfinance as yf

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end + dt.timedelta(days=1),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as e:
        return pd.DataFrame(), str(e)
    norm = _normalize_ohlc_df(raw)
    if norm is None or norm.empty:
        return pd.DataFrame(), "빈 응답 또는 OHLC 컬럼 없음"
    return norm, None


def _load_stooq_csv(ticker: str, start: dt.date, end: dt.date) -> Tuple[pd.DataFrame, str | None]:
    """Stooq 일봉 CSV (pandas_datareader 없이, Python 3.12 호환)."""
    base = ticker.strip().upper()
    if not base:
        return pd.DataFrame(), "티커가 비어 있습니다."
    stooq_sym = f"{base}.US" if "." not in base else base
    url = f"https://stooq.com/q/d/l/?s={stooq_sym.lower()}&i=d"
    try:
        resp = requests.get(
            url,
            timeout=18,
            headers={"User-Agent": "stock-app/1.0", "Accept": "text/csv"},
        )
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        return pd.DataFrame(), str(e)

    required_cols = {"Date", "Open", "High", "Low", "Close"}
    if not required_cols.issubset(set(raw.columns)):
        return pd.DataFrame(), f"Stooq 컬럼 이상: {list(raw.columns)}"

    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    raw = raw.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if raw.empty:
        return pd.DataFrame(), "선택 기간에 Stooq 데이터 없음"

    out = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "volume" not in out.columns:
        out["volume"] = pd.NA
    return out, None


def _load_price_data_impl(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> Tuple[pd.DataFrame, str | None]:
    """지정 기간 일봉: yfinance 우선, 실패·빈 데이터 시 Stooq CSV 폴백 (캐시·스레드용 코어)."""
    msgs: list[str] = []

    with _price_fetch_lock:
        df_yf, err_yf = _load_yfinance(ticker, start, end)
        if not df_yf.empty:
            return df_yf, None
        if err_yf:
            msgs.append(f"yfinance: {err_yf}")

        df_s, err_s = _load_stooq_csv(ticker, start, end)
        if not df_s.empty:
            return df_s, None
        if err_s:
            msgs.append(f"Stooq: {err_s}")

    detail = "\n".join(msgs) if msgs else "알 수 없음"
    return (
        pd.DataFrame(),
        f"{ticker} 데이터를 가져오지 못했습니다.\n{detail}",
    )


def last_valid_close_snapshot(
    df: pd.DataFrame,
) -> tuple[pd.Series | None, float | None, float | None]:
    """가장 최근의 유효한 종가 행과 가격(전일 종가 포함).

    장중·미완성 봉 등으로 마지막 행의 close가 NaN인 경우, 직전 확정 종가를 사용한다.
    """
    if df is None or df.empty or "close" not in df.columns:
        return None, None, None
    s = pd.to_numeric(df["close"], errors="coerce").dropna()
    if s.empty:
        return None, None, None
    row = df.loc[s.index[-1]]
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else None
    return row, last, prev


def trim_df_to_last_valid_close(df: pd.DataFrame) -> pd.DataFrame:
    """미확정(종가 NaN) 말단 행을 제거해 분석 시 마지막 행에 항상 유효 종가가 오게 함."""
    if df.empty or "close" not in df.columns:
        return df
    mask = pd.to_numeric(df["close"], errors="coerce").notna()
    if not mask.any():
        return df.iloc[0:0].copy()
    pos = int(mask.to_numpy().nonzero()[0][-1])
    return df.iloc[: pos + 1].copy()


def cross_event_date_label(d: object) -> str:
    """크로스 일자 셀용(인덱스 타입이 달라도 strftime 예외 방지)."""
    if d is None:
        return ""
    try:
        ts = pd.Timestamp(d)
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        return ""
    if pd.isna(ts):
        return ""
    try:
        return ts.strftime("%Y-%m-%d")
    except (AttributeError, OSError, ValueError):
        return ""


@st.cache_data(ttl=600, max_entries=256, show_spinner=False)
def _load_price_data_cached(ticker: str, start_iso: str, end_iso: str) -> tuple:
    """메인 분석 티커용 캐시(동일 티커·기간 재조회 시 네트워크 생략)."""
    s = dt.date.fromisoformat(start_iso)
    e = dt.date.fromisoformat(end_iso)
    df, err = _load_price_data_impl(ticker, s, e)
    if df is None or df.empty:
        return df, err
    return df.copy(), err


def load_price_data(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> Tuple[pd.DataFrame, str | None]:
    """일봉 로드(캐시). 포트폴리오 병렬 로드는 `load_price_data_parallel` 사용."""
    t = ticker.strip().upper()
    return _load_price_data_cached(t, start.isoformat(), end.isoformat())


def load_price_data_parallel(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> Tuple[pd.DataFrame, str | None]:
    """스레드·다종목용(캐시 비적용 — Streamlit 캐시는 메인 스레드 전용)."""
    return _load_price_data_impl(ticker.strip().upper(), start, end)


def add_moving_averages(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """단기/장기 이동평균선 컬럼 추가."""
    df = df.copy()
    df[f"ma_{short_window}"] = df["close"].rolling(window=short_window).mean()
    df[f"ma_{long_window}"] = df["close"].rolling(window=long_window).mean()
    return df


def add_ichimoku_columns(df: pd.DataFrame) -> pd.DataFrame:
    """일목균형표: 전환·기준·선행스팬 A/B (선행은 26일 시프트)."""
    d = df.copy()
    high = d["high"]
    low = d["low"]
    d["ichi_tenkan"] = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2
    d["ichi_kijun"] = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2
    d["ichi_senkou_a"] = ((d["ichi_tenkan"] + d["ichi_kijun"]) / 2).shift(26)
    d["ichi_senkou_b"] = (
        (high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2
    ).shift(26)
    return d


def add_institutional_indicators(
    df: pd.DataFrame,
    rsi_window: int = 14,
    atr_window: int = 14,
    bb_window: int = 20,
    bb_k: float = 2.0,
) -> pd.DataFrame:
    df = df.copy()

    # RSI (Wilder)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR (Wilder)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_window, adjust=False, min_periods=atr_window).mean()

    # Bollinger Bands
    bb_mid = df["close"].rolling(window=bb_window, min_periods=bb_window).mean()
    bb_std = df["close"].rolling(window=bb_window, min_periods=bb_window).std(ddof=0)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + bb_k * bb_std
    df["bb_lower"] = bb_mid - bb_k * bb_std

    # Volume filters (optional; may be missing)
    if "volume" in df.columns:
        df["vol_sma20"] = df["volume"].rolling(window=20, min_periods=20).mean()
    else:
        df["volume"] = pd.NA
        df["vol_sma20"] = pd.NA

    # 20-day breakout levels
    df["hh20"] = df["high"].rolling(window=20, min_periods=20).max()
    df["ll20"] = df["low"].rolling(window=20, min_periods=20).min()
    return add_ichimoku_columns(df)


def _clampf(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _weighted_linear_regression(x, y, weights) -> tuple[float, float]:
    """가중 최소제곱: y ≈ slope * x + intercept."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    W = np.diag(w)
    X = np.column_stack([x, np.ones_like(x)])
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _forecast_precompute(df: pd.DataFrame, lookback_days: int = 20) -> dict | None:
    """WLS 로그추세·변동성 성분 1회 계산(다중 예측일 재사용)."""
    import numpy as np

    if df is None or df.empty:
        return None
    closes = df["close"].dropna()
    if closes.shape[0] < 10:
        return None
    closes = closes.tail(int(lookback_days))
    if closes.shape[0] < 10:
        return None

    atr_series = df["atr"].reindex(closes.index) if "atr" in df.columns else None
    y = np.log(np.asarray(closes.values, dtype=float))
    n = len(y)
    x = np.arange(n, dtype=float)
    w = np.exp(2.2 * x / max(n - 1, 1.0))
    if n >= 5:
        w[-5:] *= 1.85
    if n >= 3:
        w[-3:] *= 1.25
    w = w * (n / np.sum(w))
    slope, intercept = _weighted_linear_regression(x, y, w)
    log_ret = np.diff(y)
    vol_log = float(log_ret.std()) if log_ret.size > 0 else 0.0
    last_close = float(closes.iloc[-1])
    last_atr = atr_series.iloc[-1] if atr_series is not None else pd.NA
    atr_pct = float(last_atr) / last_close if pd.notna(last_atr) and last_close > 0 else 0.0
    atr_ratio = 1.0
    if atr_series is not None and atr_series.dropna().shape[0] >= 20:
        atr_ma = float(atr_series.tail(20).mean())
        if atr_ma > 0 and pd.notna(last_atr):
            atr_ratio = float(last_atr) / atr_ma
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "vol_log": vol_log,
        "atr_pct": atr_pct,
        "atr_ratio": atr_ratio,
    }


def _forecast_at_horizon(
    pre: dict,
    days_ahead: int,
    *,
    z_score: float = 1.96,
) -> tuple[str, str, dict]:
    import numpy as np

    if days_ahead <= 0:
        return "예측 불가", "days_ahead는 1 이상이어야 합니다.", {}
    n = int(pre["n"])
    slope = float(pre["slope"])
    intercept = float(pre["intercept"])
    vol_log = float(pre["vol_log"])
    atr_pct = float(pre["atr_pct"])
    atr_ratio = float(pre["atr_ratio"])

    future_x = n - 1 + int(days_ahead)
    future_log_price = slope * future_x + intercept
    future_price = float(np.exp(future_log_price))
    sigma_path = vol_log * float(np.sqrt(days_ahead))
    sigma_atr = atr_pct * float(np.sqrt(days_ahead))
    sigma_combined = float(np.hypot(sigma_path, sigma_atr * 0.95))
    if atr_ratio > 1.0:
        sigma_combined *= float(1.0 + 0.35 * min(atr_ratio - 1.0, 2.0))

    z = float(z_score)
    low_price = float(np.exp(future_log_price - z * sigma_combined))
    high_price = float(np.exp(future_log_price + z * sigma_combined))

    label = f"약 {future_price:,.2f} USD"
    detail = (
        f"- 추정 방식: 최근 {n}거래일 로그종가 **가중(WMA형) 선형추세** "
        f"(최근 3~5일 가중 강화) + **ATR·역사적 변동성** 혼합 불확실성\n"
        f"- {days_ahead}거래일 후 예상 가격(중심): 약 {future_price:,.2f} USD\n"
        f"- **확률적 구간(근사 {z:.2f}σ)**: {low_price:,.2f} ~ {high_price:,.2f} USD\n"
        f"- 참고: σ_path≈{sigma_path:.5f}, σ_ATR≈{sigma_atr:.5f}(로그스케일 합성), ATR/20일비≈{atr_ratio:.2f}\n"
        "※ 참고용 추정이며 실제 가격을 보장하지 않습니다."
    )
    meta = {
        "center": future_price,
        "low": low_price,
        "high": high_price,
        "sigma_combined": sigma_combined,
        "atr_ratio": atr_ratio,
        "slope_log_per_day": slope,
    }
    return label, detail, meta


def multi_horizon_price_labels(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10),
    lookback_days: int = 20,
) -> dict[int, str]:
    """여러 거래일 예측 라벨을 회귀 1회로 계산."""
    pre = _forecast_precompute(df, lookback_days)
    if pre is None:
        return {h: "예측 불가" for h in horizons}
    out: dict[int, str] = {}
    for h in horizons:
        lab, _, _ = _forecast_at_horizon(pre, h, z_score=1.96)
        out[h] = lab
    return out


def _log_price_forecast_weighted_atr(
    df: pd.DataFrame,
    days_ahead: int,
    lookback_days: int = 20,
    *,
    z_score: float = 1.96,
) -> tuple[str, str, dict]:
    """로그 종가에 대해 최근일 가중(WMA 스타일) 선형 추세 + ATR·로그변동성 혼합 불확실성 구간."""
    if days_ahead <= 0:
        return "예측 불가", "days_ahead는 1 이상이어야 합니다.", {}
    pre = _forecast_precompute(df, lookback_days)
    if pre is None:
        return "예측 불가", "데이터가 부족해 가격 예측 범위를 계산할 수 없습니다.", {}
    return _forecast_at_horizon(pre, days_ahead, z_score=z_score)


def compute_atr_risk_levels(
    last: pd.Series,
    atr_stop_mult: float = 2.0,
    atr_take_mult: float = 3.0,
) -> dict:
    """현재가·ATR 기준 손절가·목표가(롱 관점) 및 손익비."""
    out: dict = {"stop_loss": pd.NA, "take_profit": pd.NA, "rr_ratio": pd.NA}
    close = last.get("close")
    atr = last.get("atr")
    if pd.isna(close) or pd.isna(atr) or float(atr) <= 0:
        return out
    c, a = float(close), float(atr)
    sl = c - atr_stop_mult * a
    tp = c + atr_take_mult * a
    risk = c - sl
    reward = tp - c
    out["stop_loss"] = sl
    out["take_profit"] = tp
    if risk > 0:
        out["rr_ratio"] = reward / risk
    return out


def quant_multi_factor_analysis(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    *,
    atr_stop_mult: float = 2.0,
    atr_take_mult: float = 3.0,
    volume_filter: bool = True,
    weights: dict[str, float] | None = None,
) -> dict:
    """멀티팩터 점수(0~100): 추세·모멘텀·변동성·거래량 가중 결합 + ATR 손절/목표."""
    empty = {
        "composite_100": pd.NA,
        "factors": {},
        "weights": {},
        "headline": "데이터 없음",
        "action": "관망 (뚜렷한 우위 없음)",
        "reasons": [],
        "stop_loss": pd.NA,
        "take_profit": pd.NA,
        "rr_ratio": pd.NA,
        "volume_quality": "—",
    }
    if df.empty:
        return empty

    w = weights or {"trend": 0.35, "momentum": 0.30, "volatility": 0.20, "volume": 0.15}
    w = {k: float(v) for k, v in w.items()}
    if not volume_filter:
        v0 = w.get("volume", 0.15)
        w["trend"] = w.get("trend", 0.35) + v0
        w["volume"] = 0.0
    ssum = sum(w.values()) or 1.0
    w = {k: v / ssum for k, v in w.items()}

    last = df.iloc[-1]
    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"
    reasons: list[str] = []

    ma_sc = 0.0
    if pd.notna(last.get(short_col)) and pd.notna(last.get(long_col)):
        if last[short_col] > last[long_col]:
            ma_sc = 1.0
            reasons.append("추세(이평): 단기 MA > 장기 MA")
        elif last[short_col] < last[long_col]:
            ma_sc = -1.0
            reasons.append("추세(이평): 단기 MA < 장기 MA")

    sa, sb = last.get("ichi_senkou_a"), last.get("ichi_senkou_b")
    tk, kj = last.get("ichi_tenkan"), last.get("ichi_kijun")
    close = last.get("close")

    cloud_sc = 0.0
    if pd.notna(sa) and pd.notna(sb) and pd.notna(close):
        top = max(float(sa), float(sb))
        bot = min(float(sa), float(sb))
        c = float(close)
        if c > top:
            cloud_sc = 1.0
            reasons.append("추세(구름): 종가가 구름 위")
        elif c < bot:
            cloud_sc = -1.0
            reasons.append("추세(구름): 종가가 구름 아래")
        else:
            reasons.append("추세(구름): 종가가 구름대 안")

    tk_sc = 0.0
    if pd.notna(tk) and pd.notna(kj):
        if float(tk) > float(kj):
            tk_sc = 1.0
            reasons.append("추세(일목): 전환선 > 기준선")
        elif float(tk) < float(kj):
            tk_sc = -1.0
            reasons.append("추세(일목): 전환선 < 기준선")

    trend_score = _clampf((ma_sc + cloud_sc + tk_sc) / 3.0)

    ma_bull = (
        pd.notna(last.get(short_col))
        and pd.notna(last.get(long_col))
        and last[short_col] > last[long_col]
    )
    ret_1d = pd.NA
    vol_ratio = pd.NA
    if len(df) >= 2 and pd.notna(last.get("close")) and pd.notna(df.iloc[-2].get("close")):
        ret_1d = float(last["close"]) / float(df.iloc[-2]["close"]) - 1.0
    if (
        pd.notna(last.get("volume"))
        and pd.notna(last.get("vol_sma20"))
        and float(last["vol_sma20"]) > 0
    ):
        vol_ratio = float(last["volume"]) / float(last["vol_sma20"])
    trend_vol_confirm = (
        bool(ma_bull)
        and pd.notna(ret_1d)
        and float(ret_1d) > 0
        and pd.notna(vol_ratio)
        and float(vol_ratio) >= 1.2
    )

    rsi = last.get("rsi")
    rsi_sc = 0.0
    if pd.notna(rsi):
        rsi_f = float(rsi)
        div = 40.0 if (trend_vol_confirm and rsi_f > 55) else 25.0
        rsi_sc = _clampf((rsi_f - 50.0) / div)
        if trend_vol_confirm and pd.notna(vol_ratio) and float(vol_ratio) >= 1.5 and rsi_f >= 58:
            rsi_sc = min(1.0, rsi_sc + 0.22)
            reasons.append(
                f"모멘텀(RSI): {rsi_f:.1f} (상승+거래량 동반 → 추세 전개로 해석, 과열 페널티 완화)"
            )
        else:
            reasons.append(f"모멘텀(RSI): {rsi_f:.1f}")

    macd_hist = last.get("macd_hist")
    macd_sc = 0.0
    if pd.notna(macd_hist):
        tail = df["macd_hist"].dropna().tail(60)
        std = float(tail.std()) if len(tail) > 1 else 0.0
        if std > 0:
            macd_sc = _clampf(float(macd_hist) / (2.0 * std))
        else:
            macd_sc = 1.0 if float(macd_hist) > 0 else (-1.0 if float(macd_hist) < 0 else 0.0)
        reasons.append(f"모멘텀(MACD 히스토그램): {float(macd_hist):.5f}")

    momentum_score = _clampf(0.5 * rsi_sc + 0.5 * macd_sc)

    vol_score = 0.0
    if (
        pd.notna(last.get("bb_upper"))
        and pd.notna(last.get("bb_lower"))
        and pd.notna(close)
    ):
        u, lo = float(last["bb_upper"]), float(last["bb_lower"])
        if u > lo:
            pct_b = (float(close) - lo) / (u - lo)
            pct_b = _clampf(pct_b, 0.0, 1.0)
            vol_score = _clampf(2.0 * (pct_b - 0.5))
            reasons.append(f"변동성(BB %B): 약 {pct_b * 100:.0f}%")
            if trend_vol_confirm:
                # 상단 밴드 근처를 '일시 과열'만이 아니라 추세·수급이 맞을 때 가속으로 부분 반영
                vol_score = _clampf(vol_score * 0.42 + 0.48)
                reasons.append("변동성(BB): 이평 추세+상승+거래량 정합 → 밴드 위치를 추세 쪽으로 보정")
        else:
            reasons.append("변동성(BB): 밴드 폭 없음")

    volume_score = 0.0
    vol_label = "거래량 데이터 없음"
    if len(df) >= 2:
        prev = df.iloc[-2]
        prc = float(last["close"]) if pd.notna(last.get("close")) else None
        pprc = float(prev["close"]) if pd.notna(prev.get("close")) else None
        vol = last.get("volume")
        vsma = last.get("vol_sma20")
        if (
            prc is not None
            and pprc is not None
            and pd.notna(vol)
            and pd.notna(vsma)
            and float(vsma) > 0
        ):
            ret = prc / pprc - 1.0
            vr = float(vol) / float(vsma)
            if ret > 0 and vr >= 1.2:
                volume_score = 1.0
                if ma_bull and vr >= 1.5:
                    volume_score = min(1.0, volume_score + 0.12)
                    vol_label = "상승 + 거래량 급증 — 추세 전개 신호"
                    reasons.append("수급: 거래량 동반 상승을 추세 시작으로 가중")
                else:
                    vol_label = "상승 + 거래량 확인(≥20일평균 1.2배)"
                    reasons.append("수급: 상승에 거래량 동반")
            elif ret > 0 and vr < 0.85:
                volume_score = 0.2
                vol_label = "상승, 거래량 약함"
                reasons.append("수급: 상승이나 거래량 동반 약함")
            elif ret < 0 and vr >= 1.2:
                volume_score = -1.0
                vol_label = "하락 + 거래량 급증"
                reasons.append("수급: 하락에 거래량 동반")
            elif ret < 0:
                volume_score = -0.35
                vol_label = "하락 구간"
            else:
                volume_score = 0.0
                vol_label = "보합·거래량 보통"
        elif prc is not None and pprc is not None:
            ret = prc / pprc - 1.0
            vol_label = "거래량 SMA 없음"
            volume_score = 0.1 if ret > 0 else (-0.1 if ret < 0 else 0.0)

    volume_score = _clampf(volume_score)

    raw = (
        w["trend"] * trend_score
        + w["momentum"] * momentum_score
        + w["volatility"] * vol_score
        + w["volume"] * volume_score
    )
    composite = 50.0 + 50.0 * _clampf(raw)

    if composite >= 65:
        headline = "퀀트 멀티팩터: 매수·보유 우위"
        action = "사라 (매수/보유 우위)"
    elif composite <= 35:
        headline = "퀀트 멀티팩터: 매도·회피 우위"
        action = "팔라 (매도 또는 회피 우위)"
    else:
        headline = "퀀트 멀티팩터: 중립·관망"
        action = "관망 (뚜렷한 우위 없음)"

    risk = compute_atr_risk_levels(last, atr_stop_mult, atr_take_mult)

    return {
        "composite_100": round(composite, 1),
        "factors": {
            "trend": round(trend_score, 3),
            "momentum": round(momentum_score, 3),
            "volatility": round(vol_score, 3),
            "volume": round(volume_score, 3),
        },
        "weights": w,
        "headline": headline,
        "action": action,
        "reasons": reasons,
        "stop_loss": risk["stop_loss"],
        "take_profit": risk["take_profit"],
        "rr_ratio": risk["rr_ratio"],
        "volume_quality": vol_label,
    }


def institutional_signal_summary(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    volume_filter: bool = True,
    atr_stop_mult: float = 2.0,
    atr_take_mult: float = 3.0,
) -> Tuple[str, dict]:
    """멀티팩터(추세·모멘텀·변동성·거래량) 가중 점수 + ATR 손절/목표 요약."""
    if df.empty:
        return "데이터 없음", {}

    q = quant_multi_factor_analysis(
        df,
        short_window,
        long_window,
        atr_stop_mult=atr_stop_mult,
        atr_take_mult=atr_take_mult,
        volume_filter=volume_filter,
    )
    details = {
        "score": q.get("composite_100"),
        "composite_100": q.get("composite_100"),
        "factors": q.get("factors"),
        "weights": q.get("weights"),
        "reasons": q.get("reasons", []),
        "action": q.get("action"),
        "stop_loss": q.get("stop_loss"),
        "take_profit": q.get("take_profit"),
        "rr_ratio": q.get("rr_ratio"),
        "volume_quality": q.get("volume_quality"),
    }
    return str(q.get("headline", "—")), details


def backtest_ma_atr_strategy(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    atr_stop_mult: float = 2.0,
    atr_take_mult: float = 3.0,
    fee_bps: float = 2.0,
) -> pd.DataFrame:
    """단순 기관 스타일: MA 추세 + ATR 손절/익절 + 수수료(bps) 포함 백테스트.

    - 진입: 단기 MA > 장기 MA 이면 다음날 시가(근사: 당일 종가) 기준 롱 보유
    - 청산: 단기 MA < 장기 MA, 또는 ATR 기반 손절/익절
    """
    d = df.copy()
    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"

    d["trend_long"] = (d[short_col] > d[long_col]).astype("Int64")

    pos = 0
    entry = pd.NA
    stop = pd.NA
    take = pd.NA
    equity = 1.0
    curve = []

    fee = fee_bps / 10000.0

    for idx, row in d.iterrows():
        price = row.get("close")
        atr = row.get("atr")
        trend = row.get("trend_long")

        if pd.isna(price):
            curve.append((idx, equity, pos, entry, stop, take, 0, pd.NA))
            continue

        trade = 0  # +1 buy, -1 sell

        if pos == 0:
            if trend == 1 and pd.notna(atr):
                pos = 1
                entry = float(price)
                stop = float(entry - atr_stop_mult * float(atr))
                take = float(entry + atr_take_mult * float(atr))
                trade = 1
                equity *= (1 - fee)
        else:
            exit_now = False
            if trend == 0:
                exit_now = True
            if pd.notna(stop) and float(price) <= float(stop):
                exit_now = True
            if pd.notna(take) and float(price) >= float(take):
                exit_now = True

            if exit_now:
                ret = float(price) / float(entry) - 1
                equity *= (1 + ret)
                equity *= (1 - fee)
                pos = 0
                entry = pd.NA
                stop = pd.NA
                take = pd.NA
                trade = -1
            else:
                # mark-to-market equity curve (hold)
                if pd.notna(entry) and float(entry) != 0:
                    ret = float(price) / float(entry) - 1
                    # equity not compounding daily here; approximate by revaluing from last equity at entry
                    # keep equity as-is; still show open PnL separately
        open_pnl = pd.NA
        if pos == 1 and pd.notna(entry) and float(entry) != 0:
            open_pnl = float(price) / float(entry) - 1
        curve.append((idx, equity, pos, entry, stop, take, trade, open_pnl))

    res = pd.DataFrame(
        curve,
        columns=["date", "equity", "position", "entry", "stop", "take", "trade", "open_pnl"],
    ).set_index("date")
    res["equity_curve"] = res["equity"].ffill()
    return res


def performance_summary(equity_curve: pd.Series) -> dict:
    ec = equity_curve.dropna()
    if ec.empty:
        return {}
    total_return = ec.iloc[-1] / ec.iloc[0] - 1
    daily_ret = ec.pct_change().dropna()
    vol = daily_ret.std() * (252**0.5) if not daily_ret.empty else pd.NA
    sharpe = (daily_ret.mean() / daily_ret.std()) * (252**0.5) if (not daily_ret.empty and daily_ret.std() != 0) else pd.NA
    running_max = ec.cummax()
    dd = ec / running_max - 1
    max_dd = dd.min()
    return {
        "total_return": total_return,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def one_week_outlook(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> Tuple[str, str]:
    """단기(향후 1주일) 전망 요약.

    - 최근 20거래일 추세(단기/장기 MA)
    - 최근 5거래일 수익률
    - RSI/MACD 조합

    을 기준으로 '상승/보합/하락'과 간단 설명을 만들어준다.
    실제 예측이 아닌, 현재 시점의 기술적 상황 요약임.
    """
    if df.empty:
        return "전망 불가", "데이터가 부족해 1주일 전망을 계산할 수 없습니다."

    tail = df.tail(20).copy()
    if tail.empty:
        return "전망 불가", "최근 데이터가 부족해 1주일 전망을 계산할 수 없습니다."

    # 기본 정보
    last = tail.iloc[-1]
    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"

    # 최근 5거래일 수익률
    if len(tail) >= 6:
        recent = tail["close"].iloc[-6:]
        week_ret = recent.iloc[-1] / recent.iloc[0] - 1
    else:
        week_ret = 0.0

    trend_up = pd.notna(last.get(short_col)) and pd.notna(last.get(long_col)) and last[short_col] > last[long_col]
    trend_down = pd.notna(last.get(short_col)) and pd.notna(last.get(long_col)) and last[short_col] < last[long_col]

    rsi = last.get("rsi")
    macd_hist = last.get("macd_hist")

    bullish_points = 0
    bearish_points = 0

    if trend_up:
        bullish_points += 2
    if trend_down:
        bearish_points += 2

    if pd.notna(week_ret):
        if week_ret > 0.03:
            bullish_points += 1
        elif week_ret < -0.03:
            bearish_points += 1

    if pd.notna(rsi):
        if rsi < 35:
            bullish_points += 1
        elif rsi > 65:
            bearish_points += 1

    if pd.notna(macd_hist):
        if macd_hist > 0:
            bullish_points += 1
        elif macd_hist < 0:
            bearish_points += 1

    if bullish_points - bearish_points >= 2:
        label = "단기 상승 우위"
        reason = "향후 1주일은 기술적으로 상승 우위 가능성이 조금 더 높아 보입니다."
    elif bearish_points - bullish_points >= 2:
        label = "단기 하락/조정 우위"
        reason = "향후 1주일은 하락 또는 조정 국면일 가능성이 조금 더 높아 보입니다."
    else:
        label = "단기 보합/혼조"
        reason = "상승/하락 신호가 뒤섞여 있어, 향후 1주일은 뚜렷한 방향성보다 박스권·혼조 가능성이 커 보입니다."

    rsi_str = f"{float(rsi):.1f}" if pd.notna(rsi) else "N/A"
    macd_sign = (
        "+"
        if pd.notna(macd_hist) and macd_hist > 0
        else "-"
        if pd.notna(macd_hist) and macd_hist < 0
        else "N/A"
    )

    detail = (
        f"- 최근 5거래일 수익률: {week_ret*100:.1f}%\n"
        f"- 추세(단기/장기 MA): {'상승' if trend_up else '하락' if trend_down else '중립'}\n"
        f"- RSI: {rsi_str}, MACD 히스토그램: {macd_sign}\n"
        f"{reason}\n"
        "※ 실제 수익을 보장하는 예측이 아니라, 현재 기술적 지표를 바탕으로 한 참고용 의견입니다."
    )
    return label, detail


def one_week_price_projection(df: pd.DataFrame) -> Tuple[str, str]:
    """1주(5거래일) 후 가격: 최근일 가중(WMA형) 로그추세 + ATR·σ 혼합 불확실성 구간."""
    if df.empty or df["close"].dropna().shape[0] < 10:
        return "예측 불가", "데이터가 부족해 1주일 후 가격 예측 범위를 계산할 수 없습니다."

    label, detail, meta = _log_price_forecast_weighted_atr(df, days_ahead=5, lookback_days=20)
    if meta:
        c = float(meta["center"])
        label = f"약 {c:,.2f} USD (중심값)"
    return label, detail


def price_projection(
    df: pd.DataFrame,
    days_ahead: int,
    lookback_days: int = 20,
) -> Tuple[str, str, dict]:
    """N거래일 후 가격: 최근일 가중(WMA형) 로그추세 + ATR·σ 혼합 불확실성 구간."""
    if days_ahead <= 0:
        return "예측 불가", "days_ahead는 1 이상이어야 합니다.", {}
    label, detail, meta = _log_price_forecast_weighted_atr(
        df,
        days_ahead=int(days_ahead),
        lookback_days=int(lookback_days),
    )
    return label, detail, meta


def volume_change_summary(df: pd.DataFrame) -> Tuple[str, str]:
    """거래량 변화 요약 (최근 vs 과거 평균)."""
    if df.empty or "volume" not in df.columns:
        return "정보 부족", "거래량 데이터가 없어 변화 추이를 계산할 수 없습니다."

    vol = df["volume"].dropna().tail(40)
    if vol.shape[0] < 10:
        return "정보 부족", "최근 거래량 데이터가 부족해 변화 추이를 계산할 수 없습니다."

    last_vol = float(vol.iloc[-1])
    recent_5 = float(vol.tail(5).mean())
    prev_5 = float(vol.iloc[:-5].tail(5).mean()) if vol.shape[0] >= 10 else recent_5
    avg_20 = float(vol.tail(20).mean())

    ratio_last_to_20 = last_vol / avg_20 if avg_20 > 0 else 0.0
    ratio_recent_to_prev = recent_5 / prev_5 if prev_5 > 0 else 1.0

    if ratio_last_to_20 >= 1.5:
        label = "최근 거래량 급증"
    elif ratio_last_to_20 <= 0.7:
        label = "최근 거래량 위축"
    else:
        label = "최근 거래량 보통 수준"

    detail = (
        f"- 어제/최근 거래일 거래량: {last_vol:,.0f}\n"
        f"- 최근 5일 평균 거래량: {recent_5:,.0f}\n"
        f"- 이전 5일 평균 거래량: {prev_5:,.0f}\n"
        f"- 최근 20일 평균 대비 비율: {ratio_last_to_20*100:.1f}%\n"
        f"- 최근 5일 vs 이전 5일 비율: {ratio_recent_to_prev*100:.1f}%\n"
        "※ 단순 거래량 통계로, 수급의 강·약을 정밀하게 측정하는 건 아닙니다."
    )
    return label, detail


def _composite_to_signal_bucket(composite: object) -> str:
    """멀티팩터 종합점수 → institutional_signal_summary 와 동일한 구간."""
    try:
        c = float(composite)
    except (TypeError, ValueError):
        return "관망"
    if pd.isna(c):
        return "관망"
    if c >= 65:
        return "사라"
    if c <= 35:
        return "팔라"
    return "관망"


def signal_bucket_from_action_line(action: str | None) -> str:
    """한줄 의사결정 문구 기준 사라 / 팔라 / 관망.

    신호최초일·당시 종가는 이 구간과 동일한 의미로 스캔한다(PC·모바일 공통).
    """
    if not action:
        return "관망"
    a = str(action)
    if "사라" in a:
        return "사라"
    if "팔라" in a:
        return "팔라"
    return "관망"


def first_sara_pala_signal_date_price(
    df: pd.DataFrame,
    bucket: str,
    *,
    short_window: int = 20,
    long_window: int = 60,
) -> tuple[str, str]:
    """조회 일봉 구간에서 사라 또는 팔라 신호가 처음 뜬 거래일·종가(당시 시점 기준).

    관망이거나 데이터가 부족하면 빈 문자열.
    동일 end 인덱스의 quant 재계산은 lru_cache로 막아 가볍게 함.
    """
    if bucket not in ("사라", "팔라") or df is None or df.empty:
        return "", ""

    min_len = max(long_window + 5, 66)
    if len(df) < min_len:
        return "", ""

    min_i = min_len - 1
    n = len(df)

    @lru_cache(maxsize=None)
    def _bucket_at_end(end_idx: int) -> str | None:
        if end_idx < min_i or end_idx >= n:
            return None
        sub = df.iloc[: end_idx + 1]
        q = quant_multi_factor_analysis(
            sub,
            short_window,
            long_window,
            volume_filter=True,
            atr_stop_mult=2.0,
            atr_take_mult=3.0,
        )
        c = q.get("composite_100")
        if c is None or pd.isna(c):
            return None
        try:
            cf = float(c)
        except (TypeError, ValueError):
            return None
        return _composite_to_signal_bucket(cf)

    def _fmt_first(end_idx: int) -> tuple[str, str]:
        sub = df.iloc[: end_idx + 1]
        row = sub.iloc[-1]
        first_date = cross_event_date_label(row.name)
        px = row.get("close")
        first_px = f"{float(px):,.2f}" if px is not None and pd.notna(px) else ""
        return first_date, first_px

    # 거친 간격(3일)으로 후보를 찾고, 없으면 전 구간 선형 스캔(기존과 동일 결과).
    stride = 3
    end_c: int | None = None
    for end in range(min_i, n, stride):
        b = _bucket_at_end(end)
        if b == bucket:
            end_c = end
            break
    if end_c is None:
        for end in range(min_i, n):
            b = _bucket_at_end(end)
            if b == bucket:
                end_c = end
                break
    if end_c is None:
        return "", ""

    e = end_c
    while e > min_i:
        b_prev = _bucket_at_end(e - 1)
        if b_prev == bucket:
            e -= 1
        else:
            break
    return _fmt_first(e)


# 사용자 보유 포트폴리오: 최초·파일 없을 때 기본값 (실제 목록은 portfolio_holdings.json)
DEFAULT_PORTFOLIO_HOLDINGS: dict[str, int] = {
    "ARKB": 11151,
    "CRM": 50,
    "VST": 25,
    "AVGO": 66,
    "IONQ": 600,
    "GOOGL": 15,
    "NVDA": 250,
    "TSLA": 120,
    "MSFT": 63,
}


def _portfolio_holdings_path() -> Path:
    return Path(__file__).resolve().parent / "portfolio_holdings.json"


def normalize_portfolio_ticker(raw: str) -> str:
    """티커 정규화(대문자·공백 제거). 유효하지 않으면 빈 문자열."""
    t = raw.strip().upper()
    if not t or len(t) > 16:
        return ""
    for c in t:
        if not (c.isalnum() or c in ".-^"):
            return ""
    return t


def load_portfolio_holdings() -> dict[str, int]:
    """로컬 JSON에서 보유 목록 로드. 없거나 오류 시 기본값."""
    path = _portfolio_holdings_path()
    if not path.is_file():
        return dict(DEFAULT_PORTFOLIO_HOLDINGS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, int] = {}
        for k, v in raw.items():
            t = normalize_portfolio_ticker(str(k))
            if not t:
                continue
            try:
                q = int(v)
            except (TypeError, ValueError):
                continue
            if q < 0:
                continue
            out[t] = q
        return out if out else dict(DEFAULT_PORTFOLIO_HOLDINGS)
    except Exception:
        return dict(DEFAULT_PORTFOLIO_HOLDINGS)


def save_portfolio_holdings(holdings: dict[str, int]) -> None:
    """보유 목록을 앱 폴더의 portfolio_holdings.json 에 저장."""
    clean: dict[str, int] = {}
    for k, v in holdings.items():
        t = normalize_portfolio_ticker(str(k))
        if not t:
            continue
        try:
            q = int(v)
        except (TypeError, ValueError):
            continue
        if q < 0:
            continue
        clean[t] = q
    path = _portfolio_holdings_path()
    path.write_text(
        json.dumps(dict(sorted(clean.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _portfolio_error_row(tkr: str, qty: int, msg: str) -> dict:
    """데이터 부족·오류 시에도 티커 행을 유지해 보유 전체 목록이 끊기지 않게 함."""
    return {
        "티커": tkr,
        "보유수량": qty,
        "현재가(USD)": "",
        "오늘변동(%)": "",
        "평가금액(USD)": "",
        "_전일평가금액(USD)": "",
        "멀티팩터(0~100)": "",
        "추세": "",
        "모멘텀": "",
        "변동성": "",
        "거래량점수": "",
        "거래량신호": "",
        "손절가(ATR)": "",
        "목표가(ATR)": "",
        "한줄 의사결정": msg,
        "신호최초일": "",
        "신호당시종가(USD)": "",
        "퀀트판단": "",
        "최근 크로스/매매 의견": "",
        "최근 크로스 일자": "",
    }


@st.cache_data(ttl=600, max_entries=512, show_spinner=False)
def _cached_portfolio_unit_analysis(tkr: str, as_of_iso: str) -> dict:
    """수량과 무관한 종목×기준일 분석(캐시). 성공 시 _base_close·_base_prev 포함, 실패 시 _err."""
    try:
        as_of = dt.date.fromisoformat(as_of_iso)
        start = as_of - dt.timedelta(days=180)
        df, _ = load_price_data_parallel(tkr, start, as_of)
        if df.empty:
            return {"_err": "일봉 데이터 없음 (티커·종료일·데이터 소스 확인)"}

        df = trim_df_to_last_valid_close(df)
        if df.empty:
            return {"_err": "일봉 데이터 없음 (티커·종료일·데이터 소스 확인)"}

        df = calculate_cross_signals(df, 20, 60)
        df = add_institutional_indicators(df)
        headline, details = institutional_signal_summary(
            df, 20, 60, volume_filter=True, atr_stop_mult=2.0, atr_take_mult=3.0
        )
        _row, close, prev_close_px = last_valid_close_snapshot(df)
        if close is None:
            return {"_err": "유효한 종가가 없습니다."}

        today_change_pct = pd.NA
        if prev_close_px is not None and prev_close_px != 0:
            today_change_pct = (close / prev_close_px - 1) * 100

        latest_cross_text, latest_cross_date = get_latest_signal(df)

        sl = details.get("stop_loss")
        tp = details.get("take_profit")
        fac = details.get("factors") or {}
        action = details.get("action") or ""
        sig_bucket = signal_bucket_from_action_line(action)
        first_sig_date, first_sig_px = first_sara_pala_signal_date_price(df, sig_bucket)

        return {
            "_base_close": float(close),
            "_base_prev": float(prev_close_px) if prev_close_px is not None else None,
            "현재가(USD)": round(float(close), 2),
            "오늘변동(%)": round(float(today_change_pct), 2) if pd.notna(today_change_pct) else "",
            "멀티팩터(0~100)": details.get("composite_100", ""),
            "추세": fac.get("trend", ""),
            "모멘텀": fac.get("momentum", ""),
            "변동성": fac.get("volatility", ""),
            "거래량점수": fac.get("volume", ""),
            "거래량신호": details.get("volume_quality", ""),
            "손절가(ATR)": round(float(sl), 2) if pd.notna(sl) else "",
            "목표가(ATR)": round(float(tp), 2) if pd.notna(tp) else "",
            "한줄 의사결정": action,
            "신호최초일": first_sig_date,
            "신호당시종가(USD)": first_sig_px,
            "퀀트판단": headline,
            "최근 크로스/매매 의견": latest_cross_text,
            "최근 크로스 일자": cross_event_date_label(latest_cross_date),
        }
    except Exception:
        return {"_err": ""}


def _portfolio_row_for_ticker(tkr: str, qty: int, as_of: dt.date) -> dict:
    """단일 보유 종목 행(병렬 워커용 — 네트워크는 load_price_data_parallel)."""
    nt = normalize_portfolio_ticker(tkr) or tkr.strip().upper()
    u = _cached_portfolio_unit_analysis(nt, as_of.isoformat())
    if "_err" in u:
        err = str(u.get("_err", ""))
        if err == "":
            return _portfolio_error_row(tkr, qty, "")
        return _portfolio_error_row(tkr, qty, err)

    bc = float(u.pop("_base_close"))
    bp = u.pop("_base_prev", None)
    prev_value_usd = bp * qty if bp is not None else pd.NA
    value_usd = bc * qty

    return {
        "티커": tkr,
        "보유수량": qty,
        **u,
        "평가금액(USD)": int(round(value_usd)) if pd.notna(value_usd) else "",
        "_전일평가금액(USD)": round(float(prev_value_usd), 2) if pd.notna(prev_value_usd) else "",
    }


def _holdings_cache_key(holdings: dict[str, int]) -> str:
    return json.dumps(sorted(holdings.items()), ensure_ascii=False)


def _holdings_from_cache_key(holdings_key: str) -> dict[str, int]:
    pairs = json.loads(holdings_key)
    return {str(k): int(v) for k, v in pairs}


@st.cache_data(ttl=600, max_entries=48, show_spinner=False)
def _cached_portfolio_snapshot_df(as_of_iso: str, holdings_key: str) -> pd.DataFrame:
    """동일 기준일·보유 목록에 대한 포트폴리오 표 전체 캐시."""
    as_of = dt.date.fromisoformat(as_of_iso)
    h = _holdings_from_cache_key(holdings_key)
    items = list(h.items())
    rows: list[dict] = []
    if not items:
        return pd.DataFrame()
    max_w = min(6, len(items))
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futures = {ex.submit(_portfolio_row_for_ticker, t, q, as_of): (t, q) for t, q in items}
        for fut in as_completed(futures):
            tkr, qty = futures[fut]
            try:
                r = fut.result()
                rows.append(r if r is not None else _portfolio_error_row(tkr, qty, ""))
            except Exception:
                rows.append(_portfolio_error_row(tkr, qty, ""))

    if not rows:
        return pd.DataFrame()

    df_snap = pd.DataFrame(rows)
    sort_key = pd.to_numeric(df_snap["평가금액(USD)"], errors="coerce")
    return (
        df_snap.assign(_pf_sort=sort_key)
        .sort_values(by="_pf_sort", ascending=False, na_position="last")
        .drop(columns=["_pf_sort"])
        .reset_index(drop=True)
    )


def build_portfolio_snapshot(
    as_of: dt.date,
    holdings: dict[str, int] | None = None,
) -> pd.DataFrame:
    """보유 종목 기준 요약(티커별 네트워크 병렬). 실패 종목도 행으로 남겨 전체 보유가 보이게 함."""
    h = holdings if holdings is not None else load_portfolio_holdings()
    return _cached_portfolio_snapshot_df(as_of.isoformat(), _holdings_cache_key(h))


def calculate_cross_signals(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """골든/데드 크로스 시점 계산."""
    df = add_moving_averages(df, short_window, long_window)

    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"

    # 신호 기준: 단기 > 장기 (1), 단기 < 장기 (-1)
    df["signal"] = 0
    df.loc[df[short_col] > df[long_col], "signal"] = 1
    df.loc[df[short_col] < df[long_col], "signal"] = -1

    # 신호 변화 지점에서 크로스 발생
    df["signal_shift"] = df["signal"].shift(1)
    df["cross"] = ""

    golden_mask = (df["signal_shift"] == -1) & (df["signal"] == 1)
    dead_mask = (df["signal_shift"] == 1) & (df["signal"] == -1)

    df.loc[golden_mask, "cross"] = "Golden Cross"
    df.loc[dead_mask, "cross"] = "Dead Cross"

    return df


def get_latest_signal(df: pd.DataFrame) -> Tuple[str, pd.Timestamp | None]:
    """가장 최근 골든/데드 크로스와 매매 의견 반환."""
    cross_df = df[df["cross"] != ""]
    if cross_df.empty:
        return "신호 없음", None

    latest = cross_df.iloc[-1]
    cross_type = latest["cross"]
    raw_date = latest.name
    date: pd.Timestamp | None = None
    if raw_date is not None:
        try:
            ts = pd.Timestamp(raw_date)
            date = ts if pd.notna(ts) else None
        except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
            date = None

    if cross_type == "Golden Cross":
        decision = "매수(또는 보유) 우위"
    elif cross_type == "Dead Cross":
        decision = "매도(또는 관망) 우위"
    else:
        decision = "신호 없음"

    return f"{cross_type} - {decision}", date


def cross_projection_summary(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    lookback_days: int = 60,
) -> Tuple[str, str]:
    """현재 크로스 상태와, 단순 추세 기반 예상 다음 크로스 시점 요약.

    - 최근 lookback_days 동안의 (단기MA - 장기MA) 추세를 선형으로 근사
    - 직선이 0을 지나는 미래 시점이 있으면, 그 날짜를 예상 크로스 날짜로 제시
    """
    if df.empty:
        return "크로스 정보 없음", "데이터가 부족해 크로스 상태를 계산할 수 없습니다."

    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"

    if short_col not in df.columns or long_col not in df.columns:
        return "크로스 정보 없음", "이동평균 자료가 없어 크로스 상태를 계산할 수 없습니다."

    d = df[[short_col, long_col]].dropna().copy()
    if d.empty:
        return "크로스 정보 없음", "유효한 이동평균 데이터가 없어 크로스를 계산할 수 없습니다."

    d = d.tail(lookback_days)
    if d.shape[0] < 5:
        return "크로스 정보 부족", "최근 데이터가 부족해 향후 크로스 예상 시점을 추정하기 어렵습니다."

    diff = d[short_col] - d[long_col]
    last_diff = float(diff.iloc[-1])

    # 현재 상태
    if last_diff > 0:
        current_status = "현재: 단기 MA가 장기 MA 위(상승 쪽 우위, Golden Cross 이후 상태)"
    elif last_diff < 0:
        current_status = "현재: 단기 MA가 장기 MA 아래(하락 쪽 우위, Dead Cross 이후 상태)"
    else:
        current_status = "현재: 단기·장기 MA가 거의 같은 수준(크로스 직전/직후)"

    # 향후 크로스 예상 (단순 선형회귀)
    import numpy as np

    y = diff.values
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    if slope == 0:
        detail = current_status + "\n\n최근 MA 차이가 거의 일정해, 단순 추세 기준으론 크로스 예상 시점을 잡기 어렵습니다."
        return "향후 크로스 예상 어려움", detail

    # diff = 0 되는 x
    root_x = -intercept / slope

    last_x = len(y) - 1
    if root_x <= last_x:
        detail = current_status + "\n\n단순 선형 추세 상, 최근 데이터를 기준으로 미래에 뚜렷한 교차 지점을 찾기 어렵습니다."
        return "향후 크로스 예상 불명확", detail

    days_ahead = int(round(root_x - last_x))
    if days_ahead <= 0 or days_ahead > 120:
        detail = current_status + "\n\n단순 추세로 계산한 크로스 시점이 너무 멀거나 불안정해, 신뢰하기 어렵습니다."
        return "향후 크로스 예상 불안정", detail

    last_date = d.index[-1]
    try:
        est_date = last_date + pd.tseries.offsets.BDay(days_ahead)
    except Exception:
        est_date = last_date + pd.Timedelta(days=days_ahead)

    # 어떤 방향의 크로스인지
    if last_diff > 0 and slope < 0:
        next_type = "Dead Cross(하향 교차)"
    elif last_diff < 0 and slope > 0:
        next_type = "Golden Cross(상향 교차)"
    else:
        next_type = "방향 불명확한 교차"

    headline = f"향후 크로스 예상: {est_date.date()} 전후 {next_type} 가능성"
    detail = (
        current_status
        + f"\n\n단순 추세(최근 {d.shape[0]}거래일 기준)로 보면, "
        f"{days_ahead}거래일 뒤인 {est_date.date()} 전후에 "
        f"{next_type}가 발생할 가능성이 있습니다.\n"
        "※ 이동평균 추세를 선형으로 단순 근사한 값으로, 실제 시장 상황에 따라 크게 달라질 수 있습니다."
    )
    return headline, detail


def plot_price_and_ma(
    df: pd.DataFrame,
    ticker: str,
    short_window: int,
    long_window: int,
):
    import plotly.graph_objects as go

    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[short_col],
            mode="lines",
            name=f"MA {short_window}",
            line=dict(color="orange"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[long_col],
            mode="lines",
            name=f"MA {long_window}",
            line=dict(color="blue"),
        )
    )

    # 크로스 지점 표시
    golden_df = df[df["cross"] == "Golden Cross"]
    dead_df = df[df["cross"] == "Dead Cross"]

    fig.add_trace(
        go.Scatter(
            x=golden_df.index,
            y=golden_df["close"],
            mode="markers",
            name="Golden Cross",
            marker=dict(color="green", size=10, symbol="triangle-up"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dead_df.index,
            y=dead_df["close"],
            mode="markers",
            name="Dead Cross",
            marker=dict(color="red", size=10, symbol="triangle-down"),
        )
    )

    fig.update_layout(
        title=f"{ticker} 가격 및 이동평균선",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def _price_vs_ma_text(close: float, ma: object) -> str:
    if isinstance(close, float) and math.isnan(close):
        return "—"
    if pd.isna(ma):
        return "—"
    m = float(ma)
    if m == 0:
        return "—"
    pct = (close / m - 1.0) * 100.0
    if close > m * 1.001:
        return f"종가>MA (+{pct:.1f}%)"
    if close < m * 0.999:
        return f"종가<MA ({pct:.1f}%)"
    return f"종가≈MA ({pct:+.1f}%)"


def _cloud_position_text(last: pd.Series) -> str:
    sa, sb = last.get("ichi_senkou_a"), last.get("ichi_senkou_b")
    c = last.get("close")
    if pd.isna(sa) or pd.isna(sb) or pd.isna(c):
        return "구름·종가 데이터 부족"
    top = max(float(sa), float(sb))
    bot = min(float(sa), float(sb))
    fc = float(c)
    if fc > top:
        return "종가가 구름 위 (장기 강세권)"
    if fc < bot:
        return "종가가 구름 아래 (장기 약세권)"
    return "종가가 구름대 안 (중립·전환 구간)"


def _macd_state_line(last: pd.Series) -> tuple[str, str]:
    m = last.get("macd")
    sig = last.get("macd_signal")
    h = last.get("macd_hist")
    if pd.isna(m) or pd.isna(sig):
        return "MACD 데이터 부족", "—"
    mf, sf = float(m), float(sig)
    cross = "MACD>시그널 (모멘텀 강세)" if mf > sf else "MACD<시그널 (모멘텀 약세)"
    if pd.notna(h):
        hf = float(h)
        cross += f" · 히스트 {hf:+.5f}"
    return cross, f"MACD {mf:.4f} / Sig {sf:.4f}"


def _day_change_pct_html(pct: float | None) -> str:
    """전일 종가 대비 등락률(%) HTML 조각."""
    if pct is None:
        return '<span class="qt-muted">—</span>'
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return '<span class="qt-muted">—</span>'
    cls = "qt-ok" if v > 0 else ("qt-warn" if v < 0 else "qt-muted")
    return f'<span class="{cls}">{v:+.2f}%</span>'


def institutional_terminal_html(
    ticker: str,
    last: pd.Series,
    *,
    short_window: int,
    mid_window: int,
    long_window: int,
    rsi_window: int,
    atr_window: int,
    atr_stop_mult: float,
    atr_take_mult: float,
    inst_headline: str,
    inst_details: dict,
    day_change_pct: float | None = None,
) -> str:
    """기관용 터미널: 단기·중기·장기 + MACD + ATR 리스크."""
    fac = inst_details.get("factors") or {}
    wts = inst_details.get("weights") or {}
    comp = inst_details.get("composite_100", "")
    vq = inst_details.get("volume_quality", "—")
    sl = inst_details.get("stop_loss")
    tp = inst_details.get("take_profit")
    rr = inst_details.get("rr_ratio")

    short_col = f"ma_{short_window}"
    mid_col = f"ma_{mid_window}"
    long_col = f"ma_{long_window}"

    close = float(last["close"]) if pd.notna(last.get("close")) else float("nan")
    close_txt = "—" if pd.isna(last.get("close")) else f"{float(last['close']):,.2f}"
    ma_s = last.get(short_col)
    ma_m = last.get(mid_col)
    ma_l = last.get(long_col)

    rsi_s = f"{float(last['rsi']):.1f}" if pd.notna(last.get("rsi")) else "—"
    rsi_zone = ""
    if pd.notna(last.get("rsi")):
        rv = float(last["rsi"])
        if rv >= 70:
            rsi_zone = " (과열)"
        elif rv <= 30:
            rsi_zone = " (과매도)"
        else:
            rsi_zone = " (중립대)"

    macd_cross, macd_nums = _macd_state_line(last)

    bb_note = ""
    if (
        pd.notna(last.get("bb_upper"))
        and pd.notna(last.get("bb_lower"))
        and pd.notna(last.get("close"))
    ):
        u, lo = float(last["bb_upper"]), float(last["bb_lower"])
        if u > lo:
            pb = (float(last["close"]) - lo) / (u - lo) * 100.0
            mid_bb = last.get("bb_mid")
            mid_txt = f"{float(mid_bb):,.2f}" if pd.notna(mid_bb) else "—"
            bb_note = f"BB %B 약 {pb:.0f}% (중간 {mid_txt})"

    wt_line = ""
    if wts:
        wt_line = (
            f"가중치  추세 {wts.get('trend', 0):.0%} · 모멘텀 {wts.get('momentum', 0):.0%} · "
            f"변동성 {wts.get('volatility', 0):.0%} · 거래량 {wts.get('volume', 0):.0%}"
        )

    fac_line = (
        f"팩터(−1~1)  추세 {fac.get('trend', '—')} · 모멘텀 {fac.get('momentum', '—')} · "
        f"변동성 {fac.get('volatility', '—')} · 거래량 {fac.get('volume', '—')}"
    )

    sl_s = f"{float(sl):,.2f}" if pd.notna(sl) else "—"
    tp_s = f"{float(tp):,.2f}" if pd.notna(tp) else "—"
    rr_s = f"{float(rr):.2f}:1" if pd.notna(rr) else "—"
    atr_s = f"{float(last['atr']):.2f}" if pd.notna(last.get("atr")) else "—"

    tk = last.get("ichi_tenkan")
    kj = last.get("ichi_kijun")
    ichi_mid = "—"
    if pd.notna(tk) and pd.notna(kj):
        ichi_mid = "전환>기준 (중기 추세 우호)" if float(tk) > float(kj) else "전환<기준 (중기 추세 불리)"

    sec_short = f"""<div class="qt-section">■ 단기 (Short · ~{short_window}일·모멘텀)</div>
<div class="qt-row"><span class="qt-k">MA {short_window}</span> <span class="qt-v">{_price_vs_ma_text(close, ma_s)}</span></div>
<div class="qt-row"><span class="qt-k">RSI({rsi_window})</span> <span class="qt-v">{rsi_s}</span><span class="qt-muted">{rsi_zone}</span></div>
<div class="qt-row"><span class="qt-k">MACD(12/26/9)</span> <span class="qt-v">{macd_cross}</span></div>
<div class="qt-row qt-muted">{macd_nums}</div>"""

    sec_mid = f"""<div class="qt-section">■ 중기 (Medium · ~{mid_window}일·일목·밴드)</div>
<div class="qt-row"><span class="qt-k">MA {mid_window}</span> <span class="qt-v">{_price_vs_ma_text(close, ma_m)}</span></div>
<div class="qt-row"><span class="qt-k">일목(9/26)</span> <span class="qt-v">{ichi_mid}</span></div>
<div class="qt-row"><span class="qt-k">볼린저</span> <span class="qt-v">{bb_note or "—"}</span></div>"""

    sec_long = f"""<div class="qt-section">■ 장기 (Long · ~{long_window}일·구름)</div>
<div class="qt-row"><span class="qt-k">MA {long_window}</span> <span class="qt-v">{_price_vs_ma_text(close, ma_l)}</span></div>
<div class="qt-row"><span class="qt-k">일목 구름</span> <span class="qt-v">{_cloud_position_text(last)}</span></div>"""

    chg_cell = _day_change_pct_html(day_change_pct)
    head = f"""<div class="quant-terminal">
<div class="qt-row"><span class="qt-k">티커</span> <span class="qt-v">{ticker.upper()}</span></div>
<div class="qt-row"><span class="qt-k">종가(USD)</span> <span class="qt-v">{close_txt}</span> <span class="qt-muted">· 전일대비</span> {chg_cell} <span class="qt-muted">(최근 확정)</span></div>
<div class="qt-row"><span class="qt-k">멀티팩터</span> <span class="qt-v">{comp}</span> <span class="qt-muted">(0~100)</span></div>
<div class="qt-row"><span class="qt-k">종합 판단</span> <span class="qt-ok">{inst_headline}</span></div>
<div class="qt-row"><span class="qt-k">한줄 의사결정</span> <span class="qt-v">{inst_details.get("action", "관망")}</span></div>
{sec_short}
{sec_mid}
{sec_long}
<div class="qt-section">■ 리스크 · 수급 · 퀀트 요약</div>
<div class="qt-row qt-muted">{fac_line}</div>
<div class="qt-row qt-muted">{wt_line}</div>
<div class="qt-row"><span class="qt-k">거래량 품질</span> <span class="qt-v">{vq}</span></div>
<div class="qt-row"><span class="qt-k">ATR({atr_window}일)</span> <span class="qt-v">{atr_s}</span></div>
<div class="qt-row"><span class="qt-k">손절</span> <span class="qt-warn">{sl_s}</span>
  <span class="qt-k"> 목표</span> <span class="qt-ok">{tp_s}</span>
  <span class="qt-k"> R</span> <span class="qt-v">{rr_s}</span></div>
<div class="qt-row qt-muted">손절/목표 = 종가 ± ATR×({atr_stop_mult:g} / {atr_take_mult:g})</div>
</div>"""
    return head


def plot_price_ma_ichimoku_rsi(
    df: pd.DataFrame,
    ticker: str,
    short_window: int,
    long_window: int,
    mid_window: int | None = None,
):
    """상단: 캔들 + 단·중·장 이평 + 일목 구름만(전환/기준선·크로스 마커 제거) / MACD / RSI."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = add_ichimoku_columns(df)
    short_col = f"ma_{short_window}"
    long_col = f"ma_{long_window}"
    mid_col = f"ma_{mid_window}" if mid_window is not None else None

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.54, 0.22, 0.24],
        subplot_titles=(
            f"{ticker} — 가격 · 구름대 · MA(단·중·장)",
            "MACD (12 / 26 / 9)",
            "RSI",
        ),
    )

    idx = d.index
    sa = d["ichi_senkou_a"]
    sb = d["ichi_senkou_b"]

    # 구름대: 선행 A/B 사이 (양운 A≥B 녹색, 음운 A<B 적색 — 유효 구간만 마스크)
    top = pd.concat([sa, sb], axis=1).max(axis=1)
    bot = pd.concat([sa, sb], axis=1).min(axis=1)
    valid = sa.notna() & sb.notna()
    bull = valid & (sa >= sb)
    bear = valid & (sa < sb)

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=top.where(bull),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=bot.where(bull),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(39, 174, 96, 0.28)",
            name="구름 양운(A≥B)",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=top.where(bear),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=bot.where(bear),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(231, 76, 60, 0.28)",
            name="구름 음운(A<B)",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=d[short_col],
            name=f"단기 MA {short_window}",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )
    if mid_col is not None and mid_col in d.columns:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=d[mid_col],
                name=f"중기 MA {mid_window}",
                line=dict(color="#16a085", width=1.4),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=d[long_col],
            name=f"장기 MA {long_window}",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="가격",
        ),
        row=1,
        col=1,
    )

    macd_line = d["macd"] if "macd" in d.columns else pd.Series(index=d.index, dtype=float)
    macd_sig = d["macd_signal"] if "macd_signal" in d.columns else pd.Series(index=d.index, dtype=float)
    macd_hist = d["macd_hist"] if "macd_hist" in d.columns else pd.Series(index=d.index, dtype=float)
    hist_colors = [
        "#2ecc71" if (pd.notna(v) and float(v) >= 0) else "#e74c3c"
        for v in macd_hist.fillna(0).values
    ]
    fig.add_trace(
        go.Bar(
            x=idx,
            y=macd_hist,
            name="MACD Hist",
            marker_color=hist_colors,
            opacity=0.45,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=macd_line,
            name="MACD",
            line=dict(color="#2980b9", width=1.3),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=macd_sig,
            name="Signal",
            line=dict(color="#e67e22", width=1.1),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(0,0,0,0.25)", row=2, col=1)

    rsi_series = d["rsi"] if "rsi" in d.columns else pd.Series(index=d.index, dtype=float)
    fig.add_trace(
        go.Scatter(x=idx, y=rsi_series, name="RSI", line=dict(color="#2980b9")),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(0,0,0,0.35)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,0,0,0.35)", row=3, col=1)

    fig.update_layout(
        height=980,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    for r in (1, 2, 3):
        fig.update_xaxes(rangeslider_visible=False, row=r, col=1)

    return fig


def main() -> None:
    st.set_page_config(page_title="미국 주식 매매타이밍 대시보드", layout="wide")

    # 화면에 더 많이 보이도록: 전체 글씨 크기 통일 + 여백 축소 (세션당 1회만 주입)
    base_font_px = 14
    if st.session_state.get("_stockapp_css_done") is not True:
        st.session_state["_stockapp_css_done"] = True
        st.markdown(
            f"""
<style>
/* Reduce overall paddings/margins */
div.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 1.0rem;
  padding-left: 1.0rem;
  padding-right: 1.0rem;
  max-width: 100%;
}}

/* Unify font size everywhere (including headers/metrics/inputs) */
html, body, [class*="st-"], .stApp, .stMarkdown, .stText, .stDataFrame, .stMetric, .stButton,
label, p, span, div {{
  font-size: {base_font_px}px !important;
  line-height: 1.25 !important;
}}

/* Streamlit markdown headers -> same size as body */
h1, h2, h3, h4, h5, h6 {{
  font-size: {base_font_px}px !important;
  line-height: 1.25 !important;
  margin: 0.25rem 0 !important;
}}

/* Metric value/label -> same size */
div[data-testid="stMetricValue"] {{
  font-size: {base_font_px}px !important;
  line-height: 1.25 !important;
}}
div[data-testid="stMetricLabel"] {{
  font-size: {base_font_px}px !important;
  line-height: 1.25 !important;
}}

/* Tighten spacing between elements */
div[data-testid="stVerticalBlock"] > div {{
  gap: 0.35rem;
}}

/* 터미널 스타일 요약 패널 */
.quant-terminal {{
  background: #0d1117;
  color: #c9d1d9;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  padding: 0.9rem 1.1rem;
  border-radius: 6px;
  border: 1px solid #30363d;
  margin: 0.35rem 0 0.75rem 0;
  font-size: {base_font_px}px !important;
  line-height: 1.45 !important;
}}
.quant-terminal .qt-row {{ margin: 0.2rem 0; }}
.quant-terminal .qt-k {{ color: #8b949e; }}
.quant-terminal .qt-v {{ color: #58a6ff; font-weight: 600; }}
.quant-terminal .qt-ok {{ color: #3fb950; }}
.quant-terminal .qt-warn {{ color: #f85149; }}
.quant-terminal .qt-muted {{ color: #6e7681; font-size: 0.92em; }}
.quant-terminal .qt-section {{
  color: #d29922;
  font-weight: 700;
  margin-top: 0.55rem;
  margin-bottom: 0.15rem;
  padding-bottom: 0.15rem;
  border-bottom: 1px solid #30363d;
  letter-spacing: 0.02em;
}}
</style>
""",
            unsafe_allow_html=True,
        )

    st.title("📈 미국 주식 매매타이밍 대시보드")

    # 사이드바 입력
    with st.sidebar:
        st.header("설정")

        if "portfolio_holdings" not in st.session_state:
            st.session_state.portfolio_holdings = load_portfolio_holdings()
        portfolio_holdings: dict[str, int] = st.session_state.portfolio_holdings

        with st.expander("나의 포트폴리오 편집", expanded=False):
            st.caption("추가·삭제 시 `portfolio_holdings.json`에 저장됩니다.")
            if portfolio_holdings:
                st.dataframe(
                    pd.DataFrame(
                        [{"티커": k, "보유수량": v} for k, v in sorted(portfolio_holdings.items())]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("등록된 종목이 없습니다. 아래에서 추가하세요.")

            a1, a2, a3 = st.columns([2, 1, 1])
            with a1:
                add_sym = st.text_input("추가할 티커", value="", key="pf_add_sym", placeholder="예: AAPL")
            with a2:
                add_qty = st.number_input("수량", min_value=0, value=10, step=1, key="pf_add_qty")
            with a3:
                st.write("")
                st.write("")
                do_add = st.button("종목 추가", key="pf_do_add")
            if do_add:
                sym = normalize_portfolio_ticker(add_sym)
                if not sym:
                    st.warning("유효한 티커를 입력하세요.")
                else:
                    nh = dict(st.session_state.portfolio_holdings)
                    nh[sym] = int(add_qty)
                    st.session_state.portfolio_holdings = nh
                    save_portfolio_holdings(nh)
                    st.success(f"{sym} {int(add_qty)}주 반영됨")
                    st.rerun()

            if portfolio_holdings:
                rm_sym = st.selectbox(
                    "삭제할 티커",
                    options=sorted(portfolio_holdings.keys()),
                    key="pf_rm_sym",
                )
                if st.button("선택 종목 삭제", key="pf_do_rm"):
                    nh = dict(st.session_state.portfolio_holdings)
                    nh.pop(rm_sym, None)
                    st.session_state.portfolio_holdings = nh
                    save_portfolio_holdings(nh)
                    st.rerun()

        st.divider()

        # 내 보유 종목 리스트
        holdings_list = sorted(portfolio_holdings.keys())
        default_ticker = holdings_list[0] if holdings_list else "AAPL"

        if holdings_list:
            selected_holding = st.selectbox(
                "내 보유 종목에서 선택",
                options=holdings_list,
                index=0,
            )
        else:
            st.caption("포트폴리오가 비어 있으면 위 편집에서 종목을 추가하세요.")
            selected_holding = default_ticker

        # text_input과 동기화: 사용자가 직접 수정도 가능
        current_ticker_default = selected_holding if holdings_list else default_ticker
        ticker = st.text_input(
            "티커 (예: AAPL, TSLA, MSFT)",
            value=current_ticker_default,
        )

        # 한국 시간 기준 현재 시각
        now_kst = dt.datetime.now(ZoneInfo("Asia/Seoul"))
        # 미국 동부 시간 기준 현재 날짜 (미국 주식 시장 기준)
        now_us = now_kst.astimezone(ZoneInfo("America/New_York")).date()
        # 미국 시장 최근 개장일 기준 (주말/휴장일 보정)
        if now_us.weekday() >= 5:  # 5=토요일, 6=일요일
            # 토요일이면 금요일(-1), 일요일이면 금요일(-2)
            days_back = now_us.weekday() - 4
            market_last_day = now_us - dt.timedelta(days=days_back)
        else:
            market_last_day = now_us

        default_start = market_last_day - dt.timedelta(days=365)

        start_date = st.date_input("시작일", value=default_start)
        end_date = st.date_input("종료일", value=market_last_day)

        col1, col2 = st.columns(2)
        with col1:
            short_window = st.number_input("단기 이동평균 (일)", min_value=3, max_value=120, value=20, step=1)
        with col2:
            long_window = st.number_input("장기 이동평균 (일)", min_value=10, max_value=365, value=60, step=1)
        mid_window = st.number_input(
            "중기 이동평균 (일)",
            min_value=5,
            max_value=240,
            value=50,
            step=1,
            help="차트·터미널 중기 구간: 단기와 장기 사이 이평",
        )

        st.divider()
        st.subheader("퀀트 멀티팩터 · 리스크")
        volume_filter = st.checkbox("거래량 팩터 반영(권장)", value=True)
        rsi_window = st.number_input("RSI 기간", min_value=5, max_value=50, value=14, step=1)
        atr_window = st.number_input("ATR 기간", min_value=5, max_value=50, value=14, step=1)
        bb_window = st.number_input("볼린저 기간", min_value=10, max_value=60, value=20, step=1)
        bb_k = st.number_input("볼린저 k", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
        atr_stop_mult = st.number_input("ATR 손절 배수", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        atr_take_mult = st.number_input("ATR 익절 배수", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
        fee_bps = st.number_input("왕복 수수료/슬리피지(bps)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

        if not (short_window < mid_window < long_window):
            st.warning("이동평균은 단기 < 중기 < 장기 순이어야 합니다.")

        load_portfolio = st.checkbox(
            "내 포트폴리오 요약 로드 (보유 전체·다종목 병렬)",
            value=False,
            help="켜면 분석 후 설정된 보유 종목 전체를 표로 보여 줍니다. 끄면 메인 분석만 실행되어 더 빠릅니다.",
        )

        run = st.button("분석하기")

    if not run:
        st.info("왼쪽 사이드바에서 매개변수를 설정한 뒤 **분석하기** 버튼을 눌러주세요.")
        return

    if not (short_window < mid_window < long_window):
        st.error("단기 < 중기 < 장기 순으로 설정해주세요. (크로스는 단기·장기 기준입니다.)")
        return

    # 한국/미국 시차 + 미국 시장 개장일을 고려해 종료일 보정
    now_kst = dt.datetime.now(ZoneInfo("Asia/Seoul"))
    now_us = now_kst.astimezone(ZoneInfo("America/New_York")).date()
    if now_us.weekday() >= 5:
        days_back = now_us.weekday() - 4
        market_last_day = now_us - dt.timedelta(days=days_back)
    else:
        market_last_day = now_us

    if end_date > market_last_day:
        st.warning(
            f"미국 최근 거래일({market_last_day}) 이후 날짜까지 선택되어, "
            "종가 데이터가 없을 수 있어 종료일을 최근 미국 거래일로 자동 조정했습니다."
        )
        end_date = market_last_day

    with st.spinner("데이터를 불러오는 중..."):
        df, err_msg = load_price_data(ticker, start_date, end_date)

    if df.empty:
        # 상세한 에러 메시지(가능하면)를 화면에 표시
        if err_msg:
            st.error(
                f"데이터를 가져오지 못했습니다.\n\n"
                f"**상세 오류 메시지**\n{err_msg}\n\n"
                f"선택한 티커: {ticker.upper()}, 기간: {start_date} ~ {end_date}"
            )
        else:
            st.error(
                "데이터를 가져오지 못했습니다. 티커나 기간, 미국 시장 개장 여부를 다시 확인해주세요."
            )
        return

    df = calculate_cross_signals(df, short_window, long_window)
    df[f"ma_{int(mid_window)}"] = df["close"].rolling(
        window=int(mid_window), min_periods=int(mid_window)
    ).mean()
    df = add_institutional_indicators(
        df,
        rsi_window=int(rsi_window),
        atr_window=int(atr_window),
        bb_window=int(bb_window),
        bb_k=float(bb_k),
    )
    latest_signal_text, latest_date = get_latest_signal(df)
    inst_headline, inst_details = institutional_signal_summary(
        df,
        short_window=short_window,
        long_window=long_window,
        volume_filter=bool(volume_filter),
        atr_stop_mult=float(atr_stop_mult),
        atr_take_mult=float(atr_take_mult),
    )
    week_outlook_label, week_outlook_detail = one_week_outlook(
        df,
        short_window=short_window,
        long_window=long_window,
    )
    week_price_label, week_price_detail = one_week_price_projection(df)
    vol_label, vol_detail = volume_change_summary(df)
    cross_proj_headline, cross_proj_detail = cross_projection_summary(
        df,
        short_window=short_window,
        long_window=long_window,
    )

    # 상단 요약 카드
    st.subheader("매매 타이밍 요약")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("티커", ticker.upper())
    with col_b:
        st.metric("최근 크로스/매매 의견", latest_signal_text)
    with col_c:
        if latest_date is not None:
            st.metric("최근 크로스 일자", latest_date.strftime("%Y-%m-%d"))
        else:
            st.metric("최근 크로스 일자", "없음")

    # 크로스 현재 상태 및 향후 예상 요약
    st.markdown(f"**크로스 상태/예상:** {cross_proj_headline}")
    with st.expander("크로스 상세 설명 보기"):
        st.write(cross_proj_detail)

    # 분석 티커 가격 요약 (현재가/1/2/3/5/2주(10거래일) 예상) — 회귀 1회
    price_terminal_row, last_close_num, prev_close_num = last_valid_close_snapshot(df)
    last_close = pd.NA if last_close_num is None else float(last_close_num)
    _mh = multi_horizon_price_labels(df, (1, 2, 3, 5, 10))
    proj1_label = _mh.get(1, "예측 불가")
    proj2_label = _mh.get(2, "예측 불가")
    proj3_label = _mh.get(3, "예측 불가")
    proj5_label = _mh.get(5, "예측 불가")
    proj10_label = _mh.get(10, "예측 불가")

    st.subheader("가격 요약 (참고용)")
    col_px1, col_px2, col_px3 = st.columns(3)
    with col_px1:
        st.metric("현재가격(USD)", "N/A" if pd.isna(last_close) else f"{float(last_close):,.2f}")
    with col_px2:
        st.metric("1거래일 후 예상", proj1_label)
    with col_px3:
        st.metric("2거래일 후 예상", proj2_label)

    col_px4, col_px5, col_px6 = st.columns(3)
    with col_px4:
        st.metric("3거래일 후 예상", proj3_label)
    with col_px5:
        st.metric("5거래일 후 예상", proj5_label)
    with col_px6:
        st.metric("2주 후 예상(10거래일)", proj10_label)

    st.subheader("향후 1주일 단기 전망 및 가격 예측 (참고용)")
    col_o1, col_o2 = st.columns([1, 2])
    with col_o1:
        st.metric("1주일 방향성 전망", week_outlook_label)
        st.metric("1주일 후 가격 예측치", week_price_label)
        st.metric("거래량 변화", vol_label)
    with col_o2:
        st.write(week_outlook_detail)
        st.write(week_price_detail)
        st.write(vol_detail)

    st.subheader("기관용 터미널 — 단기 · 중기 · 장기 · MACD · ATR")
    last = price_terminal_row if price_terminal_row is not None else df.iloc[-1]
    term_day_pct: float | None = None
    if last_close_num is not None and prev_close_num is not None and float(prev_close_num) != 0:
        term_day_pct = (float(last_close_num) / float(prev_close_num) - 1.0) * 100.0
    term_html = institutional_terminal_html(
        ticker,
        last,
        short_window=int(short_window),
        mid_window=int(mid_window),
        long_window=int(long_window),
        rsi_window=int(rsi_window),
        atr_window=int(atr_window),
        atr_stop_mult=float(atr_stop_mult),
        atr_take_mult=float(atr_take_mult),
        inst_headline=inst_headline,
        inst_details=inst_details,
        day_change_pct=term_day_pct,
    )
    st.markdown(term_html, unsafe_allow_html=True)

    reasons = inst_details.get("reasons", [])
    if reasons:
        with st.expander("멀티팩터 근거(항목별)", expanded=False):
            st.write("\n".join([f"- {r}" for r in reasons]))

    # 내 포트폴리오 (선택 시에만 — 다종목 네트워크 부하)
    if load_portfolio:
        st.subheader("내 포트폴리오 (멀티팩터 · 거래량 · ATR 손절/목표)")
        with st.spinner("포트폴리오 종목 데이터를 불러오는 중..."):
            snap = build_portfolio_snapshot(as_of=end_date, holdings=portfolio_holdings)
        if snap.empty:
            st.info("포트폴리오 요약을 계산할 수 없습니다. 데이터 소스 또는 티커를 확인해주세요.")
        else:
            total_usd = float(pd.to_numeric(snap["평가금액(USD)"], errors="coerce").fillna(0).sum())
            prev_total_raw = snap.get("_전일평가금액(USD)")
            prev_total_usd = pd.NA
            if prev_total_raw is not None:
                prev_total_series = pd.to_numeric(prev_total_raw, errors="coerce")
                if prev_total_series.notna().any():
                    prev_total_usd = float(prev_total_series.fillna(0).sum())

            if pd.notna(prev_total_usd) and float(prev_total_usd) > 0:
                delta_usd = total_usd - float(prev_total_usd)
                delta_pct = delta_usd / float(prev_total_usd) * 100
                st.metric(
                    "포트폴리오 총 평가금액(USD 기준)",
                    f"{total_usd:,.0f}",
                    delta=f"{delta_usd:,.0f} USD ({delta_pct:+.2f}%)",
                )
            else:
                st.metric("포트폴리오 총 평가금액(USD 기준)", f"{total_usd:,.0f}")

            snap_no_hidden = snap.drop(columns=["_전일평가금액(USD)"], errors="ignore")
            st.markdown("**종목별 상세 요약**")
            st.dataframe(snap_no_hidden, use_container_width=True, hide_index=True)

    st.subheader("룰 기반 백테스트(단순)")
    bt = backtest_ma_atr_strategy(
        df,
        short_window=short_window,
        long_window=long_window,
        atr_stop_mult=float(atr_stop_mult),
        atr_take_mult=float(atr_take_mult),
        fee_bps=float(fee_bps),
    )
    perf = performance_summary(bt["equity_curve"])
    if perf:
        import plotly.graph_objects as go

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric("총 수익률", f"{perf['total_return']*100:.1f}%")
        with col_p2:
            v = perf["vol"]
            st.metric("연환산 변동성", "N/A" if pd.isna(v) else f"{float(v)*100:.1f}%")
        with col_p3:
            s = perf["sharpe"]
            st.metric("샤프(근사)", "N/A" if pd.isna(s) else f"{float(s):.2f}")
        with col_p4:
            st.metric("최대낙폭", f"{float(perf['max_dd'])*100:.1f}%")

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["equity_curve"], mode="lines", name="Equity"))
        eq_fig.update_layout(
            title="전략 누적 수익(근사)",
            xaxis_title="Date",
            yaxis_title="Equity (Start=1.0)",
            template="plotly_white",
        )
        st.plotly_chart(eq_fig, use_container_width=True)
    else:
        st.info("백테스트를 계산하기에 데이터가 부족합니다(지표 계산에 필요한 기간을 늘려보세요).")

    st.subheader("가격 차트 (상단: 캔들·구름·MA만 / 하단: MACD·RSI)")
    combo_fig = plot_price_ma_ichimoku_rsi(
        df,
        ticker.upper(),
        short_window=short_window,
        long_window=long_window,
        mid_window=int(mid_window),
    )
    st.plotly_chart(combo_fig, use_container_width=True)

    st.subheader("크로스 발생 내역")
    cross_table = df[df["cross"] != ""][
        ["close", f"ma_{short_window}", f"ma_{long_window}", "cross"]
    ].copy()
    cross_table = cross_table.rename(
        columns={
            "close": "종가",
            f"ma_{short_window}": f"MA{short_window}",
            f"ma_{long_window}": f"MA{long_window}",
            "cross": "신호",
        }
    )
    cross_table.index.name = "날짜"

    if cross_table.empty:
        st.write("선택한 기간 내 크로스 신호가 없습니다.")
    else:
        st.dataframe(cross_table[::-1], use_container_width=True)


if __name__ == "__main__":
    main()
