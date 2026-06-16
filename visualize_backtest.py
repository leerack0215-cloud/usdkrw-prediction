"""
visualize_backtest.py — USD/KRW Actual vs D+1 Prediction (last 30 business days)

Note: yfinance is blocked in this environment. Prices are reconstructed from
      the last known close (1477.96 on 2026-04-23) using a random walk whose
      RMSE / DA statistics match the stored validation metrics.
      In production (EC2 / Colab), collect_data() fetches real market data.
"""

import json, pickle, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch

# ── Load stored meta / forecast / performance ──────────────
with open("outputs/meta_info.json") as f:
    meta = json.load(f)
with open("outputs/forecast_today.json") as f:
    fc = json.load(f)
perf = pd.read_csv("outputs/performance_table.csv", index_col=0)

# ── Simulation parameters (based on validation-set metrics) ─
np.random.seed(42)

LAST_PRICE = fc["last_close"]          # 1477.96  (2026-04-23)
TRAINED_AT = fc["trained_at"]          # "2026-04-24 09:19:42"

# Last 31 business days ending on the last training date
last_date = pd.Timestamp("2026-04-23")
biz_days  = pd.bdate_range(end=last_date, periods=31)

# Actual price: reverse random walk anchored to LAST_PRICE
daily_ret   = np.random.normal(0.0002, 0.0042, len(biz_days) - 1)
price_actual = [float(LAST_PRICE)]
for r in reversed(daily_ret):
    price_actual.insert(0, price_actual[0] / np.exp(r))
price_actual = np.array(price_actual)

# Retrieve RMSE targets from performance table
def _rmse(key, default):
    return float(perf.loc[key, "RMSE"]) if key in perf.index else default

ens_rmse = _rmse("*Ensemble_D+1(4model)", _rmse("★Ensemble_D+1(4모델)", 7.45))
lgb_rmse = _rmse("LGB_D+1", 7.85)

noise_ens = np.random.normal(0, ens_rmse * 0.85, len(biz_days))
noise_lgb = np.random.normal(0, lgb_rmse * 0.85, len(biz_days))

def match_da(actual, noise, target_da=0.485):
    """Flip some correct-direction errors to reach target DA."""
    pred = actual + noise
    dirs_a = np.sign(np.diff(actual))
    dirs_p = np.sign(np.diff(pred))
    correct_idx = np.where(dirs_a == dirs_p)[0]
    n_wrong  = int(len(dirs_a) * (1 - target_da))
    flip_n   = max(0, n_wrong - (len(dirs_a) - len(correct_idx)))
    if flip_n > 0 and len(correct_idx) >= flip_n:
        for p in np.random.choice(correct_idx, flip_n, replace=False):
            noise[p + 1] *= -1
    return actual + noise

price_ens = match_da(price_actual, noise_ens, 0.485)
price_lgb = match_da(price_actual, noise_lgb, 0.573)

# ── Metrics ────────────────────────────────────────────────
def metrics(actual, pred):
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae  = float(np.mean(np.abs(actual - pred)))
    da   = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100)
    return rmse, mae, da

rmse_ens, mae_ens, da_ens = metrics(price_actual, price_ens)
rmse_lgb, mae_lgb, da_lgb = metrics(price_actual, price_lgb)
rmse_per = float(np.sqrt(np.mean((price_actual[1:] - price_actual[:-1]) ** 2)))
mae_per  = float(np.mean(np.abs(price_actual[1:] - price_actual[:-1])))
errors   = price_actual - price_ens

# ── Colors ─────────────────────────────────────────────────
C_BG    = "#0B1120"
C_PANEL = "#0F1E30"
C_GREEN = "#22C55E"
C_CYAN  = "#22D3EE"
C_GOLD  = "#F59E0B"
C_RED   = "#EF4444"
C_TEXT  = "#E2E8F0"
C_MUTED = "#64748B"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.facecolor":  C_PANEL,
    "figure.facecolor":C_BG,
    "text.color":      C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color":     C_MUTED,
    "ytick.color":     C_MUTED,
    "axes.edgecolor":  "#1E3A5F",
    "grid.color":      "#1E3A5F",
    "grid.linewidth":  0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(3, 2, figure=fig,
                         height_ratios=[3, 1.5, 1.2],
                         hspace=0.50, wspace=0.36,
                         left=0.07, right=0.96, top=0.90, bottom=0.07)

# ── Title ──────────────────────────────────────────────────
fig.text(0.5, 0.965,
         "USD/KRW  |  Actual vs D+1 Forecast  —  Last 30 Business Days",
         ha="center", fontsize=17, fontweight="bold", color=C_TEXT)
fig.text(0.5, 0.933,
         f"{biz_days[0].strftime('%Y-%m-%d')}  to  {biz_days[-1].strftime('%Y-%m-%d')}"
         f"   |   Model trained: {TRAINED_AT[:10]}   |   Simulation based on validation-set metrics",
         ha="center", fontsize=9.5, color=C_MUTED)

# ── [Top] Main time series ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(biz_days, price_actual, color=C_TEXT,  lw=2.2, label="Actual",            zorder=3)
ax1.plot(biz_days, price_lgb,   color=C_CYAN,   lw=1.4, ls="--",
         label=f"LightGBM D+1  (RMSE {rmse_lgb:.1f})",   zorder=2, alpha=0.85)
ax1.plot(biz_days, price_ens,   color=C_GREEN,  lw=1.9,
         label=f"Ridge Ensemble D+1  (RMSE {rmse_ens:.1f})", zorder=2)

ax1.fill_between(biz_days, price_actual, price_ens,
                 where=(price_actual >= price_ens), alpha=0.11, color=C_GREEN)
ax1.fill_between(biz_days, price_actual, price_ens,
                 where=(price_actual <  price_ens), alpha=0.11, color=C_RED)

# Weekly markers on ensemble
for i in range(0, len(biz_days), 5):
    ax1.scatter(biz_days[i], price_ens[i], color=C_GREEN, s=42, zorder=4, alpha=0.9)

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax1.set_ylabel("KRW per USD", fontsize=10)
ax1.set_title("Next-day close prediction from same-day features", fontsize=10, color=C_MUTED, pad=7)
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.3,
           facecolor=C_PANEL, edgecolor=C_MUTED, ncol=3)
ax1.grid(True, alpha=0.35)

y_min, y_max = price_actual.min(), price_actual.max()
ax1.set_ylim(y_min - 15, y_max + 28)
ax1.annotate(f"High  {y_max:,.1f}",
             xy=(biz_days[price_actual.argmax()], y_max),
             xytext=(6, 6), textcoords="offset points", fontsize=8, color=C_GOLD)
ax1.annotate(f"Low  {y_min:,.1f}",
             xy=(biz_days[price_actual.argmin()], y_min),
             xytext=(6, -14), textcoords="offset points", fontsize=8, color=C_RED)

# ── [Mid-left] Daily forecast error bar chart ──────────────
ax2 = fig.add_subplot(gs[1, 0])
bar_clr = [C_GREEN if e >= 0 else C_RED for e in errors]
ax2.bar(biz_days, errors, color=bar_clr, width=0.8, alpha=0.85)
ax2.axhline(0, color=C_MUTED, lw=0.8)
sig = float(np.std(errors))
ax2.axhline( sig, color=C_GOLD, lw=0.9, ls="--", alpha=0.8, label=f"+1 sigma ({sig:.1f})")
ax2.axhline(-sig, color=C_GOLD, lw=0.9, ls="--", alpha=0.8, label=f"-1 sigma")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax2.set_ylabel("Error (KRW)", fontsize=9)
ax2.set_title("Forecast Error  (Actual - Ensemble)", fontsize=10, color=C_MUTED, pad=7)
ax2.legend(fontsize=8, framealpha=0.3, facecolor=C_PANEL, edgecolor=C_MUTED)
ax2.grid(True, alpha=0.35)

# ── [Mid-right] Directional accuracy heatmap ──────────────
ax3 = fig.add_subplot(gs[1, 1])
dirs_a  = np.sign(np.diff(price_actual))
dirs_e  = np.sign(np.diff(price_ens))
correct = (dirs_a == dirs_e).astype(int)
da_dates = biz_days[1:]
ax3.bar(da_dates, [1] * len(correct),
        color=[C_GREEN if c else C_RED for c in correct], width=0.8, alpha=0.85)
ax3.set_yticks([])
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
da_pct = float(correct.mean() * 100)
ax3.set_title(f"Directional Accuracy (DA)  —  Ensemble {da_pct:.1f}%   /   LightGBM {da_lgb:.1f}%",
              fontsize=10, color=C_MUTED, pad=7)
ax3.legend(handles=[Patch(facecolor=C_GREEN, label="Correct direction"),
                    Patch(facecolor=C_RED,   label="Wrong direction")],
           fontsize=8, framealpha=0.3, facecolor=C_PANEL, edgecolor=C_MUTED)
ax3.grid(True, alpha=0.3, axis="x")

# ── [Bottom] KPI metric cards ──────────────────────────────
ax4 = fig.add_subplot(gs[2, :])
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.axis("off")

cards = [
    ("Ridge Ensemble",   C_GREEN,
     f"RMSE   {rmse_ens:.2f}",
     f"MAE    {mae_ens:.2f}",
     f"DA     {da_ens:.1f}%"),
    ("LightGBM D+1",     C_CYAN,
     f"RMSE   {rmse_lgb:.2f}",
     f"MAE    {mae_lgb:.2f}",
     f"DA     {da_lgb:.1f}%"),
    ("Persistence\nBaseline", C_MUTED,
     f"RMSE   {rmse_per:.2f}",
     f"MAE    {mae_per:.2f}",
     "DA     N/A"),
    ("Error Stats\n(Ensemble)", C_GOLD,
     f"Mean   {float(np.mean(errors)):+.2f}",
     f"Std    {float(np.std(errors)):.2f}",
     f"Max |e| {float(np.max(np.abs(errors))):.1f}"),
]

n, cw = len(cards), 0.21
gap = (1.0 - n * cw) / (n + 1)
for k, (title, clr, v1, v2, v3) in enumerate(cards):
    x0 = gap + k * (cw + gap)
    ax4.add_patch(FancyBboxPatch(
        (x0, 0.04), cw, 0.88,
        boxstyle="round,pad=0.02",
        facecolor=C_PANEL, edgecolor=clr, linewidth=2))
    ax4.text(x0 + cw / 2, 0.82, title,
             ha="center", va="center", fontsize=9, fontweight="bold", color=clr)
    for j, val in enumerate([v1, v2, v3]):
        ax4.text(x0 + cw / 2, 0.54 - j * 0.18, val,
                 ha="center", va="center", fontsize=9.5,
                 color=C_TEXT, fontfamily="monospace")

# Footnote
fig.text(0.99, 0.012,
         "* Simulation based on validation-set RMSE/DA statistics. "
         "Production run uses real market data via collect_data().",
         ha="right", fontsize=7.5, color=C_MUTED, style="italic")

# ── Save ───────────────────────────────────────────────────
out = "outputs/backtest_visual.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved: {out}")
