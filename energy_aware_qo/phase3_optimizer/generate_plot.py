#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Set plotting style
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)
os.makedirs('generated_paper_figures', exist_ok=True)

print(
    'Loading data and generating paper figures (with dual JPG/PDF export)'
)

# Centralized Style Configuration (Borderless/Clean Markers)
PLOT_STYLE = {
    'color_mlp': '#d62728',
    'color_gnn': '#2ca02c',
    'color_mlp_alt': '#ff7f0e',
    'alpha_base': 0.7,   
    'alpha_target': 0.8, 
    'edge_color': 'none', 
    'size': 45,         
    'linewidth': 1.5,
    'font_title': 12,   
    'font_label': 9,
}


def save_fig(base_name):
  """Helper to export figures in both high-resolution raster (JPG) and vector (PDF) formats."""
  plt.savefig(
      f'generated_paper_figures/{base_name}.jpg', dpi=300, bbox_inches='tight'
  )
  plt.savefig(f'generated_paper_figures/{base_name}.pdf', bbox_inches='tight')


# ---------------------------------------------------------------------------
# 0. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def load_csv(path):
  if os.path.exists(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
    return df
  return None


def clean_val(val):
  if pd.isna(val) or val is None:
    return np.nan
  val_str = re.sub(r'[^\d.]', '', str(val).strip())
  try:
    return float(val_str)
  except ValueError:
    return np.nan


def get_col(df, keywords):
  if df is None:
    return None
  for col in df.columns:
    col_clean = str(col).replace('\xa0', ' ').strip().lower()
    if all(k.lower() in col_clean for k in keywords):
      return col
  return None


def create_comparison_plot(
    ax, df_mlp, df_gnn, p_col, a_col, title, xlabel, ylabel
):
  if (
      df_mlp is not None
      and not df_mlp.empty
      and p_col in df_mlp.columns
      and a_col in df_mlp.columns
  ):
    ax.scatter(
        pd.to_numeric(df_mlp[p_col], errors='coerce'),
        pd.to_numeric(df_mlp[a_col], errors='coerce'),
        color=PLOT_STYLE['color_mlp'],
        marker='o',
        alpha=PLOT_STYLE['alpha_base'],
        edgecolors=PLOT_STYLE['edge_color'],
        s=PLOT_STYLE['size'],
        label='Baseline MLP',
    )

  if (
      df_gnn is not None
      and not df_gnn.empty
      and p_col in df_gnn.columns
      and a_col in df_gnn.columns
  ):
    ax.scatter(
        pd.to_numeric(df_gnn[p_col], errors='coerce'),
        pd.to_numeric(df_gnn[a_col], errors='coerce'),
        color=PLOT_STYLE['color_gnn'],
        marker='^',
        alpha=PLOT_STYLE['alpha_target'],
        edgecolors=PLOT_STYLE['edge_color'],
        s=PLOT_STYLE['size'],
        label='GreenQO',
    )

  max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
  ax.plot(
      [0, max_val],
      [0, max_val],
      'k--',
      linewidth=PLOT_STYLE['linewidth'],
      label='Optimal Alignment',
  )
  ax.set_title(title, fontsize=PLOT_STYLE['font_title'], fontweight='normal')
  ax.set_xlabel(xlabel, fontsize=PLOT_STYLE['font_label'])
  ax.set_ylabel(ylabel, fontsize=PLOT_STYLE['font_label'])
  ax.legend(loc='upper left', frameon=True)


# ---------------------------------------------------------------------------
# 1. LOAD DATASETS
# ---------------------------------------------------------------------------
baseline_path = (
    '/home/admin_tidenek/energy_aware_qo/phase3_optimizer/paper_baseline_postgres_metrics.csv'
)
history_path = (
    '/home/admin_tidenek/energy_aware_qo/phase3_optimizer/optimizer_selection_history.csv'
)
mlp_eval_path = (
    '/home/admin_tidenek/energy_aware_qo/phase1_mlp/evaluation_predictions.csv'
)
gnn_time_path = (
    '/home/admin_tidenek/energy_aware_qo/phase2_gnn/gnn_evaluation_predictions.csv'
)
gnn_energy_path = (
    '/home/admin_tidenek/energy_aware_qo/phase2_gnn/gnn_energy_evaluation_predictions.csv'
)
disparity_path = (
    '/home/admin_tidenek/energy_aware_qo/phase3_optimizer/all_detected_energy_disparity.csv'
)
weight_xlsx = (
    '/home/admin_tidenek/energy_aware_qo/phase1_mlp/extracted_model_weights_comparison.xlsx'
)
weight_csv = (
    '/home/admin_tidenek/energy_aware_qo/phase1_mlp/extracted_model_weight_comparison.csv'
)

df_baseline = load_csv(baseline_path)
df_history = load_csv(history_path)
df_mlp = load_csv(mlp_eval_path)
df_gnn_time = load_csv(gnn_time_path)
df_gnn_energy = load_csv(gnn_energy_path)
df_disparity = load_csv(disparity_path)

df_weights = None
if os.path.exists(weight_xlsx):
  try:
    df_weights = pd.read_excel(weight_xlsx, engine='openpyxl')
    df_weights.columns = df_weights.columns.str.strip()
  except Exception:
    pass
if df_weights is None and os.path.exists(weight_csv):
  df_weights = load_csv(weight_csv)
# ---------------------------------------------------------------------------
# FIGURE 2: HEAVY-TAIL TRANSFORMATION
# ---------------------------------------------------------------------------
fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(14, 6))

if (
    df_gnn_time is not None
    and 'Actual_Execution_Time_Seconds' in df_gnn_time.columns
):
    actual_times = (
        pd.to_numeric(
            df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce'
        )
        .dropna()
        .sort_values()
        .values
    )

    ax_raw.plot(
        actual_times,
        marker='o',
        color='#d62728',
        linestyle='-',
        alpha=0.7,
        markersize=4,
    )
    ax_raw.set_title(
        'Raw Target Space (Heavy-Tail Distribution)',
        fontsize=14,
        fontweight='bold',
    )
    ax_raw.set_xlabel('Query Index (Sorted)', fontsize=12, fontweight='bold')
    ax_raw.set_ylabel('Actual Execution Time (Seconds)', fontsize=12, fontweight='bold')
    for label in ax_raw.get_xticklabels() + ax_raw.get_yticklabels():
        label.set_fontsize(11)
        label.set_fontweight('bold')

    log_times = np.log1p(actual_times)
    ax_log.plot(
        log_times,
        marker='o',
        color='#2ca02c',
        linestyle='-',
        alpha=0.7,
        markersize=4,
    )
    ax_log.set_title(
        'Log-Transformed Target Space (Stabilized Variance)',
        fontsize=14,
        fontweight='bold',
    )
    ax_log.set_xlabel('Query Index (Sorted)', fontsize=12, fontweight='bold')
    ax_log.set_ylabel('Log(Execution Time + 1)', fontsize=12, fontweight='bold')
    for label in ax_log.get_xticklabels() + ax_log.get_yticklabels():
        label.set_fontsize(11)
        label.set_fontweight('bold')

plt.tight_layout()
save_fig('fig2_heavy_tail_transformation')
plt.close()
print('Figure 2 generated.')


# ---------------------------------------------------------------------------
# FIGURE 2A: RAW TARGET SPACE (HEAVY-TAIL DISTRIBUTION)
# ---------------------------------------------------------------------------
fig, ax_raw = plt.subplots(figsize=(8, 6), dpi=300)

if (
    df_gnn_time is not None
    and 'Actual_Execution_Time_Seconds' in df_gnn_time.columns
):
    actual_times = (
        pd.to_numeric(
            df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce'
        )
        .dropna()
        .sort_values()
        .values
    )

    ax_raw.plot(
        actual_times,
        marker='o',
        color='#d62728',
        linestyle='-',
        alpha=0.7,
        markersize=4,
    )
    ax_raw.set_title(
        'Raw Target Space (Heavy-Tail Distribution)',
        fontsize=14,
        fontweight='bold',
        pad=12,
    )
    ax_raw.set_xlabel('Query Index (Sorted)', fontsize=12, fontweight='bold')
    ax_raw.set_ylabel('Actual Execution Time (Seconds)', fontsize=12, fontweight='bold')
    
    for label in ax_raw.get_xticklabels() + ax_raw.get_yticklabels():
        label.set_fontsize(11)
        label.set_fontweight('bold')

plt.tight_layout()
save_fig('fig2a_raw_target_space')
plt.close()
print('Figure 2a (Raw Target Space) generated.')

# ---------------------------------------------------------------------------
# FIGURE 2B: LOG-TRANSFORMED TARGET SPACE (STABILIZED VARIANCE)
# ---------------------------------------------------------------------------
fig, ax_log = plt.subplots(figsize=(8, 6), dpi=300)

if (
    df_gnn_time is not None
    and 'Actual_Execution_Time_Seconds' in df_gnn_time.columns
):
    actual_times = (
        pd.to_numeric(
            df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce'
        )
        .dropna()
        .sort_values()
        .values
    )
    log_times = np.log1p(actual_times)

    ax_log.plot(
        log_times,
        marker='o',
        color='#2ca02c',
        linestyle='-',
        alpha=0.7,
        markersize=4,
    )
    ax_log.set_title(
        'Log-Transformed Target Space (Stabilized Variance)',
        fontsize=14,
        fontweight='bold',
        pad=12,
    )
    ax_log.set_xlabel('Query Index (Sorted)', fontsize=12, fontweight='bold')
    ax_log.set_ylabel('Log(Execution Time + 1)', fontsize=12, fontweight='bold')
    
    for label in ax_log.get_xticklabels() + ax_log.get_yticklabels():
        label.set_fontsize(11)
        label.set_fontweight('bold')

plt.tight_layout()
save_fig('fig2b_log_transformed_space')
plt.close()
print('Figure 2b (Log-Transformed Space) generated.')


# ---------------------------------------------------------------------------
# FIGURE 3: PREDICTED VS ACTUAL LATENCY
# ---------------------------------------------------------------------------
fig, (ax_mlp, ax_gnn) = plt.subplots(1, 2, figsize=(14, 6))

if (
    df_mlp is not None
    and 'Predicted_Execution_Time_Seconds' in df_mlp.columns
):
  pred_mlp = pd.to_numeric(
      df_mlp['Predicted_Execution_Time_Seconds'], errors='coerce'
  )
  act_mlp = pd.to_numeric(
      df_mlp['Actual_Execution_Time_Seconds'], errors='coerce'
  )

  ax_mlp.scatter(
      pred_mlp,
      act_mlp,
      color=PLOT_STYLE['color_mlp_alt'],
      alpha=PLOT_STYLE['alpha_base'],
      edgecolors=PLOT_STYLE['edge_color'],
      s=PLOT_STYLE['size'],
      label='Baseline MLP',
  )
  max_val = max(pred_mlp.max(), act_mlp.max())
  ax_mlp.plot(
      [0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Optimal Alignment'
  )
  ax_mlp.set_title(
      'Baseline MLP: Predicted vs. Actual Latency',
      fontsize=12,
      fontweight='bold',
  )
  ax_mlp.set_xlabel('Predicted Time (Seconds)', fontsize=11)
  ax_mlp.set_ylabel('Actual Time (Seconds)', fontsize=11)
  ax_mlp.legend(loc='upper left')

if (
    df_gnn_time is not None
    and 'Predicted_Execution_Time_Seconds' in df_gnn_time.columns
):
  pred_gnn = pd.to_numeric(
      df_gnn_time['Predicted_Execution_Time_Seconds'], errors='coerce'
  )
  act_gnn = pd.to_numeric(
      df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce'
  )

  ax_gnn.scatter(
      pred_gnn,
      act_gnn,
      color=PLOT_STYLE['color_gnn'],
      alpha=PLOT_STYLE['alpha_target'],
      edgecolors=PLOT_STYLE['edge_color'],
      s=PLOT_STYLE['size'],
      label='GreenQO',
  )
  max_val = max(pred_gnn.max(), act_gnn.max())
  ax_gnn.plot(
      [0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Optimal Alignment'
  )
  ax_gnn.set_title(
      'GreenQO: Predicted vs. Actual Latency',
      fontsize=12,
      fontweight='bold',
  )
  ax_gnn.set_xlabel('Predicted Time (Seconds)', fontsize=11)
  ax_gnn.set_ylabel('Actual Time (Seconds)', fontsize=11)
  ax_gnn.legend(loc='upper left')

plt.tight_layout()
save_fig('fig3_predicted_vs_actual')
plt.close()
print('Figure 3 generated.')


# ---------------------------------------------------------------------------
# FIGURE 3B: PREDICTED VS ACTUAL ENERGY
# ---------------------------------------------------------------------------
fig, (ax_mlp_energy, ax_gnn_energy) = plt.subplots(1, 2, figsize=(14, 6))

mlp_pred_e_col = get_col(df_mlp, ['predicted', 'energy']) or get_col(
    df_mlp, ['predicted', 'joule']
)
mlp_act_e_col = get_col(df_mlp, ['actual', 'energy']) or get_col(
    df_mlp, ['actual', 'joule']
)

if df_mlp is not None and mlp_pred_e_col and mlp_act_e_col:
  pred_mlp_e = pd.to_numeric(df_mlp[mlp_pred_e_col], errors='coerce')
  act_mlp_e = pd.to_numeric(df_mlp[mlp_act_e_col], errors='coerce')

  ax_mlp_energy.scatter(
      pred_mlp_e,
      act_mlp_e,
      color=PLOT_STYLE['color_mlp_alt'],
      alpha=PLOT_STYLE['alpha_base'],
      edgecolors=PLOT_STYLE['edge_color'],
      s=PLOT_STYLE['size'],
      label='Baseline MLP',
  )
  max_val_e = max(pred_mlp_e.max(), act_mlp_e.max())
  ax_mlp_energy.plot(
      [0, max_val_e],
      [0, max_val_e],
      'k--',
      linewidth=1.5,
      label='Optimal Alignment',
  )
  ax_mlp_energy.set_title(
      'Baseline MLP: Predicted vs. Actual Energy',
      fontsize=12,
      fontweight='bold',
  )
  ax_mlp_energy.set_xlabel('Predicted Energy (Joules)', fontsize=11)
  ax_mlp_energy.set_ylabel('Actual Energy (Joules)', fontsize=11)
  ax_mlp_energy.legend(loc='upper left')

gnn_pred_e_col = (
    (
        get_col(df_gnn_energy, ['predicted', 'energy'])
        or get_col(df_gnn_energy, ['predicted', 'joule'])
    )
    if df_gnn_energy is not None
    else None
)
gnn_act_e_col = (
    (
        get_col(df_gnn_energy, ['actual', 'energy'])
        or get_col(df_gnn_energy, ['actual', 'joule'])
    )
    if df_gnn_energy is not None
    else None
)

if (
    df_gnn_energy is not None
    and not df_gnn_energy.empty
    and gnn_pred_e_col
    and gnn_act_e_col
):
  pred_gnn_e = pd.to_numeric(df_gnn_energy[gnn_pred_e_col], errors='coerce')
  act_gnn_e = pd.to_numeric(df_gnn_energy[gnn_act_e_col], errors='coerce')

  ax_gnn_energy.scatter(
      pred_gnn_e,
      act_gnn_e,
      color=PLOT_STYLE['color_gnn'],
      alpha=PLOT_STYLE['alpha_target'],
      edgecolors=PLOT_STYLE['edge_color'],
      s=PLOT_STYLE['size'],
      label='GreenQO',
  )
  max_val_gnn_e = max(pred_gnn_e.max(), act_gnn_e.max())
  ax_gnn_energy.plot(
      [0, max_val_gnn_e],
      [0, max_val_gnn_e],
      'k--',
      linewidth=1.5,
      label='Optimal Alignment',
  )
  ax_gnn_energy.set_title(
      'GreenQO: Predicted vs. Actual Energy',
      fontsize=12,
      fontweight='bold',
  )
  ax_gnn_energy.set_xlabel('Predicted Energy (Joules)', fontsize=11)
  ax_gnn_energy.set_ylabel('Actual Energy (Joules)', fontsize=11)
  ax_gnn_energy.legend(loc='upper left')

plt.tight_layout()
save_fig('fig3b_predicted_vs_actual_energy')
plt.close()
print('Figure 3b generated.')

scale_colors = {1: '#1f77b4', 3: '#ff7f0e', 5: '#2ca02c'}
scale_markers = {1: 'o', 3: 's', 5: '^'}


def get_scale_legend_handles():
  return [
      Line2D(
          [0],
          [0],
          marker=scale_markers[1],
          color='w',
          markerfacecolor=scale_colors[1],
          markeredgecolor='none',
          markersize=8,
          alpha=0.8,
          label='1 GB Scale',
      ),
      Line2D(
          [0],
          [0],
          marker=scale_markers[3],
          color='w',
          markerfacecolor=scale_colors[3],
          markeredgecolor='none',
          markersize=8,
          alpha=0.8,
          label='3 GB Scale',
      ),
      Line2D(
          [0],
          [0],
          marker=scale_markers[5],
          color='w',
          markerfacecolor=scale_colors[5],
          markeredgecolor='none',
          markersize=8,
          alpha=0.8,
          label='5 GB Scale',
      ),
      Line2D(
          [0],
          [0],
          color='k',
          linestyle='--',
          linewidth=1.5,
          label='Optimal Alignment',
      ),
  ]


# ---------------------------------------------------------------------------
# FIGURE 3C: MULTI-SCALE LATENCY BREAKDOWN
# ---------------------------------------------------------------------------
fig, (ax_mlp_scale, ax_gnn_scale) = plt.subplots(1, 2, figsize=(14, 6))
scale_col = get_col(df_gnn_time, ['scale']) or 'Scale_GB'
mlp_scale_col = get_col(df_mlp, ['scale']) or 'Scale_GB'

if df_mlp is not None and mlp_scale_col in df_mlp.columns:
  for scale in sorted(df_mlp[mlp_scale_col].dropna().unique()):
    sub_mlp = df_mlp[df_mlp[mlp_scale_col] == scale]
    p_sub = pd.to_numeric(
        sub_mlp['Predicted_Execution_Time_Seconds'], errors='coerce'
    )
    a_sub = pd.to_numeric(
        sub_mlp['Actual_Execution_Time_Seconds'], errors='coerce'
    )
    ax_mlp_scale.scatter(
        p_sub,
        a_sub,
        color=scale_colors.get(int(scale), '#333333'),
        marker=scale_markers.get(int(scale), 'o'),
        alpha=0.8,
        edgecolors=PLOT_STYLE['edge_color'],
        s=70,
    )
  max_m_val = max(
      pd.to_numeric(
          df_mlp['Predicted_Execution_Time_Seconds'], errors='coerce'
      ).max(),
      pd.to_numeric(
          df_mlp['Actual_Execution_Time_Seconds'], errors='coerce'
      ).max(),
  )
  ax_mlp_scale.plot([0, max_m_val], [0, max_m_val], 'k--', linewidth=1.5)

ax_mlp_scale.set_title(
    'Baseline MLP Latency Across Volumes', fontsize=12, fontweight='bold'
)
ax_mlp_scale.set_xlabel('Predicted Latency (Seconds)', fontsize=11)
ax_mlp_scale.set_ylabel('Actual Latency (Seconds)', fontsize=11)
ax_mlp_scale.legend(
    handles=get_scale_legend_handles(), loc='upper left', frameon=True
)

if df_gnn_time is not None and scale_col in df_gnn_time.columns:
  for scale in sorted(df_gnn_time[scale_col].dropna().unique()):
    sub_df = df_gnn_time[df_gnn_time[scale_col] == scale]
    pred_sub = pd.to_numeric(
        sub_df['Predicted_Execution_Time_Seconds'], errors='coerce'
    )
    act_sub = pd.to_numeric(
        sub_df['Actual_Execution_Time_Seconds'], errors='coerce'
    )
    ax_gnn_scale.scatter(
        pred_sub,
        act_sub,
        color=scale_colors.get(int(scale), '#333333'),
        marker=scale_markers.get(int(scale), 'o'),
        alpha=0.8,
        edgecolors=PLOT_STYLE['edge_color'],
        s=70,
    )
  max_scale_val = max(
      pd.to_numeric(
          df_gnn_time['Predicted_Execution_Time_Seconds'], errors='coerce'
      ).max(),
      pd.to_numeric(
          df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce'
      ).max(),
  )
  ax_gnn_scale.plot([0, max_scale_val], [0, max_scale_val], 'k--', linewidth=1.5)

ax_gnn_scale.set_title(
    'GreenQO Latency Across Volumes', fontsize=12, fontweight='bold'
)
ax_gnn_scale.set_xlabel('Predicted Latency (Seconds)', fontsize=11)
ax_gnn_scale.set_ylabel('Actual Latency (Seconds)', fontsize=11)
ax_gnn_scale.legend(
    handles=get_scale_legend_handles(), loc='upper left', frameon=True
)

plt.tight_layout()
save_fig('fig3c_multiscale_latency')
plt.close()
print('Figure 3c generated.')


# ---------------------------------------------------------------------------
# FIGURE 3D: MULTI-SCALE ENERGY BREAKDOWN
# ---------------------------------------------------------------------------
fig, (ax_mlp_escala, ax_gnn_escala) = plt.subplots(1, 2, figsize=(14, 6))


def get_energy_scale_handles():
  return [
      Line2D(
          [0],
          [0],
          marker=scale_markers[1],
          color='w',
          markerfacecolor=scale_colors[1],
          markeredgecolor='none',
          markersize=8,
          alpha=0.8,
          label='1 GB Scale',
      ),
      Line2D(
          [0],
          [0],
          marker=scale_markers[3],
          color='w',
          markerfacecolor=scale_colors[3],
          markeredgecolor='none',
          markersize=8,
          alpha=0.8,
          label='3 GB Scale',
      ),
      Line2D(
          [0],
          [0],
          marker=scale_markers[5],
          color='w',
          markerfacecolor=scale_colors[5],
          markeredgecolor='none',
          markersize=8,
          alpha=0.8,
          label='5 GB Scale',
      ),
      Line2D(
          [0],
          [0],
          color='k',
          linestyle='--',
          linewidth=1.5,
          label='Optimal Alignment',
      ),
  ]


mlp_scale_e_col = get_col(df_mlp, ['scale']) or 'Scale_GB'
if (
    df_mlp is not None
    and mlp_scale_e_col in df_mlp.columns
    and mlp_pred_e_col
    and mlp_act_e_col
):
  for scale in sorted(df_mlp[mlp_scale_e_col].dropna().unique()):
    sub_m_e = df_mlp[df_mlp[mlp_scale_e_col] == scale]
    pe_sub = pd.to_numeric(sub_m_e[mlp_pred_e_col], errors='coerce')
    ae_sub = pd.to_numeric(sub_m_e[mlp_act_e_col], errors='coerce')
    ax_mlp_escala.scatter(
        pe_sub,
        ae_sub,
        color=scale_colors.get(int(scale), '#333333'),
        marker=scale_markers.get(int(scale), 'o'),
        alpha=0.8,
        edgecolors=PLOT_STYLE['edge_color'],
        s=70,
    )
  max_me_val = max(
      pd.to_numeric(df_mlp[mlp_pred_e_col], errors='coerce').max(),
      pd.to_numeric(df_mlp[mlp_act_e_col], errors='coerce').max(),
  )
  ax_mlp_escala.plot([0, max_me_val], [0, max_me_val], 'k--', linewidth=1.5)

ax_mlp_escala.set_title(
    'Baseline MLP Energy Across Volumes', fontsize=12, fontweight='bold'
)
ax_mlp_escala.set_xlabel('Predicted Energy (Joules)', fontsize=11)
ax_mlp_escala.set_ylabel('Actual Energy (Joules)', fontsize=11)
ax_mlp_escala.legend(
    handles=get_energy_scale_handles(), loc='upper left', frameon=True
)

energy_scale_col = get_col(df_gnn_energy, ['scale']) or 'Scale_GB'

if (
    df_gnn_energy is not None
    and not df_gnn_energy.empty
    and energy_scale_col in df_gnn_energy.columns
    and gnn_pred_e_col
    and gnn_act_e_col
):
  for scale in sorted(df_gnn_energy[energy_scale_col].dropna().unique()):
    sub_df_e = df_gnn_energy[df_gnn_energy[energy_scale_col] == scale]
    pred_sub_e = pd.to_numeric(sub_df_e[gnn_pred_e_col], errors='coerce')
    act_sub_e = pd.to_numeric(sub_df_e[gnn_act_e_col], errors='coerce')
    ax_gnn_escala.scatter(
        pred_sub_e,
        act_sub_e,
        color=scale_colors.get(int(scale), '#333333'),
        marker=scale_markers.get(int(scale), 'o'),
        alpha=0.8,
        edgecolors=PLOT_STYLE['edge_color'],
        s=70,
    )
  max_energy_scale_val = max(
      pd.to_numeric(df_gnn_energy[gnn_pred_e_col], errors='coerce').max(),
      pd.to_numeric(df_gnn_energy[gnn_act_e_col], errors='coerce').max(),
  )
  ax_gnn_escala.plot(
      [0, max_energy_scale_val], [0, max_energy_scale_val], 'k--', linewidth=1.5
  )

ax_gnn_escala.set_title(
    'GreenQO Energy Across Volumes', fontsize=12, fontweight='bold'
)
ax_gnn_escala.set_xlabel('Predicted Energy (Joules)', fontsize=11)
ax_gnn_escala.set_ylabel('Actual Energy (Joules)', fontsize=11)
ax_gnn_escala.legend(
    handles=get_energy_scale_handles(), loc='upper left', frameon=True
)

plt.tight_layout()
save_fig('fig3d_multiscale_energy')
plt.close()
print('Figure 3d generated.')


# ---------------------------------------------------------------------------
# FIGURE 4: RESIDUAL ERROR DISTRIBUTION
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

if df_mlp is not None and 'Absolute_Execution_Time_Error' in df_mlp.columns:
  err_mlp = pd.to_numeric(
      df_mlp['Absolute_Execution_Time_Error'], errors='coerce'
  ).dropna()
  ax.hist(
      err_mlp,
      bins=20,
      alpha=0.5,
      color='#ff7f0e',
      label='Baseline MLP Error',
      edgecolor='none',
  )

if (
    df_gnn_time is not None
    and 'Absolute_Execution_Time_Error' in df_gnn_time.columns
):
  err_gnn = pd.to_numeric(
      df_gnn_time['Absolute_Execution_Time_Error'], errors='coerce'
  ).dropna()
  ax.hist(
      err_gnn,
      bins=20,
      alpha=0.5,
      color='#2ca02c',
      label='GreenQO Error',
      edgecolor='none',
  )

ax.set_title(
    'Prediction Error Distribution', fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel('Absolute Execution Time Error (Seconds)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
save_fig('fig4_residual_distribution')
plt.close()
print('Figure 4 generated.')


# ---------------------------------------------------------------------------
# FIGURE 5: PARETO FRONTIER
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

if df_baseline is not None and not df_baseline.empty:
    df_baseline.columns = df_baseline.columns.astype(str).str.replace('\xa0', ' ').str.strip()
    col_t = 'Actual_Execution_Time_Sec'
    col_e = 'Actual_Energy_Consumption_Joules'

    if col_t in df_baseline.columns and col_e in df_baseline.columns:
        base_t = pd.to_numeric(df_baseline[col_t], errors='coerce')
        base_e = pd.to_numeric(df_baseline[col_e], errors='coerce')
        valid_base = pd.DataFrame({'Time': base_t, 'Energy': base_e}).dropna()

        ax.scatter(
            valid_base['Time'], valid_base['Energy'],
            color='#d62728', marker='o',
            alpha=0.6, edgecolors='none', linewidths=0.8,
            s=45, zorder=2
        )

if df_history is not None and not df_history.empty:
    df_history.columns = df_history.columns.astype(str).str.replace('\xa0', ' ').str.strip()
    idx_col = 'Selected_Plan_Index'
    q_col = 'Query_Identifier'
    s_col = 'Dataset_Scale_Gigabytes'

    opt0_t_col = 'Plan_Option_0_Cost_Seconds'
    opt1_t_col = 'Plan_Option_1_Cost_Seconds'
    opt0_e_col = 'Plan_Option_0_Cost_Joules'
    opt1_e_col = 'Plan_Option_1_Cost_Joules'

    df_clean = df_history.copy()
    for c in [opt0_t_col, opt1_t_col, opt0_e_col, opt1_e_col]:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    lookup = df_clean.groupby([q_col, s_col]).agg({
        opt0_t_col: 'max', opt1_t_col: 'max',
        opt0_e_col: 'max', opt1_e_col: 'max'
    }).to_dict('index')

    g_time, g_energy = [], []
    for _, row in df_clean.iterrows():
        q_key = (row[q_col], row[s_col])
        metrics = lookup.get(q_key, {})
        sel_idx = 1 if idx_col in df_clean.columns and pd.notnull(row[idx_col]) and '1' in str(row[idx_col]).strip() else 0

        chosen_t = metrics.get(opt1_t_col if sel_idx == 1 else opt0_t_col, np.nan)
        chosen_e = metrics.get(opt1_e_col if sel_idx == 1 else opt0_e_col, np.nan)

        if pd.notnull(chosen_t) and pd.notnull(chosen_e):
            g_time.append(chosen_t)
            g_energy.append(chosen_e)

    g_df = pd.DataFrame({'Time': g_time, 'Energy': g_energy})
    if not g_df.empty:
        ax.scatter(
            g_df['Time'], g_df['Energy'],
            color='#2ca02c', marker='^',
            alpha=0.8, edgecolors='none', linewidths=0.8,
            s=55, zorder=3
        )

legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', 
           markeredgecolor='none', markersize=6, alpha=1.0, label='Native PostgreSQL'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ca02c', 
           markeredgecolor='none', markersize=6, alpha=1.0, label='GreenQO Selected Plans'),
]

ax.set_title('Latency vs. Total Energy Consumption', fontsize=14, fontweight='bold', pad=8)
ax.set_xlabel('Execution Latency (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Energy (J)', fontsize=12, fontweight='bold')

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight('bold')

ax.legend(handles=legend_handles, loc='upper left', frameon=True, prop={'weight': 'bold', 'size': 11})

plt.tight_layout()
save_fig('fig5_pareto_frontier')
plt.close()
print('Figure 5 generated.')


# ---------------------------------------------------------------------------
# FIGURE 6: GNN OPERATOR IMPORTANCE & ATTENTION WEIGHTS
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

if df_weights is not None and not df_weights.empty:
    df_weights.columns = df_weights.columns.str.strip()
    operator_cols = [
        col
        for col in df_weights.columns
        if col
        in [
            'Seq Scan',
            'Hash Join',
            'Nested Loop',
            'Aggregate',
            'Sort',
            'Other_Operators',
        ]
    ]

    if not operator_cols:
        operator_cols = (
            df_weights.select_dtypes(include=[np.number]).columns.tolist()
        )

    if operator_cols:
        mean_weights = df_weights[operator_cols].mean().sort_values()
        ax.barh(
            mean_weights.index,
            mean_weights.values,
            color='#1f77b4',
            edgecolor='none',
            alpha=0.8,
        )
        ax.set_title(
            'GNN Operator Attention & Feature Importance Weights',
            fontsize=14,
            fontweight='bold',
            pad=12,
        )
        ax.set_xlabel('Average Attention / Weight Magnitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Database Plan Operators', fontsize=12, fontweight='bold')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(11)
            label.set_fontweight('bold')

plt.tight_layout()
save_fig('fig6_gnn_attention_weights')
plt.close()
print('Figure 6 generated.')

# ---------------------------------------------------------------------------
# FIGURE 7: HARDWARE POWER INTENSITY PROFILES
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

if df_baseline is not None and 'Actual_Hardware_Wattage' in df_baseline.columns:
    scale_colors_p = {1: '#1f77b4', 3: '#ff7f0e', 5: '#2ca02c'}
    scales = sorted(df_baseline['Scale_GB'].dropna().unique())

    for s in scales:
        sub = (
            df_baseline[df_baseline['Scale_GB'] == s]
            .sort_values(by='Actual_Hardware_Wattage')
            .reset_index(drop=True)
        )
        wattage = pd.to_numeric(sub['Actual_Hardware_Wattage'], errors='coerce')
        c = scale_colors_p.get(int(s), '#333333')
        ax.plot(
            sub.index + 1, wattage, color=c, linewidth=2.5, label=f'{int(s)}GB Scale'
        )

    avg_w = pd.to_numeric(
        df_baseline['Actual_Hardware_Wattage'], errors='coerce'
    ).mean()
    ax.axhline(
        y=avg_w,
        color='red',
        linestyle='--',
        linewidth=1.2,
        label=f'Average Workload Power Draw ({avg_w:.2f} W)',
    )

# CONTROL X-AXIS TICKS 
ax.set_xticks([1, 5, 10, 15, 20, 22])

# CONTROL Y-AXIS TICKS & LIMITS  stepping by 5W
ax.set_ylim(0, 25)
ax.set_yticks(np.arange(0, 26, 5))

ax.set_title(
    'Power Draw Profiles across Dataset Volumes',
    fontsize=14,
    fontweight='bold',
    pad=12,
)
ax.set_xlabel('Sorted Query Runs (1–22)', fontsize=12, fontweight='bold')
ax.set_ylabel('Measured Power Draw (W)', fontsize=12, fontweight='bold')

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight('bold')

ax.legend(loc='upper left', frameon=True, prop={'weight': 'bold', 'size': 11})
plt.tight_layout()
save_fig('fig7_hardware_power_profiles')
plt.close()
print('Figure 7 generated.')
# ---------------------------------------------------------------------------
# COMBINED FIGURE 3: OVERALL AGGREGATE PERFORMANCE (Latency & Energy)
# ---------------------------------------------------------------------------
fig, (ax_lat, ax_eng) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Latency Subplot
if df_mlp is not None and not df_mlp.empty:
    mlp_p_t = pd.to_numeric(df_mlp['Predicted_Execution_Time_Seconds'], errors='coerce')
    mlp_a_t = pd.to_numeric(df_mlp['Actual_Execution_Time_Seconds'], errors='coerce')
    ax_lat.scatter(mlp_p_t, mlp_a_t, color=PLOT_STYLE['color_mlp_alt'], marker='o',
                   alpha=PLOT_STYLE['alpha_base'], edgecolors=PLOT_STYLE['edge_color'],
                   s=PLOT_STYLE['size'], label='Baseline MLP')

if df_gnn_time is not None and not df_gnn_time.empty:
    gnn_p_t = pd.to_numeric(df_gnn_time['Predicted_Execution_Time_Seconds'], errors='coerce')
    gnn_a_t = pd.to_numeric(df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce')
    ax_lat.scatter(gnn_p_t, gnn_a_t, color=PLOT_STYLE['color_gnn'], marker='^',
                   alpha=PLOT_STYLE['alpha_target'], edgecolors=PLOT_STYLE['edge_color'],
                   s=PLOT_STYLE['size'], label='GreenQO')

max_t = max(ax_lat.get_xlim()[1], ax_lat.get_ylim()[1])
ax_lat.plot([0, max_t], [0, max_t], 'k--', linewidth=1.2, label='Optimal ($y=x$)')
ax_lat.set_title('Overall Execution Latency', fontsize=PLOT_STYLE['font_title'], fontweight='normal')
ax_lat.set_xlabel('Predicted Latency (s)', fontsize=PLOT_STYLE['font_label'])
ax_lat.set_ylabel('Actual Latency (s)', fontsize=PLOT_STYLE['font_label'])
ax_lat.legend(loc='upper left', frameon=True, fontsize=8)

# Energy Subplot
if df_mlp is not None and not df_mlp.empty and mlp_pred_e_col and mlp_act_e_col:
    mlp_p_e = pd.to_numeric(df_mlp[mlp_pred_e_col], errors='coerce')
    mlp_a_e = pd.to_numeric(df_mlp[mlp_act_e_col], errors='coerce')
    ax_eng.scatter(mlp_p_e, mlp_a_e, color=PLOT_STYLE['color_mlp_alt'], marker='o',
                   alpha=PLOT_STYLE['alpha_base'], edgecolors=PLOT_STYLE['edge_color'],
                   s=PLOT_STYLE['size'], label='Baseline MLP')

if df_gnn_energy is not None and not df_gnn_energy.empty and gnn_pred_e_col and gnn_act_e_col:
    gnn_p_e = pd.to_numeric(df_gnn_energy[gnn_pred_e_col], errors='coerce')
    gnn_a_e = pd.to_numeric(df_gnn_energy[gnn_act_e_col], errors='coerce')
    ax_eng.scatter(gnn_p_e, gnn_a_e, color=PLOT_STYLE['color_gnn'], marker='^',
                   alpha=PLOT_STYLE['alpha_target'], edgecolors=PLOT_STYLE['edge_color'],
                   s=PLOT_STYLE['size'], label='GreenQO')

max_e = max(ax_eng.get_xlim()[1], ax_eng.get_ylim()[1])
ax_eng.plot([0, max_e], [0, max_e], 'k--', linewidth=1.2, label='Optimal ($y=x$)')
ax_eng.set_title('Overall Energy Consumption', fontsize=PLOT_STYLE['font_title'], fontweight='normal')
ax_eng.set_xlabel('Predicted Energy (J)', fontsize=PLOT_STYLE['font_label'])
ax_eng.set_ylabel('Actual Energy (J)', fontsize=PLOT_STYLE['font_label'])
ax_eng.legend(loc='upper left', frameon=True, fontsize=8)

plt.tight_layout()
save_fig('fig3_aggregate_overall')
plt.close()

# ---------------------------------------------------------------------------
# INDIVIDUAL FIGURE: LATENCY PERFORMANCE (Larger & Bolder)
# ---------------------------------------------------------------------------
fig, ax_lat = plt.subplots(figsize=(8, 6), dpi=300)

if df_mlp is not None and not df_mlp.empty:
    mlp_p_t = pd.to_numeric(df_mlp['Predicted_Execution_Time_Seconds'], errors='coerce')
    mlp_a_t = pd.to_numeric(df_mlp['Actual_Execution_Time_Seconds'], errors='coerce')
    ax_lat.scatter(mlp_p_t, mlp_a_t, color=PLOT_STYLE['color_mlp_alt'], marker='o',
                   alpha=0.8, s=45, label='Baseline MLP')

if df_gnn_time is not None and not df_gnn_time.empty:
    gnn_p_t = pd.to_numeric(df_gnn_time['Predicted_Execution_Time_Seconds'], errors='coerce')
    gnn_a_t = pd.to_numeric(df_gnn_time['Actual_Execution_Time_Seconds'], errors='coerce')
    ax_lat.scatter(gnn_p_t, gnn_a_t, color=PLOT_STYLE['color_gnn'], marker='^',
                   alpha=0.9, s=45, label='GreenQO')

max_t = max(ax_lat.get_xlim()[1], ax_lat.get_ylim()[1])
ax_lat.plot([0, max_t], [0, max_t], 'k--', linewidth=1.2, label='Optimal (y=x)')

# Increased font sizes and bolding for title and axis labels
ax_lat.set_title('Overall Execution Latency', fontsize=14, fontweight='bold')
ax_lat.set_xlabel('Predicted Latency (s)', fontsize=12, fontweight='bold')
ax_lat.set_ylabel('Actual Latency (s)', fontsize=12, fontweight='bold')

# Increase and bold tick values on x and y axes
for label in ax_lat.get_xticklabels() + ax_lat.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight('bold')

# Bold and increase font size for the legend
ax_lat.legend(loc='upper left', frameon=True, prop={'weight': 'bold', 'size': 11})

plt.tight_layout()
save_fig('fig3_latency_overall')
plt.close()

# ---------------------------------------------------------------------------
# INDIVIDUAL FIGURE: ENERGY PERFORMANCE (Larger & Bolder)
# ---------------------------------------------------------------------------
fig, ax_eng = plt.subplots(figsize=(8, 6), dpi=300)

if df_mlp is not None and not df_mlp.empty and mlp_pred_e_col and mlp_act_e_col:
    mlp_p_e = pd.to_numeric(df_mlp[mlp_pred_e_col], errors='coerce')
    mlp_a_e = pd.to_numeric(df_mlp[mlp_act_e_col], errors='coerce')
    ax_eng.scatter(mlp_p_e, mlp_a_e, color=PLOT_STYLE['color_mlp_alt'], marker='o',
                   alpha=0.8, s=45, label='Baseline MLP')

if df_gnn_energy is not None and not df_gnn_energy.empty and gnn_pred_e_col and gnn_act_e_col:
    gnn_p_e = pd.to_numeric(df_gnn_energy[gnn_pred_e_col], errors='coerce')
    gnn_a_e = pd.to_numeric(df_gnn_energy[gnn_act_e_col], errors='coerce')
    ax_eng.scatter(gnn_p_e, gnn_a_e, color=PLOT_STYLE['color_gnn'], marker='^',
                   alpha=0.9, s=45, label='GreenQO')

max_e = max(ax_eng.get_xlim()[1], ax_eng.get_ylim()[1])
ax_eng.plot([0, max_e], [0, max_e], 'k--', linewidth=1.2, label='Optimal (y=x)')

# Increased font sizes and bolding for title and axis labels
ax_eng.set_title('Overall Energy Consumption', fontsize=14, fontweight='bold')
ax_eng.set_xlabel('Predicted Energy (J)', fontsize=12, fontweight='bold')
ax_eng.set_ylabel('Actual Energy (J)', fontsize=12, fontweight='bold')

# Increase and bold tick values on x and y axes
for label in ax_eng.get_xticklabels() + ax_eng.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontweight('bold')

# Bold and increase font size for the legend
ax_eng.legend(loc='upper left', frameon=True, prop={'weight': 'bold', 'size': 11})

plt.tight_layout()
save_fig('fig3_energy_overall')
plt.close()