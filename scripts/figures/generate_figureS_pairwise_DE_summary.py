#!/usr/bin/env python3
"""
Figure S8 — Pairwise DE gene counts between treatment pairs.
Horizontal bar chart showing number of differentially expressed KO terms
for each pairwise comparison. Justifies monomer pooling (G vs X has
almost no DE genes relative to G vs H or X vs H).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
TABLES_DIR = os.path.join(BASE, 'tables', 'output')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Treatment colors from manuscript
COLORS = {
    'Glucose':       '#6B7C2E',
    'Xylose':        '#C8960C',
    'Hemicellulose': '#7B3F6E',
}

FF = 'Arial'
MM = 1 / 25.4  # mm to inches


def blend_colors(c1, c2, alpha=0.5):
    """Blend two hex colors equally."""
    r1 = np.array(to_rgb(c1))
    r2 = np.array(to_rgb(c2))
    blended = alpha * r1 + (1 - alpha) * r2
    return blended


def load_data():
    """Try to load table; fall back to hardcoded confirmed values."""
    table_path = os.path.join(TABLES_DIR, 'TableS16_pairwise_DE_summary.tsv')
    if not os.path.exists(table_path):
        table_path = os.path.join(TABLES_DIR, 'TableSX_pairwise_DE_summary.tsv')

    if os.path.exists(table_path):
        df = pd.read_csv(table_path, sep='\t')
        print(f"Loaded table from {table_path}")
        print(df.to_string(index=False))
        rows = []
        for _, row in df.iterrows():
            comp = row['comparison']
            n_de = row['n_significant_FDR05']
            total = row['total_KO_tested']
            pct = row['pct_significant']
            rows.append({
                'comparison': comp,
                'n_de': int(n_de),
                'total': int(total),
                'pct': float(pct),
            })
        return rows
    else:
        print("Table not found; using hardcoded confirmed values.")
        return [
            {'comparison': 'Glucose vs Xylose',        'n_de': 0,     'total': 11215, 'pct': 0.0},
            {'comparison': 'Glucose vs Hemicellulose',  'n_de': 2307,  'total': 11215, 'pct': 20.6},
            {'comparison': 'Xylose vs Hemicellulose',   'n_de': 2416,  'total': 11215, 'pct': 21.5},
        ]


def make_figure(data):
    """Create horizontal bar chart of pairwise DE counts."""
    fig_w = 88 * MM
    fig_h = 55 * MM
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Order: G vs X at top (smallest), X vs H at bottom
    # Reverse so top-to-bottom reads naturally
    data_plot = list(reversed(data))

    labels = []
    counts = []
    colors = []
    total_ko = data_plot[0]['total']

    # Define bar colors as blends of the two treatment colors
    pair_colors = {
        'Glucose vs Xylose':       blend_colors(COLORS['Glucose'], COLORS['Xylose']),
        'Glucose vs Hemicellulose': blend_colors(COLORS['Glucose'], COLORS['Hemicellulose']),
        'Xylose vs Hemicellulose':  blend_colors(COLORS['Xylose'], COLORS['Hemicellulose']),
    }

    for d in data_plot:
        labels.append(d['comparison'])
        counts.append(d['n_de'])
        colors.append(pair_colors.get(d['comparison'], '#888888'))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, height=0.55, color=colors, edgecolor='none', zorder=3)

    # Annotate each bar with count and percentage
    for i, (bar, d) in enumerate(zip(bars, data_plot)):
        w = bar.get_width()
        pct = d['pct']
        label_text = f"{d['n_de']:,} ({pct:.1f}%)"
        # Place label outside bar if bar is small
        if w < total_ko * 0.15:
            ax.text(w + total_ko * 0.01, bar.get_y() + bar.get_height() / 2,
                    label_text, ha='left', va='center', fontsize=8, fontfamily=FF)
        else:
            ax.text(w - total_ko * 0.01, bar.get_y() + bar.get_height() / 2,
                    label_text, ha='right', va='center', fontsize=8, fontfamily=FF,
                    color='white', fontweight='bold')

    # Dashed reference line for total KOs tested
    ax.axvline(total_ko, color='#555555', linestyle='--', linewidth=0.8, zorder=2)
    ax.text(total_ko, len(labels) - 0.05, f'  {total_ko:,} KOs tested',
            ha='left', va='bottom', fontsize=8, fontfamily=FF, color='#555555')

    # Axis formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, fontfamily=FF)
    ax.set_xlabel('Number of DE KO terms (FDR < 0.05)', fontsize=10, fontfamily=FF)
    ax.set_xlim(0, total_ko * 1.18)

    # Spine removal and tick styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8, direction='out', length=2.5, width=0.6)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.6)

    plt.tight_layout()

    # Save
    pdf_path = os.path.join(FIGURES_DIR, 'FigureS8_pairwise_DE_summary.pdf')
    png_path = os.path.join(FIGURES_DIR, 'FigureS8_pairwise_DE_summary.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    data = load_data()
    make_figure(data)
