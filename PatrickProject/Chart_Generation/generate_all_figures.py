"""
Master Figure Generation Script
================================
Generates all paper figures from renamed CSV summary files.

Output structure:
    figures/
    ├── prototype_class/        Fig 3: real vs synthetic bar chart
    ├── textvision_class/       Fig 4: real vs synthetic bar chart
    ├── prototype_attributes/   Fig 5: SCDC/SCDS/SCDT horizontal bars
    ├── textvision_attributes/  Fig 6: TV color + TV size bars
    └── synonym_sensitivity/    Table 1 + appendix error bar plots

Usage:
    python generate_all_figures.py          # Generate all figures
    python generate_all_figures.py --fig 3  # Generate only Figure 3
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CHART_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CHART_DIR, 'figures')

# --- CSV timestamps ---
PROTOTYPE_CLASS_SYNTHETIC_TS = '20260127_130209'
PROTOTYPE_CLASS_KONKLAB_TS = '20260127_125452'
TEXTVISION_CLASS_SYNTHETIC_TS = '20260127_131632'
TEXTVISION_CLASS_KONKLAB_TS = '20260127_131800'

# Attribute CSVs have model-specific timestamps
PROTOTYPE_ATTR_TIMESTAMPS = {
    'cvcl-resnext': '20260118_191557',
    'clip-res': '20260118_191826',
    'siglip': '20260119_175545',
    'dino_s_resnext50': '20260118_192113',
    'dino_resnet50': '20260127_122612',
    'resnext': '20260118_192412',
}

TEXTVISION_COLOR_TIMESTAMPS = {
    'cvcl-resnext': '20260119_180604',
    'clip-res': '20260119_184213',
    'siglip': '20260119_182752',
}

SYNONYM_SENSITIVITY_TS = '20260121_121720'

# --- Model lists ---
PROTOTYPE_MODELS = ['cvcl-resnext', 'dino_s_resnext50', 'dino_resnet50', 'clip-res', 'siglip', 'resnext']
TEXTVISION_MODELS = ['cvcl-resnext', 'clip-res', 'siglip']

MODEL_DISPLAY = {
    'cvcl-resnext': 'CVCL',
    'dino_s_resnext50': 'DINO\n(Infant)',
    'dino_resnet50': 'DINO\n(ImageNet)',
    'clip-res': 'CLIP',
    'siglip': 'SigLIP',
    'resnext': 'ResNeXt',
}

MODEL_DISPLAY_FLAT = {
    'cvcl-resnext': 'CVCL',
    'dino_s_resnext50': 'DINO (Infant)',
    'dino_resnet50': 'DINO (ImageNet)',
    'clip-res': 'CLIP',
    'siglip': 'SigLIP',
    'resnext': 'ResNeXt',
}

# --- Helpers ---

def load_overall_accuracy(prefix, model, timestamp):
    """Load overall accuracy from a class-level summary CSV (has seed/overall rows)."""
    filepath = os.path.join(CHART_DIR, f'{prefix}_{model}_summary_{timestamp}.csv')
    if not os.path.exists(filepath):
        print(f"  Missing: {filepath}")
        return None, None
    df = pd.read_csv(filepath)
    overall = df[df['seed'] == 'overall'].iloc[0]
    return overall['mean_accuracy'] * 100, overall['std_accuracy'] * 100


def load_attribute_accuracy(prefix, model, timestamp):
    """Load mean accuracy from a per-class attribute summary CSV."""
    filepath = os.path.join(CHART_DIR, f'{prefix}_{model}_summary_{timestamp}.csv')
    if not os.path.exists(filepath):
        print(f"  Missing: {filepath}")
        return None, None
    df = pd.read_csv(filepath)
    mean_acc = df['mean_accuracy'].mean() * 100
    # SEM across classes
    std_acc = df['mean_accuracy'].std() * 100 / np.sqrt(len(df))
    return mean_acc, std_acc


# --- Figure 3: Prototype Class Discrimination ---

def figure_3():
    """Grouped bar: Prototype class accuracy, Synthetic vs Real (6 models)."""
    print("\n--- Figure 3: Prototype Class Discrimination ---")
    synthetic_means, synthetic_stds = [], []
    konklab_means, konklab_stds = [], []
    labels = []

    for model in PROTOTYPE_MODELS:
        s_mean, s_std = load_overall_accuracy('prototype_class', model, PROTOTYPE_CLASS_SYNTHETIC_TS)
        k_mean, k_std = load_overall_accuracy('prototype_class_konklab', model, PROTOTYPE_CLASS_KONKLAB_TS)

        if s_mean is not None and k_mean is not None:
            synthetic_means.append(s_mean)
            synthetic_stds.append(s_std if s_std else 0)
            konklab_means.append(k_mean)
            konklab_stds.append(k_std if k_std else 0)
            labels.append(MODEL_DISPLAY[model])

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_synth = ax.bar(x - width/2, synthetic_means, width, yerr=synthetic_stds, capsize=4,
                        label='Synthetic', color='#3498db', edgecolor='black', linewidth=0.5)
    bars_real = ax.bar(x + width/2, konklab_means, width, yerr=konklab_stds, capsize=4,
                       label='Real (KonkleLab)', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.axhline(y=25, color='grey', linestyle='--', alpha=0.7, linewidth=1.5, label='Chance (25%)')

    for bar in bars_synth:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_real:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Prototype Class Discrimination: Synthetic vs Real', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=12, framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.join(FIGURES_DIR, 'prototype_class')
    plt.savefig(os.path.join(out_dir, 'prototype_class_real_vs_synthetic.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'prototype_class_real_vs_synthetic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_dir}")


# --- Figure 4: TextVision Class Discrimination ---

def figure_4():
    """Grouped bar: TextVision class accuracy, Synthetic vs Real (3 models)."""
    print("\n--- Figure 4: TextVision Class Discrimination ---")
    synthetic_means, synthetic_stds = [], []
    konklab_means, konklab_stds = [], []
    labels = []

    for model in TEXTVISION_MODELS:
        s_mean, s_std = load_overall_accuracy('textvision_class', model, TEXTVISION_CLASS_SYNTHETIC_TS)
        k_mean, k_std = load_overall_accuracy('textvision_class_konklab', model, TEXTVISION_CLASS_KONKLAB_TS)

        if s_mean is not None and k_mean is not None:
            synthetic_means.append(s_mean)
            synthetic_stds.append(s_std if s_std else 0)
            konklab_means.append(k_mean)
            konklab_stds.append(k_std if k_std else 0)
            labels.append(MODEL_DISPLAY[model])

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_synth = ax.bar(x - width/2, synthetic_means, width, yerr=synthetic_stds, capsize=4,
                        label='Synthetic', color='#3498db', edgecolor='black', linewidth=0.5)
    bars_real = ax.bar(x + width/2, konklab_means, width, yerr=konklab_stds, capsize=4,
                       label='Real (KonkleLab)', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.axhline(y=25, color='grey', linestyle='--', alpha=0.7, linewidth=1.5, label='Chance (25%)')

    for bar in bars_synth:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_real:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Text-Vision Class Discrimination: Synthetic vs Real', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=12, framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.join(FIGURES_DIR, 'textvision_class')
    plt.savefig(os.path.join(out_dir, 'textvision_class_real_vs_synthetic.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'textvision_class_real_vs_synthetic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_dir}")


# --- Figure 5: Prototype Attribute Discrimination ---

def figure_5():
    """Horizontal bar: Prototype attribute discrimination (Color/Size/Texture x 6 models)."""
    print("\n--- Figure 5: Prototype Attribute Discrimination ---")
    attributes = [
        ('Color (SCDC)', 'prototype_color'),
        ('Size (SCDS)', 'prototype_size'),
        ('Texture (SCDT)', 'prototype_texture'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, (attr_label, prefix) in enumerate(attributes):
        ax = axes[ax_idx]
        means, stds, labels = [], [], []

        for model in PROTOTYPE_MODELS:
            ts = PROTOTYPE_ATTR_TIMESTAMPS.get(model)
            if ts is None:
                continue
            m, s = load_attribute_accuracy(prefix, model, ts)
            if m is not None:
                means.append(m)
                stds.append(s)
                labels.append(MODEL_DISPLAY_FLAT[model])

        y = np.arange(len(labels))
        colors = ['#E74C3C', '#9B59B6', '#8E44AD', '#3498DB', '#2ECC71', '#F39C12']

        ax.barh(y, means, xerr=stds, capsize=3, color=colors[:len(labels)],
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=25, color='grey', linestyle='--', alpha=0.7, linewidth=1.5)

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + 1.5, i, f'{m:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title(attr_label, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Prototype Attribute Discrimination (4-Way Forced Choice)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(FIGURES_DIR, 'prototype_attributes')
    plt.savefig(os.path.join(out_dir, 'prototype_attributes.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'prototype_attributes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_dir}")


# --- Figure 6: TextVision Attribute Discrimination ---

def figure_6():
    """Horizontal bar: TextVision attribute discrimination (Color/Size x 3 models)."""
    print("\n--- Figure 6: TextVision Attribute Discrimination ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Color
    ax = axes[0]
    means, stds, labels = [], [], []
    for model in TEXTVISION_MODELS:
        ts = TEXTVISION_COLOR_TIMESTAMPS.get(model)
        if ts is None:
            continue
        m, s = load_attribute_accuracy('textvision_color', model, ts)
        if m is not None:
            means.append(m)
            stds.append(s)
            labels.append(MODEL_DISPLAY_FLAT[model])

    y = np.arange(len(labels))
    tv_colors = ['#E74C3C', '#3498DB', '#2ECC71']
    ax.barh(y, means, xerr=stds, capsize=3, color=tv_colors[:len(labels)],
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=25, color='grey', linestyle='--', alpha=0.7, linewidth=1.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + 1.5, i, f'{m:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Color', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    # Size - check if textvision_size CSVs exist
    ax = axes[1]
    means, stds, labels = [], [], []
    for model in TEXTVISION_MODELS:
        # Try to find any textvision_size summary for this model
        pattern = os.path.join(CHART_DIR, f'textvision_size_{model}_summary_*.csv')
        matches = glob.glob(pattern)
        if matches:
            filepath = sorted(matches)[-1]  # most recent
            df = pd.read_csv(filepath)
            m = df['mean_accuracy'].mean() * 100
            s = df['mean_accuracy'].std() * 100 / np.sqrt(len(df))
            means.append(m)
            stds.append(s)
            labels.append(MODEL_DISPLAY_FLAT[model])

    if means:
        y = np.arange(len(labels))
        ax.barh(y, means, xerr=stds, capsize=3, color=tv_colors[:len(labels)],
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=25, color='grey', linestyle='--', alpha=0.7, linewidth=1.5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + 1.5, i, f'{m:.1f}%', va='center', fontsize=10, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No textvision_size\nCSVs found yet.\nRun run_textvision_size.py first.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, style='italic')

    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Size', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Text-Vision Attribute Discrimination (4-Way Forced Choice)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    out_dir = os.path.join(FIGURES_DIR, 'textvision_attributes')
    plt.savefig(os.path.join(out_dir, 'textvision_attributes.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'textvision_attributes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_dir}")


# --- Synonym Sensitivity ---

def synonym_sensitivity():
    """Synonym sensitivity grouped bars + error bars + appendix table."""
    print("\n--- Synonym Sensitivity ---")

    summary_path = os.path.join(CHART_DIR, f'synonym_sensitivity_summary_{SYNONYM_SENSITIVITY_TS}.csv')
    if not os.path.exists(summary_path):
        print(f"  Missing: {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)

    models = ['cvcl-resnext', 'clip-res', 'siglip']
    model_labels = {'cvcl-resnext': 'CVCL', 'clip-res': 'CLIP-RES', 'siglip': 'SigLIP'}
    colors = {'cvcl-resnext': '#E74C3C', 'clip-res': '#3498DB', 'siglip': '#2ECC71'}

    # Combined figure
    fig, ax = plt.subplots(figsize=(12, 6))

    all_synonyms = ['little', 'tiny', 'big', 'huge']
    x = np.arange(len(all_synonyms))
    width = 0.25

    for i, model in enumerate(models):
        means, cis = [], []
        for syn in all_synonyms:
            match = summary_df[(summary_df['model'] == model) & (summary_df['synonym'] == syn)]
            if len(match) > 0:
                means.append(match['mean'].values[0] * 100)
                std_val = match['std'].values[0] if 'std' in match.columns else 0
                n_val = match.iloc[0].get('n_seeds', 3) if 'n_seeds' in match.columns else 3
                cis.append(1.96 * std_val * 100 / np.sqrt(n_val) if n_val > 0 else 0)
            else:
                means.append(0)
                cis.append(0)

        ax.bar(x + i*width, means, width, label=model_labels[model],
               color=colors[model], yerr=cis, capsize=4, error_kw={'linewidth': 1.5})

    ax.axhline(y=25, color='gray', linestyle='--', label='Chance (25%)', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Synonym', fontsize=12)
    ax.set_title('Text-Vision Size Discrimination: Synonym Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['little\n(small)', 'tiny\n(small)', 'big\n(large)', 'huge\n(large)'], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=1.75, color='black', linestyle=':', alpha=0.5)
    ax.text(0.75, 65, 'Small synonyms', ha='center', fontsize=10, style='italic')
    ax.text(2.75, 65, 'Large synonyms', ha='center', fontsize=10, style='italic')
    plt.tight_layout()

    out_dir = os.path.join(FIGURES_DIR, 'synonym_sensitivity')
    plt.savefig(os.path.join(out_dir, 'synonym_sensitivity_combined.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'synonym_sensitivity_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Split figure (small vs large)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (synonym_group, group_label) in enumerate([
        (['little', 'tiny'], 'Small Size Synonyms'),
        (['big', 'huge'], 'Large Size Synonyms'),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(synonym_group))

        for i, model in enumerate(models):
            means, cis = [], []
            for syn in synonym_group:
                match = summary_df[(summary_df['model'] == model) & (summary_df['synonym'] == syn)]
                if len(match) > 0:
                    means.append(match['mean'].values[0] * 100)
                    std_val = match['std'].values[0] if 'std' in match.columns else 0
                    cis.append(1.96 * std_val * 100 / np.sqrt(3))
                else:
                    means.append(0)
                    cis.append(0)

            ax.bar(x + i*width, means, width, label=model_labels[model],
                   color=colors[model], yerr=cis, capsize=5, error_kw={'linewidth': 2})

        ax.axhline(y=25, color='gray', linestyle='--', label='Chance (25%)', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel(f'Synonym for "{synonym_group[0]}"', fontsize=12)
        ax.set_title(group_label, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(synonym_group, fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, 70)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Prompt Sensitivity: Size Synonym Accuracy with 95% CI', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'synonym_sensitivity_errorbars.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'synonym_sensitivity_errorbars.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {out_dir}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Generate all paper figures from CSV summaries')
    parser.add_argument('--fig', type=int, nargs='*', default=None,
                        help='Specific figure(s) to generate (3, 4, 5, 6). Omit for all.')

    args = parser.parse_args()

    os.makedirs(os.path.join(FIGURES_DIR, 'prototype_class'), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, 'textvision_class'), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, 'prototype_attributes'), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, 'textvision_attributes'), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, 'synonym_sensitivity'), exist_ok=True)

    figures = {
        3: figure_3,
        4: figure_4,
        5: figure_5,
        6: figure_6,
        7: synonym_sensitivity,
    }

    targets = args.fig if args.fig else list(figures.keys())

    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)

    for fig_num in targets:
        if fig_num in figures:
            figures[fig_num]()
        else:
            print(f"Unknown figure number: {fig_num}")

    print("\nDone!")


if __name__ == '__main__':
    main()
