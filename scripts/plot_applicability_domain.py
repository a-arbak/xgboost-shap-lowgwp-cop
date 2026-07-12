import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

FLUID_LABEL = {
    'R134a': 'R-134a', 'R513A': 'R-513A', 'R450A': 'R-450A', 'Tern1': 'Tern-1',
    'R515B': 'R-515B', 'R1234yf': 'R-1234yf',
    'R410A': 'R-410A', 'R32': 'R-32', 'R454B': 'R-454B', 'R452B': 'R-452B',
}
FAMILY_MAP = {
    'R134a': 'R-134a based', 'R513A': 'R-134a based',
    'R450A': 'R-134a based', 'Tern1': 'R-134a based',
    'R1234yf': 'HFO', 'R515B': 'HFO',
    'R32': 'R-32 based', 'R410A': 'R-32 based',
    'R454B': 'R-32 based', 'R452B': 'R-32 based',
}
FAMILY_COLORS = {
    'R-134a based': '#1f77b4',
    'HFO':          '#2ca02c',
    'R-32 based':   '#d62728',
}

# Natural refrigerants outside the training distribution (NIST WebBook values)
NATURALS = {
    'R-744':  {'Tc': 304.13, 'Pc': 7.377,  'omega': 0.224, 'M': 44.01},
    'R-290':  {'Tc': 369.89, 'Pc': 4.251,  'omega': 0.152, 'M': 44.10},
    'R-717':  {'Tc': 405.40, 'Pc': 11.333, 'omega': 0.256, 'M': 17.03},
}

# Manual label offsets (points) to avoid collisions
OFFSETS_A = {  # panel (a): Tc vs Pc
    'R-134a': (-8, 9), 'R-513A': (-46, -5), 'R-450A': (7, 0), 'Tern-1': (0, -14),
    'R-515B': (9, -3), 'R-1234yf': (-28, -14),
    'R-410A': (-44, -12), 'R-32': (4, 5), 'R-454B': (7, 3), 'R-452B': (-44, 4),
}
OFFSETS_B = {  # panel (b): omega vs M
    'R-134a': (2, -13), 'R-513A': (-46, -3), 'R-450A': (7, 1),
    'Tern-1': (-18, -14), 'R-515B': (6, -2), 'R-1234yf': (6, -3),
    'R-410A': (6, 0), 'R-32': (6, -2), 'R-454B': (-12, -14), 'R-452B': (7, 4),
}
NATURAL_OFFSETS_A = {'R-744': (6, 3), 'R-290': (-10, 9), 'R-717': (-34, -12)}
NATURAL_OFFSETS_B = {'R-744': (6, 3), 'R-290': (6, 3), 'R-717': (6, 3)}


def domain_box(ax, xlo, xhi, ylo, yhi):
    pad_x = (xhi - xlo) * 0.05
    pad_y = (yhi - ylo) * 0.05
    rect = mpatches.Rectangle(
        (xlo - pad_x, ylo - pad_y),
        (xhi - xlo) + 2 * pad_x, (yhi - ylo) + 2 * pad_y,
        facecolor='#888888', alpha=0.10,
        edgecolor='#555555', linewidth=1.1, linestyle='--', zorder=1)
    ax.add_patch(rect)


def run():
    X = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    desc = X.groupby('fluid_name')[
        ['Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix']].first()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.7))

    # --- Panel (a): Tc_mix vs Pc_mix ---
    domain_box(ax_a, desc['Tc_mix'].min(), desc['Tc_mix'].max(),
               desc['Pc_mix'].min(), desc['Pc_mix'].max())
    for code, row in desc.iterrows():
        fam = FAMILY_MAP[code]
        lbl = FLUID_LABEL[code]
        ax_a.scatter(row['Tc_mix'], row['Pc_mix'],
                     color=FAMILY_COLORS[fam], s=42, zorder=3,
                     edgecolors='white', linewidth=0.5)
        ax_a.annotate(lbl, (row['Tc_mix'], row['Pc_mix']),
                      textcoords='offset points',
                      xytext=OFFSETS_A.get(lbl, (5, 3)), fontsize=7)
    for name, p in NATURALS.items():
        ax_a.scatter(p['Tc'], p['Pc'], marker='x', color='#333333',
                     s=48, linewidth=1.6, zorder=3)
        ax_a.annotate(name, (p['Tc'], p['Pc']), textcoords='offset points',
                      xytext=NATURAL_OFFSETS_A.get(name, (5, 3)),
                      fontsize=7, color='#333333')
    ax_a.margins(x=0.12, y=0.10)
    ax_a.set_xlabel(r'$T_\mathrm{c,mix}$  [K]', fontsize=9.5)
    ax_a.set_ylabel(r'$P_\mathrm{c,mix}$  [MPa]', fontsize=9.5)
    ax_a.tick_params(labelsize=8.5)
    ax_a.text(0.5, -0.34, '(a)', transform=ax_a.transAxes,
              ha='center', va='top', fontsize=9.5)

    # --- Panel (b): omega_mix vs M_mix ---
    domain_box(ax_b, desc['omega_mix'].min(), desc['omega_mix'].max(),
               desc['M_mix'].min(), desc['M_mix'].max())
    for code, row in desc.iterrows():
        fam = FAMILY_MAP[code]
        lbl = FLUID_LABEL[code]
        ax_b.scatter(row['omega_mix'], row['M_mix'],
                     color=FAMILY_COLORS[fam], s=42, zorder=3,
                     edgecolors='white', linewidth=0.5)
        ax_b.annotate(lbl, (row['omega_mix'], row['M_mix']),
                      textcoords='offset points',
                      xytext=OFFSETS_B.get(lbl, (5, 3)), fontsize=7)
    for name, p in NATURALS.items():
        ax_b.scatter(p['omega'], p['M'], marker='x', color='#333333',
                     s=48, linewidth=1.6, zorder=3)
        ax_b.annotate(name, (p['omega'], p['M']), textcoords='offset points',
                      xytext=NATURAL_OFFSETS_B.get(name, (5, 3)),
                      fontsize=7, color='#333333')
    ax_b.margins(x=0.12, y=0.10)
    ax_b.set_xlabel(r'$\omega_\mathrm{mix}$  [—]', fontsize=9.5)
    ax_b.set_ylabel(r'$M_\mathrm{mix}$  [g mol$^{-1}$]', fontsize=9.5)
    ax_b.tick_params(labelsize=8.5)
    ax_b.text(0.5, -0.34, '(b)', transform=ax_b.transAxes,
              ha='center', va='top', fontsize=9.5)

    # Shared legend
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c,
                          markersize=6, label=f)
               for f, c in FAMILY_COLORS.items()]
    handles.append(plt.Line2D([], [], marker='x', linestyle='',
                              color='#333333', markersize=6, markeredgewidth=1.6,
                              label='Outside domain'))
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG_DIR, 'applicability_domain.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    run()
