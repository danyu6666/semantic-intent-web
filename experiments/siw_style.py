"""
SIW shared figure style — warm cream, clean spines, bold annotations.
Import this in all experiment plotting scripts.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Palette ──────────────────────────────────────────────────────
BG       = '#F2EDE7'
SPINE_C  = '#CCBFB5'
TEXT_C   = '#2A2320'
SUBTLE   = '#8A7E78'

C_BLUE   = '#5B8DB8'
C_ORANGE = '#C9853A'
C_RED    = '#B84848'
C_GREEN  = '#4E8E5A'
C_PURPLE = '#7A68A8'
C_GREY   = '#9A9090'
C_TEAL   = '#4A9090'

PALETTE  = [C_BLUE, C_ORANGE, C_RED, C_GREEN, C_PURPLE, C_TEAL, C_GREY]

# ── Apply global rcParams ─────────────────────────────────────────
mpl.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10.5,
    'axes.facecolor':     BG,
    'figure.facecolor':   BG,
    'axes.edgecolor':     SPINE_C,
    'axes.labelcolor':    TEXT_C,
    'xtick.color':        TEXT_C,
    'ytick.color':        TEXT_C,
    'text.color':         TEXT_C,
    'axes.prop_cycle':    mpl.cycler(color=PALETTE),
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
    'legend.frameon':     False,
    'legend.labelcolor':  TEXT_C,
    'savefig.facecolor':  BG,
    'savefig.dpi':        180,
})

def style_ax(ax):
    """Apply spine and color cleanup to a single axis."""
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_C)
    ax.spines['bottom'].set_color(SPINE_C)
    ax.tick_params(colors=TEXT_C, labelsize=10)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)

def panel_label(ax, label, x=-0.13, y=1.06):
    """Bold uppercase panel label (a, b, c …)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', color=TEXT_C, va='top')

def note(ax, text, x=0.97, y=0.96, ha='right', fontsize=9.5):
    """Subtle annotation box inside axes."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, color=SUBTLE, ha=ha, va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=BG,
                      edgecolor=SPINE_C, alpha=0.9))

def bar_labels(ax, bars, fmt='{:.1f}', offset_frac=0.02, horizontal=False):
    """Annotate bar values directly on each bar."""
    lim = ax.get_xlim() if horizontal else ax.get_ylim()
    span = lim[1] - lim[0]
    for bar in bars:
        if horizontal:
            v = bar.get_width()
            ax.text(v + span * offset_frac,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va='center', ha='left',
                    fontsize=10.5, fontweight='bold', color=TEXT_C)
        else:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + span * offset_frac,
                    fmt.format(v), ha='center', va='bottom',
                    fontsize=10.5, fontweight='bold', color=TEXT_C)
