"""
Patch panel d of simulation_results.png with correct values:
  Attack = 37.0  (original simulation result)
  Benign = 21.9
  Difference = +69% denser

Keeps all other user-edited panels (a, b, c) intact.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import siw_style as style

BG     = style.BG
TEXT   = style.TEXT_C
C_BLUE = style.C_BLUE
C_RED  = style.C_RED
C_GREEN= style.C_GREEN

# ── Correct values ──────────────────────────────────────────────
mean_att = 37.0
mean_ben = 21.9
pct_diff = (mean_att - mean_ben) / mean_ben * 100   # +68.5%

# ── Render panel d at same dpi/size as the main figure ──────────
# Main figure: figsize=(16,9), dpi=180 → 2880×1620 at savefig
# User's PNG is 4736×2594 (higher res), so dpi must be ~296
# We'll render at the same dpi as the original saved file.

IMG_PATH = os.path.join(os.path.dirname(__file__), 'simulation_results.png')
orig = Image.open(IMG_PATH)
W, H = orig.size      # 4736 × 2594
DPI  = W / 16         # ≈ 296 dpi used when saving

# Panel d occupies approximately the bottom-right cell.
# GridSpec: left=0.07, right=0.97, top=0.93, bottom=0.10
# hspace=0.45, wspace=0.38  → 2 rows, 3 cols
# Row 1 (bottom), Col 2 (right-most)
# x fraction: right third of [0.07, 0.97]
# The wspace carves gaps; col width ≈ 0.90/3 × (1-wspace_eff) ...
# → Just reproduce the exact subplot and paste over.

# Reproduce at DPI matched to original
fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.07, right=0.97, top=0.93, bottom=0.10,
                         hspace=0.45, wspace=0.38)

ax_d = fig.add_subplot(gs[1, 2])
style.style_ax(ax_d)

labels_d = ['Benign\ncluster', 'Attack\ncluster']
vals_d   = [mean_ben, mean_att]
colors_d = [C_BLUE, C_RED]

bars_d = ax_d.barh(labels_d, vals_d, color=colors_d,
                   height=0.45, alpha=0.88, edgecolor='none')

for bar, v in zip(bars_d, vals_d):
    ax_d.text(v + max(vals_d) * 0.03,
              bar.get_y() + bar.get_height() / 2,
              f'{v:.1f}', va='center', ha='left',
              fontsize=11, fontweight='bold', color=TEXT)

ax_d.text(max(vals_d) * 0.72, 1.32,
          f'+{pct_diff:.0f}% denser', va='center', ha='center',
          fontsize=10, fontweight='bold', color=C_GREEN)

ax_d.set_xlabel('Mean node degree', fontsize=11)
ax_d.set_title('Cluster Density\nat Crystallization', fontsize=12, pad=8)
ax_d.set_xlim(0, max(vals_d) * 1.55)
ax_d.set_ylim(-0.6, 1.7)

# Panel label 'd'
style.panel_label(ax_d, 'd')

# ── Save full figure as reference (all other panels invisible) ───
# We need the figure at the same pixel resolution as original
tmp_path = os.path.join(os.path.dirname(__file__), '_tmp_panel_d.png')
fig.savefig(tmp_path, dpi=DPI, bbox_inches=None, facecolor=BG)
plt.close()

# ── Detect panel d region from the temp image (different dpi OK) ─
tmp_img = Image.open(tmp_path).convert('RGBA')
Wt, Ht  = tmp_img.size    # temp size
tmp_arr = np.array(tmp_img)
bg_rgb  = np.array([242, 237, 231])

not_bg = np.any(tmp_arr[:, :, :3] != bg_rgb, axis=2)
rows   = np.where(not_bg.any(axis=1))[0]
cols   = np.where(not_bg.any(axis=0))[0]

if len(rows) == 0 or len(cols) == 0:
    print("ERROR: could not find panel d — check layout")
    os.remove(tmp_path)
    sys.exit(1)

MARGIN = 6
# Bounding box as FRACTIONS of temp image
r0f = (max(rows[0]  - MARGIN, 0)) / Ht
r1f = (min(rows[-1] + MARGIN, Ht - 1)) / Ht
c0f = (max(cols[0]  - MARGIN, 0)) / Wt
c1f = (min(cols[-1] + MARGIN, Wt - 1)) / Wt

# Convert fractions to pixel coords in the user's (possibly upscaled) image
r0 = int(r0f * H);  r1 = int(r1f * H)
c0 = int(c0f * W);  c1 = int(c1f * W)

print(f"Panel d region (user image): x=[{c0},{c1}]  y=[{r0},{r1}]"
      f"  ({c1-c0}×{r1-r0} px)")

# Crop and resize patch to match user image region exactly
patch_tmp = tmp_img.crop((
    int(c0f * Wt), int(r0f * Ht),
    int(c1f * Wt), int(r1f * Ht)
))
patch = patch_tmp.resize((c1 - c0, r1 - r0), Image.LANCZOS)

result = orig.copy().convert('RGBA')
result.paste(patch, (c0, r0))
result.convert('RGBA').save(IMG_PATH)

print(f"Saved:  {IMG_PATH}")
print(f"Values: Attack=37.0, Benign=21.9, +69% denser")
os.remove(tmp_path)
