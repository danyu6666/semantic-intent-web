"""
Export all figures as SVG (in addition to PNG).
Monkey-patches plt.savefig so every save call also writes a .svg sibling.
"""
import subprocess, sys, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt_orig

here = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ('simulation/make_figures.py',          '../simulation'),
    ('experiments/run_dcsbm_analysis.py',   '.'),
    ('experiments/derive_tau_formula.py',   '.'),
    ('experiments/run_oq2_phi_calibration.py', '.'),
    ('experiments/run_oq2_crossmodel.py',   '.'),
    ('experiments/run_oq3_proxy_signals.py','.'),
    ('experiments/run_oq3_ollama_blackbox.py', '.'),
    ('experiments/run_oq5_privacy_utility.py', '.'),
    ('experiments/run_oq6_lemma3_dcsbm.py', '.'),
]

root = os.path.dirname(here)

WRAPPER = '''
import sys, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
sys.path.insert(0, {exp_dir!r})
import siw_style

# Monkey-patch savefig to also write SVG
_orig_savefig = _plt.savefig
def _dual_savefig(fname, **kwargs):
    # PNG (original)
    _orig_savefig(fname, **kwargs)
    # SVG sibling
    svg_name = os.path.splitext(str(fname))[0] + '.svg'
    svg_kwargs = dict(kwargs)
    svg_kwargs.pop('dpi', None)   # SVG ignores dpi
    _orig_savefig(svg_name, format='svg', **svg_kwargs)
    print(f"  SVG: {{svg_name}}")
_plt.savefig = _dual_savefig

__file__ = {script_path!r}
with open(__file__) as _f:
    exec(compile(_f.read(), __file__, 'exec'),
         {{'__file__': __file__, '__name__': '__main__'}})
'''

for rel_script, rel_out in scripts:
    script_path = os.path.join(root, rel_script)
    exp_dir     = os.path.join(root, 'experiments')

    if not os.path.exists(script_path):
        print(f"  skip (not found): {rel_script}")
        continue

    print(f"\n── {rel_script}")
    code = WRAPPER.format(exp_dir=exp_dir, script_path=script_path)

    tmp = os.path.join(exp_dir, '_tmp_svg_export.py')
    with open(tmp, 'w') as f:
        f.write(code)

    result = subprocess.run(
        [sys.executable, tmp],
        cwd=os.path.join(root, os.path.dirname(rel_script)),
        capture_output=True, text=True
    )
    if result.returncode == 0:
        svgs = [l for l in result.stdout.split('\n') if 'SVG:' in l]
        for s in svgs:
            print(' ', s.strip())
        if not svgs:
            print("  ✓ done (no SVG lines detected — check script save path)")
    else:
        lines = result.stderr.strip().split('\n')
        print(f"  ✗ {lines[-1]}")

if os.path.exists(os.path.join(exp_dir, '_tmp_svg_export.py')):
    os.remove(os.path.join(exp_dir, '_tmp_svg_export.py'))

print("\nDone. SVG files saved alongside PNG files.")
