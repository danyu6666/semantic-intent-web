"""
Re-run all experiment figure scripts with siw_style applied.
Since siw_style sets matplotlib rcParams globally at import,
most style changes (background, spines, fonts) apply automatically.
"""
import subprocess, sys, os

scripts = [
    'run_dcsbm_analysis.py',
    'derive_tau_formula.py',
    'run_oq2_phi_calibration.py',
    'run_oq2_crossmodel.py',
    'run_oq3_proxy_signals.py',
    'run_oq3_ollama_blackbox.py',
    'run_oq5_privacy_utility.py',
    'run_oq6_lemma3_dcsbm.py',
]

# Wrapper: import siw_style THEN exec the target script
here = os.path.dirname(os.path.abspath(__file__))

for script in scripts:
    print(f'\n{"─"*50}')
    print(f'Restyling: {script}')
    script_path = os.path.join(here, script)

    # Write a proper temp wrapper that sets __file__ correctly
    tmp = os.path.join(here, '_tmp_restyle.py')
    with open(tmp, 'w') as f:
        f.write(f"""
import sys, os
__file__ = {repr(script_path)}
sys.path.insert(0, {repr(here)})
import siw_style   # applies rcParams globally before any plotting
with open(__file__) as _src:
    _code = compile(_src.read(), __file__, 'exec')
exec(_code, {{'__file__': __file__, '__name__': '__main__'}})
""")

    result = subprocess.run(
        [sys.executable, tmp],
        cwd=here,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f'  ✓ done')
    else:
        lines = result.stderr.strip().split('\n')
        print(f'  ✗ {lines[-1]}')

# Cleanup
tmp = os.path.join(here, '_tmp_restyle.py')
if os.path.exists(tmp): os.remove(tmp)
print('\nAll figures restyled.')
