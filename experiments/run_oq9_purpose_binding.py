"""
OQ-9: Purpose Binding — technical enforcement beyond policy statements

The SIW trilemma says cross-session monitoring (needed for effectiveness) conflicts
with privacy and decentralization: the Level-2 aggregator holds state that could be
silently repurposed for surveillance. "Purpose binding" must therefore be a
*technical* guarantee, not a policy promise. This module implements and tests a
concrete cryptographic protocol (PB-SIW), using only the Python standard library
(hashlib / hmac / secrets), and demonstrates that it blocks the misuse paths.

Four mechanisms, each answering one tension:

  1. Purpose commitment    C = H(purpose ‖ computation ‖ nonce), published before any
     data is touched. Binds the aggregator to exactly this purpose + computation;
     any later claim of a different purpose fails to match C.
  2. Purpose-bound key      K = HKDF(master, info=purpose). The cross-session state is
     sealed (authenticated) under K. A different purpose derives a different key, so
     the sealed state simply will not open — repurposing is a key-separation failure,
     not a rule violation.
  3. Tamper-evident audit    a hash chain over every access; editing or deleting any
     entry changes the head, so an external auditor detects it.
  4. k-of-n quorum release   the master secret is Shamir-split among n independent
     guardians; Level-2 activation needs k of them. No single party — not even the
     aggregator — can unilaterally surveil. (Decentralizes the *activation* of the
     centralized Level-2 computation.)

Hardware alternative: a TEE (SGX/TDX/SEV) attests the same commitment and runs the
sealed computation in an enclave. The crypto protocol here is the software-only
analogue; the guarantees and limits are the same.

Honest limit: crypto binds the USE of collected state, not its COLLECTION. The
trilemma's privacy cost (Lemma 2b) is unchanged — purpose binding governs what the
aggregator may do, it does not make cross-session monitoring free.
"""

import os
import hashlib
import hmac
import secrets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import siw_style as S

# ─────────────────────────────────────────
# CRYPTO PRIMITIVES (stdlib only — portable)
# ─────────────────────────────────────────
def H(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.digest()


def hkdf(master, info, n=32):
    """RFC-5869-style extract/expand."""
    prk = hmac.new(b'PB-SIW-salt', master, hashlib.sha256).digest()
    okm, t, i = b'', b'', 1
    while len(okm) < n:
        t = hmac.new(prk, t + info.encode() + bytes([i]), hashlib.sha256).digest()
        okm += t; i += 1
    return okm[:n]


def _keystream(key, length):
    ks, i = b'', 0
    while len(ks) < length:
        ks += hashlib.sha256(key + i.to_bytes(8, 'big')).digest(); i += 1
    return ks[:length]


def seal(key, plaintext: bytes):
    """Authenticated seal: keystream XOR + HMAC tag (stdlib AEAD analogue)."""
    ct = bytes(a ^ b for a, b in zip(plaintext, _keystream(key, len(plaintext))))
    tag = hmac.new(key, ct, hashlib.sha256).digest()
    return ct, tag


def unseal(key, ct, tag):
    """Return plaintext, or None if the tag fails (wrong key → repurposing blocked)."""
    if not hmac.compare_digest(tag, hmac.new(key, ct, hashlib.sha256).digest()):
        return None
    return bytes(a ^ b for a, b in zip(ct, _keystream(key, len(ct))))


# ── Shamir k-of-n over a prime field (521-bit Mersenne > 256-bit secret) ──
_PRIME = 2 ** 521 - 1
_MASK256 = (1 << 256) - 1
def shamir_split(secret_int, k, n):
    coeffs = [secret_int] + [secrets.randbelow(_PRIME) for _ in range(k - 1)]
    f = lambda x: sum(c * pow(x, i, _PRIME) for i, c in enumerate(coeffs)) % _PRIME
    return [(x, f(x)) for x in range(1, n + 1)]


def shamir_recover(shares):
    xs, ys = zip(*shares)
    total = 0
    for i in range(len(shares)):
        num, den = 1, 1
        for j in range(len(shares)):
            if i != j:
                num = (num * (-xs[j])) % _PRIME
                den = (den * (xs[i] - xs[j])) % _PRIME
        total = (total + ys[i] * num * pow(den, -1, _PRIME)) % _PRIME
    return total


# ─────────────────────────────────────────
# PROTOCOL SETUP (honest deployment)
# ─────────────────────────────────────────
PURPOSE_OK   = "siw.level2.distributed-attack-detection.v1"
PURPOSE_EVIL = "user.profiling.ad-targeting.v1"
COMPUTATION  = "giant_component_ratio >= phi_c"
K, N         = 3, 5                      # 3-of-5 guardian quorum

master = secrets.token_bytes(32)
nonce  = secrets.token_bytes(16)
commitment = H(PURPOSE_OK, COMPUTATION, nonce)        # published

# Seal the cross-session SIW state under the purpose-bound key
state = b"SIW Level-2 cross-session graph: |C_max|/N trajectory + cluster labels"
K_ok = hkdf(master, PURPOSE_OK)
ct, tag = seal(K_ok, state)

# Split the master secret among 5 guardians (need 3 to reconstruct)
master_int = int.from_bytes(master, 'big')
shares = shamir_split(master_int, K, N)

# Tamper-evident audit chain over accesses
def build_chain(entries):
    head = b'\x00' * 32
    for e in entries:
        head = H(head, e)
    return head

audit_entries = [b"t=0 commit published",
                 b"t=1 quorum 3/5 authorized",
                 b"t=2 unseal under purpose v1",
                 b"t=3 ran giant_component_ratio",
                 b"t=4 emitted risk flag"]
audit_head = build_chain(audit_entries)


# ─────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────
def reconstruct_key(provider_shares, claimed_purpose):
    m = (shamir_recover(provider_shares) & _MASK256).to_bytes(32, 'big')
    return hkdf(m, claimed_purpose)

results = {}   # scenario -> dict of checks

# 1) HONEST: committed purpose, full quorum, intact log
k_ok = reconstruct_key(shares[:K], PURPOSE_OK)
results['Honest\n(committed use)'] = dict(
    commit = (H(PURPOSE_OK, COMPUTATION, nonce) == commitment),
    quorum = True,
    opens  = (unseal(k_ok, ct, tag) is not None),
    audit  = (build_chain(audit_entries) == audit_head),
)

# 2) REPURPOSE: aggregator claims a different purpose (profiling)
k_evil = reconstruct_key(shares[:K], PURPOSE_EVIL)       # full quorum, wrong purpose
results['Repurpose\n(profiling)'] = dict(
    commit = (H(PURPOSE_EVIL, COMPUTATION, nonce) == commitment),   # ≠ published
    quorum = True,
    opens  = (unseal(k_evil, ct, tag) is not None),                 # wrong key → None
    audit  = True,
)

# 3) TAMPER: adversary edits one audit entry
tampered = list(audit_entries); tampered[2] = b"t=2 unseal under purpose PROFILING"
results['Tamper\n(edit audit log)'] = dict(
    commit = True, quorum = True, opens = True,
    audit  = (build_chain(tampered) == audit_head),                 # head changes
)

# 4) SUB-QUORUM: only k-1 guardians collude
sub_shares = shares[:K - 1]                                         # only 2 of 5
m_sub = (shamir_recover(sub_shares) & _MASK256).to_bytes(32, 'big') # wrong master
k_sub = hkdf(m_sub, PURPOSE_OK)
results['Sub-quorum\n(2 of 5 collude)'] = dict(
    commit = True,
    quorum = (len(sub_shares) >= K),                                # False
    opens  = (unseal(k_sub, ct, tag) is not None),                  # wrong master → None
    audit  = True,
)

CHECKS = ['commit', 'quorum', 'opens', 'audit']
CHECK_LABEL = {'commit': 'purpose\ncommit', 'quorum': 'quorum\nk-of-n',
               'opens': 'seal\nopens', 'audit': 'audit\nintact'}

print("=== PB-SIW enforcement (✓ = check passes) ===")
for scen, d in results.items():
    granted = all(d[c] for c in CHECKS)
    flat = scen.replace(chr(10), ' ')
    print(f"  {flat:28s} " + "  ".join(f"{c}={'✓' if d[c] else '✗'}" for c in CHECKS) +
          f"   → {'ACCESS GRANTED (committed use only)' if granted else 'BLOCKED / DETECTED'}")

# Quorum reconstruction curve: does j shares recover the true master?
recon_ok = []
for j in range(1, N + 1):
    got = shamir_recover(shares[:j])
    recon_ok.append(1 if got == master_int else 0)
print(f"\nQuorum: shares needed = {K} of {N}; reconstruct-success by #shares = {recon_ok}")


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.suptitle('OQ-9: Purpose Binding', fontsize=14, fontweight='bold', color=S.TEXT_C)

# ── Panel A: protocol flow ──
ax = axes[0, 0]; S.style_ax(ax); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('PB-SIW protocol: four mechanisms')
stages = [
    (0.84, 'Commit',  f'C = H(purpose ‖ comp ‖ nonce)', S.C_BLUE,   'binds purpose + computation'),
    (0.62, 'Seal',    'state sealed under K=HKDF(purpose)', S.C_TEAL, 'wrong purpose → key fails'),
    (0.40, 'Quorum',  f'master split {K}-of-{N} guardians',  S.C_PURPLE, 'no unilateral activation'),
    (0.18, 'Audit',   'hash chain over every access',       S.C_ORANGE, 'edit → head changes'),
]
for y, name, formula, col, note in stages:
    ax.add_patch(mpatches.FancyBboxPatch((0.06, y - 0.06), 0.34, 0.12,
                 boxstyle='round,pad=0.01', facecolor=col, alpha=0.85, edgecolor='none'))
    ax.text(0.23, y, name, ha='center', va='center', color='white',
            fontsize=11, fontweight='bold')
    ax.text(0.44, y + 0.018, formula, ha='left', va='center', fontsize=8.5, color=S.TEXT_C)
    ax.text(0.44, y - 0.030, note, ha='left', va='center', fontsize=8, color=S.SUBTLE)
for y0, y1 in [(0.78, 0.68), (0.56, 0.46), (0.34, 0.24)]:
    ax.annotate('', xy=(0.23, y1), xytext=(0.23, y0),
                arrowprops=dict(arrowstyle='->', color=S.SUBTLE, lw=1.4))
S.panel_label(ax, 'a')

# ── Panel B: enforcement matrix ──
ax = axes[0, 1]; S.style_ax(ax)
scen_names = list(results.keys())
ny, nx = len(scen_names), len(CHECKS)
for r, scen in enumerate(scen_names):
    for c, chk in enumerate(CHECKS):
        ok = results[scen][chk]
        ax.add_patch(plt.Rectangle((c, ny - 1 - r), 1, 1,
                     facecolor=(S.C_GREEN if ok else S.C_RED), alpha=0.30, edgecolor='white', lw=2))
        ax.text(c + 0.5, ny - 1 - r + 0.5, '✓' if ok else '✗', ha='center', va='center',
                fontsize=15, fontweight='bold', color=(S.C_GREEN if ok else S.C_RED))
ax.set_xlim(0, nx); ax.set_ylim(0, ny)
ax.set_xticks([c + 0.5 for c in range(nx)]); ax.set_xticklabels([CHECK_LABEL[c] for c in CHECKS], fontsize=8.5)
ax.set_yticks([ny - 1 - r + 0.5 for r in range(ny)]); ax.set_yticklabels(scen_names, fontsize=8.5)
ax.tick_params(length=0)
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_title('Enforcement: access needs all four ✓')
S.panel_label(ax, 'b')

# ── Panel C: quorum reconstruction ──
ax = axes[1, 0]; S.style_ax(ax)
xs = list(range(1, N + 1))
cols = [S.C_RED if v == 0 else S.C_GREEN for v in recon_ok]
bars = ax.bar(xs, [1] * N, color=cols, alpha=0.5, edgecolor='none')
ax.bar(xs, recon_ok, color=cols, alpha=0.95, edgecolor='none')
ax.axvline(K - 0.5, color=S.TEXT_C, ls='--', lw=1.4)
ax.text(K - 0.5, 1.05, f'threshold k={K}', ha='center', fontsize=9, color=S.TEXT_C)
ax.set_xticks(xs); ax.set_yticks([0, 1]); ax.set_yticklabels(['fails', 'recovers'])
ax.set_xlabel('guardian shares provided'); ax.set_ylim(0, 1.2)
ax.set_title(f'Master secret recovers only at ≥ {K} of {N} shares')
S.panel_label(ax, 'c')

# ── Panel D: mechanism → guarantee ──
ax = axes[1, 1]; S.style_ax(ax); ax.axis('off')
ax.set_title('What each mechanism guarantees')
rows = [
    ('purpose commitment', 'binds to one declared purpose + computation'),
    ('purpose-bound key',  'other purposes cannot open the state'),
    ('hash-chain audit',   'tampering is externally detectable'),
    ('k-of-n quorum',      'no party surveils unilaterally'),
]
y = 0.86
for mech, guar in rows:
    ax.text(0.03, y, mech, transform=ax.transAxes, fontsize=9.5, color=S.TEXT_C,
            va='top', fontweight='bold')
    ax.text(0.40, y, guar, transform=ax.transAxes, fontsize=9, color=S.SUBTLE, va='top')
    ax.plot([0.03, 0.97], [y - 0.045, y - 0.045], color=S.SPINE_C, lw=0.5,
            transform=ax.transAxes, clip_on=False)
    y -= 0.14
ax.text(0.03, 0.18,
        'TEE alternative: an enclave attests the same commitment and runs the\n'
        'sealed computation in hardware. Limit: crypto binds the USE of collected\n'
        'state, not its collection — the trilemma privacy cost (Lemma 2b) stands.',
        transform=ax.transAxes, fontsize=8.3, color=S.SUBTLE, va='top')
S.panel_label(ax, 'd')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), 'figure_oq9_purpose_binding.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out}")

# ─────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────
print("\n" + "=" * 64)
print("OQ-9 VERDICT: PURPOSE BINDING")
print("=" * 64)
honest_ok = all(results['Honest\n(committed use)'][c] for c in CHECKS)
repurpose_blocked = not all(results['Repurpose\n(profiling)'][c] for c in CHECKS)
tamper_detected = not results['Tamper\n(edit audit log)']['audit']
subquorum_blocked = not all(results['Sub-quorum\n(2 of 5 collude)'][c] for c in CHECKS)
print(f"""
Purpose binding is enforceable cryptographically, not just by policy. A working
stdlib protocol (PB-SIW) demonstrates all four guarantees:

  Honest committed use        : {'ACCESS GRANTED' if honest_ok else 'FAILED'}
  Repurpose to profiling      : {'BLOCKED (key separation + commit mismatch)' if repurpose_blocked else 'LEAKED'}
  Audit-log tampering         : {'DETECTED (hash-chain head changes)' if tamper_detected else 'MISSED'}
  Sub-quorum (2 of {N}) access : {'BLOCKED (cannot reconstruct master)' if subquorum_blocked else 'LEAKED'}

Each SIW-trilemma tension gets a mechanism:
  effectiveness needs Level-2 aggregation → its ACTIVATION is gated by a k-of-n
  quorum (decentralized control of a centralized computation); its USE is bound by
  a purpose commitment + purpose-derived key; its INTEGRITY is auditable.

Honest boundary: this binds what the aggregator may DO with collected state; it
does not reduce the privacy cost of COLLECTING it (Lemma 2b). Purpose binding is a
governance guarantee layered on SIW, not an escape from the trilemma.

  OQ-9: ADDRESSED (concrete cryptographic enforcement + tested misuse paths).
""")
