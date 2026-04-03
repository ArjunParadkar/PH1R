"""
PRISM-AD Tokamak Visualizer
python3 tokamak_viz.py
"""
import numpy as np
import platform
import matplotlib
_os = platform.system()
if _os == 'Linux':
    matplotlib.use("gtk4agg")
elif _os == 'Darwin':
    matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ── geometry ───────────────────────────────────────────────────────────────────
R0 = 3.0;  rm = 1.0;  kappa = 1.9;  delta = 0.4
ad = np.arcsin(delta)
nu, nv = 40, 40

um = np.linspace(0, 2*np.pi, nu+1)        # +1 closes the toroidal loop
vm = np.linspace(0, 2*np.pi, nv)
U, V = np.meshgrid(um, vm)
Rc = R0 + rm * np.cos(V + ad * np.sin(V))
X  = Rc * np.cos(U);  Y = Rc * np.sin(U);  Z = rm * kappa * np.sin(V)

# ── plasma particles ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(3);  np.random.seed(7)

NB = 32000
vb = np.random.uniform(0, 2*np.pi, NB).astype(np.float32)
ub = np.random.uniform(0, 2*np.pi, NB).astype(np.float32)
rb = (rm * np.sqrt(np.random.uniform(0, 0.83, NB))).astype(np.float32)
ib = RNG.uniform(0.55, 1.0, NB).astype(np.float32)
Rb_c = (R0 + rb * np.cos(vb + ad * np.sin(vb))).astype(np.float32)
_pg = RNG.uniform(0,0.22,NB).astype(np.float32)
_pb = RNG.uniform(0.3,0.75,NB).astype(np.float32)
pink_rgba = np.c_[ib, _pg*ib, _pb*ib, np.zeros(NB)].astype(np.float32)

# 3 tight yellow clumps (vonMises) — NC particles per clump
CLUMPS = [0.5, 2.3, 4.6];  NC = 500
np.random.seed(42)
up = np.concatenate([np.random.vonmises(c, 18, NC) for c in CLUMPS]).astype(np.float32)
vp = np.random.uniform(0, 2*np.pi, len(up)).astype(np.float32)
rp = (rm * np.sqrt(np.random.uniform(0.25, 0.80, len(up)))).astype(np.float32)
NP = len(up)
Rp_c = (R0 + rp * np.cos(vp + ad * np.sin(vp))).astype(np.float32)
yel_rgba = np.zeros((NP, 4), np.float32)
yel_rgba[:, 0] = 1.0;  yel_rgba[:, 1] = 0.97;  yel_rgba[:, 2] = 0.05

# Red instability blobs — same zones as yellow but wider spread + bigger
np.random.seed(99)
ui = np.concatenate([np.random.vonmises(c, 7, 700) for c in CLUMPS]).astype(np.float32)
vi = np.random.uniform(0, 2*np.pi, len(ui)).astype(np.float32)
ri = (rm * np.sqrt(np.random.uniform(0.20, 0.85, len(ui)))).astype(np.float32)
NI = len(ui)
ii = RNG.uniform(0.75, 1.0, NI).astype(np.float32)
Ri_c = (R0 + ri * np.cos(vi + ad * np.sin(vi))).astype(np.float32)
dred_rgba = np.c_[ii*0.92, ii*0.03, ii*0.03, np.zeros(NI)].astype(np.float32)

def xyz(Rc_, u_base, v_, r_, angle):
    u_ = u_base + angle
    return Rc_*np.cos(u_), Rc_*np.sin(u_), r_*kappa*np.sin(v_)

# ── coil geometry ──────────────────────────────────────────────────────────────
N_TORO = 16
v_c = np.linspace(0, 2*np.pi, 120)
u_c = np.linspace(0, 2*np.pi, 250)

toro_xyz = []
for uv in np.linspace(0, 2*np.pi, N_TORO, endpoint=False):
    Rt = R0 + rm * np.cos(v_c + ad * np.sin(v_c))
    toro_xyz.append((Rt*np.cos(uv), Rt*np.sin(uv), rm*kappa*np.sin(v_c)))

polo_xyz = []
for vv in [np.pi/3, -np.pi/3]:
    Rpl = R0 + rm * np.cos(vv + ad * np.sin(vv))
    polo_xyz.append((Rpl*np.cos(u_c), Rpl*np.sin(u_c),
                     np.full(len(u_c), rm*kappa*np.sin(vv))))

# ── figure (top 87% = 3D scene, bottom 13% = buttons) ─────────────────────────
CHALK  = "#0d1117"
BCOLOR = "#0d1f0d"
BHOVER = "#1a3a1a"
fig = plt.figure(figsize=(13, 10), facecolor=CHALK)
ax  = fig.add_axes([0.0, 0.13, 1.0, 0.87], projection='3d')
ax.set_facecolor(CHALK);  fig.patch.set_facecolor(CHALK);  ax.set_axis_off()

fig.text(0.5, 0.975, "PRISM AD DEMO System", ha="center",
         fontsize=16, fontweight="bold", color="white", fontfamily="monospace")

wire = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1,
                         color="white", linewidth=0.35, alpha=0.8)

xb0,yb0,zb0 = xyz(Rb_c,ub,vb,rb,0)
xp0,yp0,zp0 = xyz(Rp_c,up,vp,rp,0)
xi0,yi0,zi0 = xyz(Ri_c,ui,vi,ri,0)

sc_b = ax.scatter(xb0,yb0,zb0, c=pink_rgba, s=2.2, depthshade=False, edgecolors='none', zorder=3)
sc_i = ax.scatter(xi0,yi0,zi0, c=dred_rgba, s=90,  depthshade=False, edgecolors='none', zorder=4)
sc_p = ax.scatter(xp0,yp0,zp0, c=yel_rgba,  s=55,  depthshade=False, edgecolors='none', zorder=5)

tc_lines, th_lines = [], []
for (xt,yt,zt) in toro_xyz:
    c, = ax.plot(xt,yt,zt, color='cyan',    lw=1.6, alpha=0, zorder=6)
    h, = ax.plot(xt,yt,zt, color='#00ffff', lw=6,   alpha=0, zorder=5)
    tc_lines.append(c);  th_lines.append(h)

pc_lines, ph_lines = [], []
for (xpl,ypl,zpl) in polo_xyz:
    c, = ax.plot(xpl,ypl,zpl, color='#ff9900',  lw=1.6, alpha=0, zorder=6)
    h, = ax.plot(xpl,ypl,zpl, color='#ffaa00',  lw=6,   alpha=0, zorder=5)
    pc_lines.append(c);  ph_lines.append(h)

lim  = R0+rm+0.2;  zlim = rm*kappa+0.2
ax.set_xlim(-lim,lim);  ax.set_ylim(-lim,lim);  ax.set_zlim(-zlim,zlim)
ax.set_box_aspect([1,1,(2*zlim)/(2*lim)])
ax.view_init(elev=20, azim=0)

# ── labels ─────────────────────────────────────────────────────────────────────
def mklbl(x,y,txt,col,sz=13):
    return ax.text2D(x,y,txt, transform=ax.transAxes, ha='center',
                     fontsize=sz, fontweight='bold', color=col,
                     fontfamily='monospace', alpha=0)

lbl_plasma   = mklbl(0.87,0.58, "Plasma",                               '#ff88ff')
lbl_toro     = mklbl(0.87,0.70, "Toroidal Coils",                      'cyan')
lbl_polo     = mklbl(0.87,0.33, "Poloidal Coils",                      '#ff9900')
lbl_perturb  = mklbl(0.87,0.46, "Perturbations",                       '#ffff44', 11)
lbl_instab   = mklbl(0.50,0.05, "PRISM-AD  TOPO_FNO\nInstability Predictions",
                     '#ff3300', 11)
lbl_ctrl     = mklbl(0.50,0.05, "Control Algorithm",                   '#44ff88', 13)
lbl_prevented= mklbl(0.50,0.14, "Instability Prevented",               '#88ffcc', 12)
lbl_reactor  = mklbl(0.50,0.05, "Reactor Controlled Over Time",        '#44ffcc', 13)
lbl_fusion   = mklbl(0.50,0.05, "\u2736  Sustained Fusion Reaction  \u2736",
                     '#ff44ff', 20)

# ── state ──────────────────────────────────────────────────────────────────────
FPS = 50.0
def fr(s): return int(s*FPS)

PHASES = ['idle','plasma_in','plasma_hold',
          'toro_in','toro_hold','toro_out',
          'polo_in','polo_hold','polo_out',
          'perturb_in','perturb_hold',
          'instab_in','instab_hold',
          'return_yellow',
          'ctrl_in','ctrl_fade_yellow','instab_prevented','ctrl_out',
          'fast_spin','fusion']

HOLDS = dict(plasma=fr(1.4), toro=fr(.9), polo=fr(.9), perturb=fr(.8),
             instab=fr(0.6), return_yellow=fr(.3), ctrl_in=fr(.4),
             instab_prev=fr(1.2), fast=fr(4.0))

CTRL_TORO={1,5,9,13};  CTRL_POLO={0}

def fresh():
    return dict(phase='idle', timer=0, angle=0.0, spin=0.0, started=False,
                paused=False,
                plasma_a=0.0, toro_a=0.0, polo_a=0.0,
                perturb_a=0.0, instab_a=0.0, ctrl_t=0.0, prevented_t=0.0,
                white_t=0.0, flash_t=0, azim=0.0,
                clump_a=np.zeros(3), clump_tgt=np.zeros(3))

S = fresh()

def nxt():
    i = PHASES.index(S['phase'])
    if i+1 < len(PHASES): S['phase']=PHASES[i+1]; S['timer']=0

def falpha(lbl,a): lbl.set_alpha(float(np.clip(a,0,1)))

def set_toro(a, only=None, bright=False):
    hf = 0.70 if bright else (0.22 if a>0 else 0)
    for i,(c,h) in enumerate(zip(tc_lines,th_lines)):
        aa = a if (only is None or i in only) else 0.0
        c.set_alpha(aa);  h.set_alpha(min(1.0, aa*hf))
        if bright and aa>0: c.set_linewidth(3.0); h.set_linewidth(10)
        elif not bright:    c.set_linewidth(1.6); h.set_linewidth(6)

def set_polo(a, only=None, bright=False):
    hf = 0.70 if bright else (0.22 if a>0 else 0)
    for i,(c,h) in enumerate(zip(pc_lines,ph_lines)):
        aa = a if (only is None or i in only) else 0.0
        c.set_alpha(aa);  h.set_alpha(min(1.0, aa*hf))
        if bright and aa>0: c.set_linewidth(3.0); h.set_linewidth(10)
        elif not bright:    c.set_linewidth(1.6); h.set_linewidth(6)

def reset_all():
    global S; S = fresh()
    pink_rgba[:,3]=0; yel_rgba[:,3]=0; dred_rgba[:,3]=0
    yel_rgba[:,0]=1.0; yel_rgba[:,1]=0.97; yel_rgba[:,2]=0.05
    set_toro(0); set_polo(0)
    for lbl in [lbl_plasma,lbl_toro,lbl_polo,lbl_perturb,lbl_instab,
                lbl_ctrl,lbl_prevented,lbl_reactor,lbl_fusion]:
        falpha(lbl,0)

def jump_to(phase, **overrides):
    """Jump to a phase, setting sensible defaults for alpha values."""
    reset_all()
    defaults = {
        'idle':            dict(),
        'plasma_in':       dict(started=True),
        'toro_in':         dict(started=True, plasma_a=1.0, spin=0.14),
        'perturb_in':      dict(started=True, plasma_a=1.0, spin=0.14),
        'instab_in':       dict(started=True, plasma_a=1.0, perturb_a=1.0, spin=0.14),
        'ctrl_in':         dict(started=True, plasma_a=1.0, perturb_a=1.0, spin=0.14),
        'fast_spin':       dict(started=True, plasma_a=1.0, spin=0.3),
        'fusion':          dict(started=True, plasma_a=1.0, spin=0.2),
    }.get(phase, dict(started=True, plasma_a=1.0))
    defaults.update(overrides)
    for k,v in defaults.items(): S[k]=v
    S['phase'] = phase; S['timer'] = 0
    pink_rgba[:,3] = S['plasma_a']
    yel_rgba[:,3]  = S['perturb_a']

fig.canvas.mpl_connect("figure_enter_event",
    lambda e: (S.update(started=True,phase='plasma_in') if not S['started'] else None))
fig.canvas.mpl_connect("figure_leave_event", lambda e: reset_all())

# ── buttons ────────────────────────────────────────────────────────────────────
BTN_STYLE = dict(color=BCOLOR, hovercolor=BHOVER)

def style_btn(btn):
    btn.label.set_color('white')
    btn.label.set_fontfamily('monospace')
    btn.label.set_fontsize(8.5)

# Pause/Play
ax_pause = fig.add_axes([0.01, 0.015, 0.07, 0.09])
ax_pause.set_facecolor(BCOLOR)
btn_pause = Button(ax_pause, '⏸ PAUSE', **BTN_STYLE)
style_btn(btn_pause)

def on_pause(_):
    S['paused'] = not S['paused']
    btn_pause.label.set_text('▶ PLAY' if S['paused'] else '⏸ PAUSE')
    fig.canvas.draw_idle()

btn_pause.on_clicked(on_pause)

# Phase jump buttons
nav_defs = [
    ('Plasma',       'plasma_in'),
    ('Coils',        'toro_in'),
    ('Perturbations','perturb_in'),
    ('PRISM-AD',     'instab_in'),
    ('Control',      'ctrl_in'),
    ('Fast Spin',    'fast_spin'),
    ('Fusion',       'fusion'),
]
nav_btns = []
bw, bh, bpad = 0.107, 0.09, 0.008
x0 = 0.10
for i,(label,phase) in enumerate(nav_defs):
    axi = fig.add_axes([x0 + i*(bw+bpad), 0.015, bw, bh])
    axi.set_facecolor(BCOLOR)
    btn = Button(axi, label, **BTN_STYLE)
    style_btn(btn)
    btn.on_clicked(lambda _, ph=phase: jump_to(ph))
    nav_btns.append(btn)

# ── animation ──────────────────────────────────────────────────────────────────
def update(_):
    if S['paused']: return

    ph = S['phase']

    if ph == 'idle':
        S['azim'] = (S['azim']+0.4)%360
        ax.view_init(elev=20, azim=S['azim'])
        return

    S['angle'] += S['spin'];  S['timer'] += 1

    # ── phase machine ──────────────────────────────────────────────────────────
    if ph=='plasma_in':
        S['spin']=0.14; S['plasma_a']=min(1.0,S['plasma_a']+0.07)
        if S['plasma_a']>=1.0: nxt()

    elif ph=='plasma_hold':
        if S['timer']>=HOLDS['plasma']: nxt()

    elif ph=='toro_in':
        S['toro_a']=min(1.0,S['toro_a']+0.08)
        if S['toro_a']>=1.0: nxt()
    elif ph=='toro_hold':
        if S['timer']>=HOLDS['toro']: nxt()
    elif ph=='toro_out':
        S['toro_a']=max(0.0,S['toro_a']-0.08)
        if S['toro_a']<=0.0: nxt()

    elif ph=='polo_in':
        S['polo_a']=min(1.0,S['polo_a']+0.08)
        if S['polo_a']>=1.0: nxt()
    elif ph=='polo_hold':
        if S['timer']>=HOLDS['polo']: nxt()
    elif ph=='polo_out':
        S['polo_a']=max(0.0,S['polo_a']-0.08)
        if S['polo_a']<=0.0: nxt()

    elif ph=='perturb_in':
        S['spin']=max(0.022,S['spin']-0.005)
        S['perturb_a']=min(1.0,S['perturb_a']+0.018)
        if S['perturb_a']>=1.0: nxt()
    elif ph=='perturb_hold':
        if S['timer']>=HOLDS['perturb']: nxt()

    elif ph=='instab_in':
        S['spin']=max(0.010,S['spin']-0.002)
        S['instab_a']=min(1.0,S['instab_a']+0.055)   # fast
        if S['instab_a']>=1.0: nxt()
    elif ph=='instab_hold':
        if S['timer']>=HOLDS['instab']: nxt()

    # red fades, back to just yellow
    elif ph=='return_yellow':
        S['instab_a']=max(0.0,S['instab_a']-0.04)
        if S['instab_a']<=0.0: nxt()

    # control algorithm text fades in
    elif ph=='ctrl_in':
        S['ctrl_t']=min(1.0,S['ctrl_t']+0.05)
        if S['timer']>=HOLDS['ctrl_in']: nxt()

    # yellow fades away (control works)
    elif ph=='ctrl_fade_yellow':
        S['perturb_a']=max(0.0,S['perturb_a']-0.025)
        S['spin']=min(0.14,S['spin']+0.002)
        if S['perturb_a']<=0.0: nxt()

    # "instability prevented" holds
    elif ph=='instab_prevented':
        S['prevented_t']=min(1.0,S['prevented_t']+0.06)
        if S['timer']>=HOLDS['instab_prev']: nxt()

    # fade out both labels
    elif ph=='ctrl_out':
        S['ctrl_t']=max(0.0,S['ctrl_t']-0.04)
        S['prevented_t']=max(0.0,S['prevented_t']-0.04)
        if S['ctrl_t']<=0.0: nxt()

    elif ph=='fast_spin':
        S['spin']=min(0.75,S['spin']+0.018)
        S['flash_t']=(S['flash_t']+1)%3    # faster flash cadence
        if S['flash_t']==0:
            for i,(c,h) in enumerate(zip(tc_lines,th_lines)):
                a=float(np.random.choice([0.,0.,1.]))
                c.set_alpha(a);  h.set_alpha(a*0.25)
            for c,h in zip(pc_lines,ph_lines):
                a=float(np.random.choice([0.,1.]))
                c.set_alpha(a);  h.set_alpha(a*0.25)
            # per-clump perturbation flicker — faster, higher probability
            for i in range(3):
                if np.random.random()<0.55:
                    S['clump_tgt'][i]=1.0-S['clump_tgt'][i]
        S['clump_a'] += (S['clump_tgt']-S['clump_a'])*0.22   # faster fade
        if S['timer']>=HOLDS['fast']:
            set_toro(0);set_polo(0);nxt()

    elif ph=='fusion':
        S['clump_a'] *= 0.88   # clumps fade out
        S['white_t'] = min(1.0, S.get('white_t',0) + 0.008)  # plasma whitens

    # re-read after nxt()
    ph = S['phase']
    a  = S['angle']

    # ── plasma positions & colours ─────────────────────────────────────────────
    wt = S.get('white_t', 0.0)
    pink_rgba[:, 0] = ib
    pink_rgba[:, 1] = _pg*ib*(1-wt) + ib*wt
    pink_rgba[:, 2] = _pb*ib*(1-wt) + ib*wt
    pink_rgba[:, 3] = S['plasma_a']
    dred_rgba[:,3] = S['instab_a'] * 0.82

    for i in range(3):
        ca = S['clump_a'][i] if ph in ('fast_spin','fusion') else S['perturb_a']
        yel_rgba[i*NC:(i+1)*NC, 3] = ca
    yel_rgba[:,0]=1.0; yel_rgba[:,1]=0.97; yel_rgba[:,2]=0.05

    sc_b._offsets3d = xyz(Rb_c,ub,vb,rb,a)
    sc_p._offsets3d = xyz(Rp_c,up,vp,rp,a)
    sc_i._offsets3d = xyz(Ri_c,ui,vi,ri,a)
    sc_b.set_facecolors(pink_rgba)
    sc_p.set_facecolors(yel_rgba)
    sc_i.set_facecolors(dred_rgba)

    # ── coils ──────────────────────────────────────────────────────────────────
    if ph in ('toro_in','toro_hold','toro_out'):
        set_toro(S['toro_a'])
    elif ph in ('ctrl_in','ctrl_fade_yellow','instab_prevented','ctrl_out'):
        set_toro(S['toro_a'], only=CTRL_TORO, bright=True)
    elif ph not in ('fast_spin',):
        set_toro(0)

    if ph in ('polo_in','polo_hold','polo_out'):
        set_polo(S['polo_a'])
    elif ph in ('ctrl_in','ctrl_fade_yellow','instab_prevented','ctrl_out'):
        set_polo(S['polo_a'], only=CTRL_POLO, bright=True)
    elif ph not in ('fast_spin',):
        set_polo(0)

    # ── labels ─────────────────────────────────────────────────────────────────
    falpha(lbl_plasma, {'plasma_in':S['plasma_a'],'plasma_hold':1.0,
                        'toro_in':max(0,1.0-S['toro_a']*2)}.get(ph,0))
    falpha(lbl_toro,      S['toro_a']    if ph in ('toro_in','toro_hold','toro_out') else 0)
    falpha(lbl_polo,      S['polo_a']    if ph in ('polo_in','polo_hold','polo_out') else 0)
    falpha(lbl_perturb,   S['perturb_a'] if ph in ('perturb_in','perturb_hold',
                          'instab_in','instab_hold','return_yellow') else 0)
    falpha(lbl_instab,    1.0 if ph == 'instab_hold' else 0)   # snap on/off
    falpha(lbl_ctrl,      S['ctrl_t']    if ph in ('ctrl_in','ctrl_fade_yellow',
                          'instab_prevented','ctrl_out') else 0)
    falpha(lbl_prevented, S['prevented_t'] if ph in ('instab_prevented','ctrl_out') else 0)
    falpha(lbl_reactor,   1.0 if ph in ('fast_spin','settle') else 0)
    if ph=='fusion':
        pulse = 0.70+0.30*np.sin(a*0.5)
        falpha(lbl_fusion, pulse)
        lbl_fusion.set_color((1.0, 0.27+0.73*wt, 1.0))  # whiten tint as plasma whitens
    else:
        falpha(lbl_fusion, 0)

anim = FuncAnimation(fig, update, frames=None,
                     interval=20, blit=False, cache_frame_data=False)
plt.show()
