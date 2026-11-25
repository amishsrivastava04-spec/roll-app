"""
Roll & Pitch Stiffness / Load Transfer Calculator
- Nonlinear solver: quasi-static 4-corner model (heave + roll + pitch) with wheel-rate / spring + MR +
  bump-stop curves, tyre rate vs load, roll centre q(roll, heave) and pitch centre q(pitch, heave) maps.
- Linear tabs: Guggiani first-order suspension model (eq. 3.143-3.151, 3.157) with unsprung masses,
  and single-axis pitch / anti-dive model.

Run:  streamlit run app.py
"""
import math
import numpy as np

G = 9.81
DEG = math.pi / 180.0


# ======================================================== LINEAR MODEL ==
def mass_props(p):
    """Masses, distributions and heights."""
    mf_c, mr_c = p["m_fc"], p["m_rc"]            # corner masses (kg)
    muf_c, mur_c = p["mu_fc"], p["mu_rc"]        # unsprung corner masses (kg)
    m = 2 * (mf_c + mr_c)
    mu_f, mu_r = 2 * muf_c, 2 * mur_c
    ms = m - mu_f - mu_r
    chi = 2 * mf_c / m                            # total front distribution
    chi_s = (2 * mf_c - mu_f) / ms                # sprung front distribution
    l = p["wb"]
    a = l * (1 - chi)                             # front axle -> CG
    b = l - a
    h = p["h"]
    hs = (m * h - mu_f * p["hu_f"] - mu_r * p["hu_r"]) / ms
    return dict(m=m, mu_f=mu_f, mu_r=mu_r, ms=ms, chi=chi, chi_s=chi_s,
                l=l, a=a, b=b, h=h, hs=hs)


def wheel_rates(p, mp):
    """Wheel-centre rate per corner (N/m) - either direct or from ride frequency."""
    if p["rate_mode"] == "Ride frequency":
        ms_fc = mp["ms"] * mp["chi_s"] / 2
        ms_rc = mp["ms"] * (1 - mp["chi_s"]) / 2
        kride_f = (2 * math.pi * p["f_f"]) ** 2 * ms_fc
        kride_r = (2 * math.pi * p["f_r"]) ** 2 * ms_rc
        kw_f = kride_f * p["kt_f"] / (p["kt_f"] - kride_f)
        kw_r = kride_r * p["kt_r"] / (p["kt_r"] - kride_r)
    else:
        kw_f, kw_r = p["kw_f"], p["kw_r"]
    return kw_f, kw_r


def roll_model(p):
    mp = mass_props(p)
    kw_f, kw_r = wheel_rates(p, mp)
    t1, t2 = p["t_f"], p["t_r"]
    ay = p["ay_g"] * G

    # roll stiffnesses (Nm/rad)  k = k_wheel * t^2 / 2
    ks1 = (kw_f + p["karb_f"]) * t1 ** 2 / 2
    ks2 = (kw_r + p["karb_r"]) * t2 ** 2 / 2
    kp1 = p["kt_f"] * t1 ** 2 / 2
    kp2 = p["kt_r"] * t2 ** 2 / 2
    k1 = ks1 * kp1 / (ks1 + kp1)
    k2 = ks2 * kp2 / (ks2 + kp2)
    K = k1 + k2

    # forces
    Ys = mp["ms"] * ay
    Y1, Y2 = Ys * mp["chi_s"], Ys * (1 - mp["chi_s"])          # sprung, at no-roll centres
    Yu1, Yu2 = mp["mu_f"] * ay, mp["mu_r"] * ay                  # unsprung
    q1, q2 = p["q_f"], p["q_r"]
    qhat = mp["chi_s"] * q1 + (1 - mp["chi_s"]) * q2
    M = Ys * (mp["hs"] - qhat)                                   # roll moment on springs

    # tyre-only moments (links + unsprung)
    A1 = Y1 * q1 + Yu1 * p["hu_f"]
    A2 = Y2 * q2 + Yu2 * p["hu_r"]
    corr = k1 * k2 / K * (A2 / kp2 - A1 / kp1)                   # flexible-tyre correction
    L1 = k1 / K * M + corr                                       # elastic moment front
    L2 = k2 / K * M - corr

    dZ1 = (A1 + L1) / t1
    dZ2 = (A2 + L2) / t2
    phi_p1, phi_p2 = (A1 + L1) / kp1, (A2 + L2) / kp2
    phi_s1, phi_s2 = L1 / ks1, L2 / ks2

    check = (mp["m"] * ay * mp["h"])                             # total moment
    return dict(
        mp=mp, kw_f=kw_f, kw_r=kw_r,
        ks1=ks1 * DEG, ks2=ks2 * DEG, kp1=kp1 * DEG, kp2=kp2 * DEG,
        k1=k1 * DEG, k2=k2 * DEG, K=K * DEG,
        geo1=Y1 * q1 / t1, geo2=Y2 * q2 / t2,
        el1=L1 / t1, el2=L2 / t2,
        uns1=Yu1 * p["hu_f"] / t1, uns2=Yu2 * p["hu_r"] / t2,
        corr1=corr / t1, corr2=-corr / t2,
        dZ1=dZ1, dZ2=dZ2,
        lltd=dZ1 * t1 / (dZ1 * t1 + dZ2 * t2),
        phi_s1=phi_s1 / DEG, phi_s2=phi_s2 / DEG,
        phi_p1=phi_p1 / DEG, phi_p2=phi_p2 / DEG,
        phi=(phi_s1 + phi_p1) / DEG, phi2=(phi_s2 + phi_p2) / DEG,
        grad=(phi_s1 + phi_p1) / DEG / p["ay_g"] if p["ay_g"] else 0.0,
        mom=dZ1 * t1 + dZ2 * t2, check=check, qhat=qhat,
        spring_f=kw_f * p["mr_f"] ** 2, spring_r=kw_r * p["mr_r"] ** 2,
    )


def pitch_model(p):
    mp = mass_props(p)
    kw_f, kw_r = wheel_rates(p, mp)
    l, a, b = mp["l"], mp["a"], mp["b"]
    ax = p["ax_g"] * G
    kf, kr = 2 * kw_f, 2 * kw_r                 # axle wheel rates
    ktf, ktr = 2 * p["kt_f"], 2 * p["kt_r"]     # axle tyre rates

    Ks = l ** 2 * kf * kr / (kf + kr)           # Nm/rad
    Kt = l ** 2 * ktf * ktr / (ktf + ktr)
    Ktot = Ks * Kt / (Ks + Kt)

    dZs = mp["ms"] * ax * mp["hs"] / l
    ad, al = p["anti_dive"], p["anti_lift"]
    geo_f, geo_r = ad * dZs, al * dZs
    el_f, el_r = (1 - ad) * dZs, (1 - al) * dZs
    dZu = (mp["mu_f"] * p["hu_f"] + mp["mu_r"] * p["hu_r"]) * ax / l
    dZ = dZs + dZu
    check = mp["m"] * ax * mp["h"] / l

    sf = el_f / kf * 1000                       # spring comp (mm)
    sr = el_r / kr * 1000
    tf = dZ / ktf * 1000
    tr = dZ / ktr * 1000
    zf = -(sf + tf)                             # front RH change
    zr = sr + tr                                # rear RH change
    theta = math.degrees(math.atan((zr - zf) / 1000 / l))
    heave_cg = zf + (zr - zf) * a / l
    wing = zf - p["x_wing"] * 1000 * math.tan(math.radians(theta))
    margin = p["wing_lim"] + wing
    xr = p["x_wing"] / l
    ad_req = 1 - ((p["wing_lim"] - zr * xr) / (1 + xr) - tf) * kf / (dZs * 1000)
    return dict(
        mp=mp, kw_f=kw_f, kw_r=kw_r,
        Ks=Ks * DEG, Kt=Kt * DEG, Ktot=Ktot * DEG,
        dZs=dZs, geo_f=geo_f, geo_r=geo_r, el_f=el_f, el_r=el_r, dZu=dZu, dZ=dZ, check=check,
        sf=sf, sr=sr, tf=tf, tr=tr, zf=zf, zr=zr, theta=theta,
        grad=theta / p["ax_g"] if p["ax_g"] else 0.0,
        heave_cg=heave_cg, wing=wing, margin=margin, ad_req=ad_req,
    )


DEFAULTS = dict(
    m_fc=84.0, m_rc=74.0, mu_fc=13.0, mu_rc=14.0,
    wb=1.58, t_f=1.26, t_r=1.26, h=0.31, hu_f=0.253, hu_r=0.253,
    q_f=0.049, q_r=0.069,
    rate_mode="Direct wheel rate", f_f=2.135, f_r=2.5,
    kw_f=14439.0040, kw_r=17086.6899, kt_f=110000.0, kt_r=110000.0,
    karb_f=10238.4662, karb_r=0.0, mr_f=1.396551724, mr_r=1.25,
    ay_g=1.5, ax_g=2.0, anti_dive=0.30, anti_lift=0.30, x_wing=0.75, wing_lim=15.0,
)


# ===================================================== NONLINEAR MODEL ==
# Quasi-static 4-corner model. Unknowns: body heave z (mm, up +), body roll phi (deg, outer side down +),
# body pitch theta (deg, nose down +), and the 4 tyre loads N (N).
# Corner order: 0 front-outer, 1 front-inner, 2 rear-outer, 3 rear-inner.
# Wheel travel s (mm, bump +) measured from static ride height.

class RateCurve:
    """Integrates a rate table k(x) into F(x) with F(x0)=0.
    The rate is held flat outside the table, so F is extrapolated linearly."""

    def __init__(self, x, k, x0=0.0, n=4001):
        x = np.asarray(x, float)
        k = np.asarray(k, float)
        o = np.argsort(x)
        x, k = x[o], k[o]
        self.xt, self.kt = x, k
        lo, hi = min(x.min(), x0), max(x.max(), x0)
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        X = np.linspace(lo, hi, n)
        K = np.interp(X, x, k)
        F = np.concatenate([[0.0], np.cumsum(0.5 * (K[1:] + K[:-1]) * np.diff(X))])
        F -= np.interp(x0, X, F)
        self.X, self.Fg, self.klo, self.khi = X, F, K[0], K[-1]

    def F(self, s):
        X, F = self.X, self.Fg
        if s < X[0]:
            return F[0] + self.klo * (s - X[0])
        if s > X[-1]:
            return F[-1] + self.khi * (s - X[-1])
        return float(np.interp(s, X, F))

    def k(self, s):
        return float(np.interp(s, self.xt, self.kt))

    def outside(self, s):
        return len(self.xt) > 1 and (s < self.xt[0] - 1e-6 or s > self.xt[-1] + 1e-6)


class Grid2D:
    """Bilinear table V(angle, heave). Inputs are clamped to the grid edges."""

    def __init__(self, heave, ang, V):
        heave = np.asarray(heave, float)
        ang = np.asarray(ang, float)
        V = np.asarray(V, float).reshape(len(heave), len(ang))
        oh, oa = np.argsort(heave), np.argsort(ang)
        self.h, self.a, self.V = heave[oh], ang[oa], V[oh][:, oa]

    def __call__(self, ang, heave):
        col = [np.interp(heave, self.h, self.V[:, j]) for j in range(len(self.a))]
        return float(np.interp(ang, self.a, col))

    def outside(self, ang, heave):
        oa = len(self.a) > 1 and (ang < self.a[0] - 1e-6 or ang > self.a[-1] + 1e-6)
        oh = len(self.h) > 1 and (heave < self.h[0] - 1e-6 or heave > self.h[-1] + 1e-6)
        return oa or oh


def bump_force(c, cpts, fpts):
    """Bump stop force (N) at compression c (mm). Linear extrapolation past the last point."""
    if c <= 0 or len(cpts) == 0:
        return 0.0
    if len(cpts) > 1 and c > cpts[-1]:
        slope = (fpts[-1] - fpts[-2]) / (cpts[-1] - cpts[-2])
        return float(fpts[-1] + slope * (c - cpts[-1]))
    return float(np.interp(c, cpts, fpts))


class WheelForce:
    """Vertical force at the wheel centre (N) vs wheel travel s (mm, bump +), incl. static load.
    mode 'wheel'  : wheel-rate table (N/mm) vs wheel travel.
    mode 'spring' : spring-rate table vs spring travel, MR (wheel/spring) vs wheel travel,
                    bump stop gap (mm spring/damper travel) and force vs compression table."""

    def __init__(self, cfg, Ws):
        self.mode, self.Ws = cfg["mode"], Ws
        if self.mode == "wheel":
            self.rc = RateCurve(cfg["travel"], cfg["rate"], 0.0)
        else:
            ms_, mr = np.asarray(cfg["mr_travel"], float), np.asarray(cfg["mr"], float)
            o = np.argsort(ms_)
            self.mr_s, self.mr = ms_[o], mr[o]
            self.xc = RateCurve(self.mr_s, 1.0 / self.mr, 0.0)          # damper travel vs wheel travel
            self.spring = RateCurve(cfg["spring_x"], cfg["spring_k"], 0.0)
            self.gap = cfg["gap"]
            bc, bf = np.asarray(cfg["bump_c"], float), np.asarray(cfg["bump_f"], float)
            o = np.argsort(bc)
            bc, bf = bc[o], bf[o]
            if len(bc) == 0 or bc[0] > 0:
                bc, bf = np.concatenate([[0.0], bc]), np.concatenate([[0.0], bf])
            self.bc, self.bf = bc, bf
            self.Fs0 = Ws * float(np.interp(0.0, self.mr_s, self.mr))   # static spring force

    def __call__(self, s):
        """Returns (wheel force N, bump stop force at damper N, damper travel mm, spring force N)."""
        if self.mode == "wheel":
            F = self.Ws + self.rc.F(s)
            return F, 0.0, s, F
        x = self.xc.F(s)
        mr = float(np.interp(s, self.mr_s, self.mr))
        Fb = bump_force(x - self.gap, self.bc, self.bf)
        Fs = self.Fs0 + self.spring.F(x)
        return (Fs + Fb) / mr, Fb, x, Fs

    def rate(self, s, h=0.05):
        return (self(s + h)[0] - self(s - h)[0]) / (2 * h)

    def outside(self, s):
        if self.mode == "wheel":
            return self.rc.outside(s)
        return len(self.mr_s) > 1 and (s < self.mr_s[0] - 1e-6 or s > self.mr_s[-1] + 1e-6)


def nl_build(mp, p, c):
    """Assemble the nonlinear model. mp = mass_props(p); c = curve/config dict."""
    l, chis = mp["l"], mp["chi_s"]
    a_s, b_s = l * (1 - chis), l * chis              # front / rear axle to sprung CG
    t1, t2 = p["t_f"], p["t_r"]
    Wsf, Wsr = (p["m_fc"] - p["mu_fc"]) * G, (p["m_rc"] - p["mu_rc"]) * G
    Wuf, Wur = p["mu_fc"] * G, p["mu_rc"] * G
    N0 = np.array([Wsf + Wuf, Wsf + Wuf, Wsr + Wur, Wsr + Wur])
    wf, wr = WheelForce(c["wheel_f"], Wsf), WheelForce(c["wheel_r"], Wsr)
    tf = RateCurve(c["tyre_f"]["load"], 1.0 / np.asarray(c["tyre_f"]["rate"], float), N0[0])
    tr = RateCurve(c["tyre_r"]["load"], 1.0 / np.asarray(c["tyre_r"]["rate"], float), N0[2])
    return dict(
        mp=mp, l=l, a_s=a_s, b_s=b_s, t1=t1, t2=t2,
        y=np.array([t1 / 2, -t1 / 2, t2 / 2, -t2 / 2]),
        x=np.array([a_s, a_s, -b_s, -b_s]),
        Wu=np.array([Wuf, Wuf, Wur, Wur]), N0=N0,
        wf=[wf, wf, wr, wr], tyre=[tf, tf, tr, tr],
        tyre_tab=[c["tyre_f"]["load"], c["tyre_f"]["load"], c["tyre_r"]["load"], c["tyre_r"]["load"]],
        rcf=Grid2D(*c["rc_f"]), rcr=Grid2D(*c["rc_r"]), pc=Grid2D(*c["pc"]),
        karb_f=c["karb_f"], karb_r=c["karb_r"],
        hu_f=p["hu_f"], hu_r=p["hu_r"],
        FL=c["FL"], xcp=c["aero_bal"] * l - b_s,
        x_wing=c["x_wing"], so2=c["second_order"],
    )


SIDE = np.array([1.0, -1.0, 1.0, -1.0])
AXLE = np.array([1.0, 1.0, -1.0, -1.0])


def nl_eval(u, S, ay_g, ax_g, detail=False):
    z, phd, thd = u[0], u[1], u[2]
    N = np.asarray(u[3:7], float)
    ph, th = phd * DEG, thd * DEG
    ay, ax = ay_g * G, ax_g * G
    mp, l = S["mp"], S["l"]
    ms, chis, hs = mp["ms"], mp["chi_s"], mp["hs"]
    t1, t2 = S["t1"], S["t2"]
    y, x = S["y"], S["x"]

    zb = z - 1000.0 * (ph * y + th * x)                       # body vertical displacement at corner (mm)
    w = np.array([-S["tyre"][i].F(N[i]) for i in range(4)])   # wheel centre displacement (mm)
    s = w - zb                                                # wheel travel (bump +)

    phs_f = (s[0] - s[1]) / (1000 * t1)                       # suspension roll (rad)
    phs_r = (s[2] - s[3]) / (1000 * t2)
    hf, hr = (s[0] + s[1]) / 2, (s[2] + s[3]) / 2             # axle heave (wheel travel, mm)
    ths = (hf - hr) / (1000 * l)                              # suspension pitch (rad)
    hcg = (S["b_s"] * hf + S["a_s"] * hr) / l
    qf = S["rcf"](phs_f / DEG, hf) / 1000                     # m
    qr = S["rcr"](phs_r / DEG, hr) / 1000
    qp = S["pc"](ths / DEG, hcg) / 1000

    out = [S["wf"][i](s[i]) for i in range(4)]
    Fw = np.array([o[0] for o in out])
    arb_f = S["karb_f"] * (s[0] - s[1]) / 2
    arb_r = S["karb_r"] * (s[2] - s[3]) / 2
    Farb = np.array([arb_f, -arb_f, arb_r, -arb_r])
    Fb = Fw + Farb                                            # spring + ARB force on body

    Y1, Y2 = ms * ay * chis, ms * ay * (1 - chis)             # sprung lateral force at each RC
    Gf, Gr = Y1 * qf / t1, Y2 * qr / t2                       # geometric LLT (per wheel +/-)
    Uf = mp["mu_f"] * ay * S["hu_f"] / t1                     # unsprung LLT
    Ur = mp["mu_r"] * ay * S["hu_r"] / t2
    Lg = ms * ax * qp / l                                     # geometric (anti) long. LT
    Lu = (mp["mu_f"] * S["hu_f"] + mp["mu_r"] * S["hu_r"]) * ax / l
    lat = np.array([Gf + Uf, Gf + Uf, Gr + Ur, Gr + Ur])
    Nm = Fb + S["Wu"] + SIDE * lat + AXLE * (Lg + Lu) / 2

    qhat = chis * qf + (1 - chis) * qr
    gr = ms * G * (hs - qhat) * ph if S["so2"] else 0.0
    gp = ms * G * (hs - qp) * th if S["so2"] else 0.0
    r = np.empty(7)
    r[0] = (Fb.sum() - ms * G - S["FL"]) / 1000
    r[1] = (np.sum(Fb * y) + Y1 * qf + Y2 * qr - ms * ay * hs - gr) / 100
    r[2] = (np.sum(Fb * x) - S["FL"] * S["xcp"] + ms * ax * qp - ms * ax * hs - gp) / 100
    r[3:7] = (N - Nm) / 1000
    if not detail:
        return r

    dZf, dZr = (N[0] - N[1]) / 2, (N[2] - N[3]) / 2
    flags = []
    if (N < 0).any():
        flags.append("wheel lift")
    if any(S["wf"][i].outside(s[i]) for i in range(4)):
        flags.append("travel outside spring table")
    if any(N[i] < min(S["tyre_tab"][i]) or N[i] > max(S["tyre_tab"][i]) for i in range(4)):
        flags.append("load outside tyre table")
    if S["rcf"].outside(phs_f / DEG, hf) or S["rcr"].outside(phs_r / DEG, hr):
        flags.append("outside RC grid")
    if S["pc"].outside(ths / DEG, hcg):
        flags.append("outside PC grid")
    if any(o[3] <= 0 for o in out):
        flags.append("spring unloaded")
    zw = z - 1000.0 * th * (S["a_s"] + S["x_wing"])
    d = dict(
        ay_g=ay_g, ax_g=ax_g, comb_g=math.hypot(ay_g, ax_g),
        heave=z, roll=phd, pitch=thd,
        roll_susp_f=phs_f / DEG, roll_susp_r=phs_r / DEG, pitch_susp=ths / DEG,
        N_FO=N[0], N_FI=N[1], N_RO=N[2], N_RI=N[3],
        s_FO=s[0], s_FI=s[1], s_RO=s[2], s_RI=s[3],
        x_FO=out[0][2], x_FI=out[1][2], x_RO=out[2][2], x_RI=out[3][2],
        bump_FO=out[0][1], bump_FI=out[1][1], bump_RO=out[2][1], bump_RI=out[3][1],
        k_FO=S["wf"][0].rate(s[0]), k_FI=S["wf"][1].rate(s[1]),
        k_RO=S["wf"][2].rate(s[2]), k_RI=S["wf"][3].rate(s[3]),
        q_f=qf * 1000, q_r=qr * 1000, q_p=qp * 1000, anti=qp / hs * 100,
        dZ_f=dZf, dZ_r=dZr, geo_f=Gf, geo_r=Gr, uns_f=Uf, uns_r=Ur,
        el_f=dZf - Gf - Uf, el_r=dZr - Gr - Ur,
        lltd=dZf * t1 / (dZf * t1 + dZr * t2) * 100 if abs(ay_g) > 1e-9 else float("nan"),
        dZ_long=(N[0] + N[1]) - (S["N0"][0] + S["N0"][1]),
        geo_long=Lg, uns_long=Lu,
        rh_f=(zb[0] + zb[1]) / 2, rh_r=(zb[2] + zb[3]) / 2, wing=zw,
        flags=", ".join(flags),
    )
    return r, d


def nl_sweep(S, ay_t, ax_t, n=30):
    from scipy.optimize import root
    u = np.concatenate([[0.0, 0.0, 0.0], S["N0"]])
    rows, ok = [], True
    for f in np.linspace(0, 1, n + 1):
        ay_g, ax_g = f * ay_t, f * ax_t
        sol = root(nl_eval, u, args=(S, ay_g, ax_g), method="hybr")
        if not sol.success or np.abs(sol.fun).max() > 1e-6:
            sol = root(nl_eval, u, args=(S, ay_g, ax_g), method="lm")
        conv = sol.success and np.abs(sol.fun).max() < 1e-5
        if conv:
            u = sol.x
        _, d = nl_eval(sol.x, S, ay_g, ax_g, detail=True)
        d["converged"] = bool(conv)
        rows.append(d)
        ok = ok and conv
    return rows, ok


# ================================================================== UI ==
if __name__ == "__main__":
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd

    st.set_page_config(page_title="Roll & Pitch Calculator", layout="wide")
    st.title("Roll & Pitch Stiffness Calculator")
    st.caption("Nonlinear quasi-static solver (heave + roll + pitch) with curve inputs  |  "
               "linear Guggiani roll model  |  linear pitch / anti-dive model")

    # ---------------- helpers ----------------
    def num_df(df):
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        return df.reset_index(drop=True)

    def curve_input(key, default_df, title, help_txt=None):
        """Editable table with optional CSV upload (same columns) and CSV download."""
        st.markdown(f"**{title}**")
        if help_txt:
            st.caption(help_txt)
        up = st.file_uploader("Upload CSV (same columns as the table)", type="csv", key=f"up_{key}")
        df = default_df
        if up is not None:
            try:
                df = pd.read_csv(up)
                if df.shape[1] < default_df.shape[1]:
                    st.error(f"CSV needs {default_df.shape[1]} columns: {', '.join(default_df.columns)}")
                    df = default_df
                else:
                    df = df.iloc[:, :default_df.shape[1]]
                    df.columns = default_df.columns
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = default_df
        ed = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                            key=f"ed_{key}_{up.name if up is not None else 'default'}")
        ed = num_df(ed)
        if ed.empty:
            st.error("Table is empty - using defaults.")
            ed = num_df(default_df)
        st.download_button("Download CSV", ed.to_csv(index=False), f"{key}.csv", key=f"dl_{key}")
        return ed

    def parse_list(txt, fallback):
        try:
            v = sorted({float(t) for t in txt.replace(";", ",").split(",") if t.strip()})
            return v if v else fallback
        except ValueError:
            st.error(f"Could not read '{txt}' - use comma-separated numbers.")
            return fallback

    def grid_input(key, title, ang_label, default_val, help_txt):
        """2D map V(angle, heave): rows = heave (mm, wheel travel bump +), columns = angle (deg)."""
        st.markdown(f"**{title}**")
        if help_txt:
            st.caption(help_txt)
        c1, c2 = st.columns(2)
        a_txt = c1.text_input(f"{ang_label} breakpoints (deg)", "-4, -2, 0, 2, 4", key=f"a_{key}")
        h_txt = c2.text_input("Heave breakpoints (mm)", "-30, -15, 0, 15, 30", key=f"h_{key}")
        angs = parse_list(a_txt, [0.0])
        hvs = parse_list(h_txt, [0.0])
        df = pd.DataFrame([[h] + [default_val] * len(angs) for h in hvs],
                          columns=["heave_mm"] + [f"{a:g}" for a in angs])
        up = st.file_uploader("Upload CSV (first column heave mm, header row = angles deg)",
                              type="csv", key=f"up_{key}")
        if up is not None:
            try:
                df = pd.read_csv(up)
                [float(c) for c in df.columns[1:]]
                df = df.rename(columns={df.columns[0]: "heave_mm"})
            except Exception as e:
                st.error(f"Could not read grid CSV: {e}")
        ed = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                            key=f"ed_{key}_{a_txt}_{h_txt}_{up.name if up is not None else ''}")
        ed = num_df(ed)
        st.download_button("Download CSV", ed.to_csv(index=False), f"{key}.csv", key=f"dl_{key}")
        heave = ed.iloc[:, 0].values
        ang = np.array([float(c) for c in ed.columns[1:]])
        V = ed.iloc[:, 1:].values
        fig = go.Figure()
        for i, h in enumerate(heave):
            fig.add_scatter(x=ang, y=V[i], mode="lines+markers", name=f"heave {h:g} mm")
        fig.update_layout(height=280, xaxis_title=f"{ang_label} (deg)", yaxis_title="Height (mm)",
                          margin=dict(t=20, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        return heave, ang, V

    def spring_input(key, label, kw, mr, Ws):
        st.markdown(f"### {label}")
        mode = st.radio("Input type", ["Wheel rate curve", "Spring + motion ratio + bump stop"],
                        key=f"mode_{key}", horizontal=True)
        if mode == "Wheel rate curve":
            df = curve_input(f"wr_{key}", pd.DataFrame({"travel_mm": [-60.0, -30.0, 0.0, 30.0, 60.0],
                                                        "wheel_rate_Nmm": [round(kw / 1000, 3)] * 5}),
                             "Wheel rate vs wheel travel",
                             "Wheel travel in mm from static ride height, bump +. Include bump stops "
                             "and motion ratio effects in the rate.")
            cfg = dict(mode="wheel", travel=df.iloc[:, 0].values, rate=df.iloc[:, 1].values)
        else:
            mr_df = curve_input(f"mr_{key}", pd.DataFrame({"travel_mm": [-60.0, 0.0, 60.0],
                                                           "MR_wheel_per_spring": [round(mr, 4)] * 3}),
                                "Motion ratio vs wheel travel",
                                "MR = wheel travel / spring travel (same convention as the sheet: "
                                "spring rate = wheel rate x MR^2).")
            sp_df = curve_input(f"sp_{key}", pd.DataFrame({"spring_travel_mm": [-50.0, 0.0, 50.0],
                                                           "spring_rate_Nmm": [round(kw * mr ** 2 / 1000, 3)] * 3}),
                                "Spring rate vs spring travel", "Spring/damper travel from static, compression +.")
            gap = st.number_input("Bump stop gap at static (mm of spring/damper travel)", value=20.0,
                                  step=1.0, key=f"gap_{key}")
            bs_df = curve_input(f"bs_{key}", pd.DataFrame({"compression_mm": [0.0, 5.0, 10.0, 15.0],
                                                           "force_N": [0.0, 300.0, 1000.0, 2500.0]}),
                                "Bump stop force vs compression",
                                "Force at the damper vs bump stop compression. Extrapolated with the last slope.")
            cfg = dict(mode="spring", mr_travel=mr_df.iloc[:, 0].values, mr=mr_df.iloc[:, 1].values,
                       spring_x=sp_df.iloc[:, 0].values, spring_k=sp_df.iloc[:, 1].values, gap=gap,
                       bump_c=bs_df.iloc[:, 0].values, bump_f=bs_df.iloc[:, 1].values)
        try:
            wfo = WheelForce(cfg, Ws)
            ss = np.linspace(-60, 60, 241)
            fig = go.Figure()
            fig.add_scatter(x=ss, y=[wfo.rate(v) for v in ss], name="Wheel rate (N/mm)")
            fig.add_scatter(x=ss, y=[wfo(v)[0] for v in ss], name="Wheel force (N)", yaxis="y2")
            fig.update_layout(height=300, xaxis_title="Wheel travel (mm, bump +)",
                              yaxis_title="Wheel rate (N/mm)",
                              yaxis2=dict(title="Wheel force (N)", overlaying="y", side="right"),
                              margin=dict(t=20, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Curve error: {e}")
        return cfg

    def lines(df, xcol, xlabel, cols, names, ytitle, title, hl=None):
        fig = go.Figure()
        for c, n in zip(cols, names):
            fig.add_scatter(x=df[xcol], y=df[c], name=n, mode="lines")
        for v, lab in (hl or []):
            fig.add_hline(y=v, line_dash="dash", line_color="red", annotation_text=lab)
        fig.update_layout(title=title, height=330, xaxis_title=xlabel, yaxis_title=ytitle,
                          margin=dict(t=40, b=10), legend=dict(orientation="h"))
        return fig

    # ---------------- sidebar inputs ----------------
    sb = st.sidebar
    p = {}
    sb.header("Mass & geometry")
    c1, c2 = sb.columns(2)
    p["m_fc"] = c1.number_input("Front corner mass (kg)", value=DEFAULTS["m_fc"], step=1.0)
    p["m_rc"] = c2.number_input("Rear corner mass (kg)", value=DEFAULTS["m_rc"], step=1.0)
    p["mu_fc"] = c1.number_input("Front unsprung / corner (kg)", value=DEFAULTS["mu_fc"], step=0.5)
    p["mu_rc"] = c2.number_input("Rear unsprung / corner (kg)", value=DEFAULTS["mu_rc"], step=0.5)
    p["wb"] = sb.number_input("Wheelbase (m)", value=DEFAULTS["wb"], step=0.01, format="%.3f")
    p["t_f"] = c1.number_input("Front track (m)", value=DEFAULTS["t_f"], step=0.01, format="%.3f")
    p["t_r"] = c2.number_input("Rear track (m)", value=DEFAULTS["t_r"], step=0.01, format="%.3f")
    p["h"] = sb.number_input("Total CG height (m)", value=DEFAULTS["h"], step=0.005, format="%.3f")
    c1, c2 = sb.columns(2)
    p["hu_f"] = c1.number_input("Front unsprung CG (m)", value=DEFAULTS["hu_f"], step=0.005, format="%.3f")
    p["hu_r"] = c2.number_input("Rear unsprung CG (m)", value=DEFAULTS["hu_r"], step=0.005, format="%.3f")

    sb.header("Linear model: springs & tyres")
    p["rate_mode"] = sb.radio("Wheel rate input", ["Direct wheel rate", "Ride frequency"], horizontal=True)
    c1, c2 = sb.columns(2)
    if p["rate_mode"] == "Ride frequency":
        p["f_f"] = c1.number_input("Front ride freq (Hz)", value=DEFAULTS["f_f"], step=0.05)
        p["f_r"] = c2.number_input("Rear ride freq (Hz)", value=DEFAULTS["f_r"], step=0.05)
        p["kw_f"], p["kw_r"] = DEFAULTS["kw_f"], DEFAULTS["kw_r"]
    else:
        p["kw_f"] = c1.number_input("Front wheel rate (N/m)", value=DEFAULTS["kw_f"], step=500.0)
        p["kw_r"] = c2.number_input("Rear wheel rate (N/m)", value=DEFAULTS["kw_r"], step=500.0)
    p["kt_f"] = c1.number_input("Front tyre rate (N/m)", value=DEFAULTS["kt_f"], step=5000.0)
    p["kt_r"] = c2.number_input("Rear tyre rate (N/m)", value=DEFAULTS["kt_r"], step=5000.0)
    p["mr_f"] = c1.number_input("Front MR (wheel/spring)", value=DEFAULTS["mr_f"], step=0.01, format="%.3f")
    p["mr_r"] = c2.number_input("Rear MR (wheel/spring)", value=DEFAULTS["mr_r"], step=0.01, format="%.3f")
    sb.caption("Mass & geometry are shared by all tabs. The nonlinear solver uses its own curves "
               "(defaults are pre-filled from the linear values above).")

    tab_nl, tab_roll, tab_pitch = st.tabs(["Nonlinear solver", "Roll (linear)", "Pitch / Anti-dive (linear)"])

    # ============================== ROLL TAB ==============================
    with tab_roll:
        ci, co = st.columns([1, 3])
        with ci:
            st.subheader("Roll inputs")
            p["ay_g"] = st.number_input("Lateral acceleration (g)", value=DEFAULTS["ay_g"], step=0.1)
            p["q_f"] = st.number_input("Front roll centre height (m)", value=DEFAULTS["q_f"], step=0.005, format="%.3f")
            p["q_r"] = st.number_input("Rear roll centre height (m)", value=DEFAULTS["q_r"], step=0.005, format="%.3f")
            p["karb_f"] = st.number_input("Front ARB wheel rate (N/m)", value=DEFAULTS["karb_f"], step=500.0)
            p["karb_r"] = st.number_input("Rear ARB wheel rate (N/m)", value=DEFAULTS["karb_r"], step=500.0)

        r = roll_model(p)
        mp = r["mp"]
        with co:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Roll angle", f"{r['phi']:.3f}°")
            m2.metric("Roll gradient", f"{r['grad']:.3f} °/g")
            m3.metric("Front LLTD", f"{r['lltd']*100:.1f} %")
            m4.metric("ΔZ front", f"{r['dZ1']:.1f} N")
            m5.metric("ΔZ rear", f"{r['dZ2']:.1f} N")

            st.markdown("**Roll stiffness (Nm/deg)**")
            st.dataframe(pd.DataFrame({
                "Front": [r["ks1"], r["kp1"], r["k1"]],
                "Rear": [r["ks2"], r["kp2"], r["k2"]],
                "Total": [r["ks1"] + r["ks2"], r["kp1"] + r["kp2"], r["K"]],
            }, index=["Suspension (springs + ARB)", "Tyres", "Series (axle)"]).style.format("{:,.1f}"),
                use_container_width=True)

            st.markdown("**Lateral load transfer (N)**")
            df = pd.DataFrame({
                "Front": [r["geo1"], r["el1"], r["uns1"], r["dZ1"]],
                "Rear": [r["geo2"], r["el2"], r["uns2"], r["dZ2"]],
            }, index=["Geometric (Yᵢqᵢ/tᵢ)", "Elastic (springs)", "Unsprung", "Total"])
            st.dataframe(df.style.format("{:,.2f}"), use_container_width=True)
            st.caption(f"Flexible-tyre correction inside elastic term: {r['corr1']:+.2f} N front / "
                       f"{r['corr2']:+.2f} N rear.  Check ΣΔZ·t = {r['mom']:.2f} Nm vs m·aᵧ·h = {r['check']:.2f} Nm.")

            fig = go.Figure()
            for name, key1, key2 in [("Geometric", "geo1", "geo2"), ("Elastic", "el1", "el2"), ("Unsprung", "uns1", "uns2")]:
                fig.add_bar(name=name, x=["Front", "Rear"], y=[r[key1], r[key2]])
            fig.update_layout(barmode="stack", height=320, yaxis_title="ΔZ (N)", margin=dict(t=30, b=10))

            arbs = np.linspace(0, max(30000, 2 * p["karb_f"]), 41)
            lltd, phi = [], []
            for k in arbs:
                rr = roll_model({**p, "karb_f": k})
                lltd.append(rr["lltd"] * 100); phi.append(rr["phi"])
            fig2 = go.Figure()
            fig2.add_scatter(x=arbs, y=lltd, name="Front LLTD (%)")
            fig2.add_scatter(x=arbs, y=phi, name="Roll angle (°)", yaxis="y2")
            fig2.add_vline(x=p["karb_f"], line_dash="dash")
            fig2.update_layout(height=320, xaxis_title="Front ARB wheel rate (N/m)",
                               yaxis_title="Front LLTD (%)",
                               yaxis2=dict(title="Roll angle (°)", overlaying="y", side="right"),
                               margin=dict(t=30, b=10), legend=dict(orientation="h"))
            g1, g2 = st.columns(2)
            g1.plotly_chart(fig, use_container_width=True)
            g2.plotly_chart(fig2, use_container_width=True)

            with st.expander("Roll angle split & derived values"):
                st.dataframe(pd.DataFrame({
                    "Front": [r["phi_s1"], r["phi_p1"], r["phi"]],
                    "Rear": [r["phi_s2"], r["phi_p2"], r["phi2"]],
                }, index=["Suspension φˢ (°)", "Tyre φᵖ (°)", "Total φ (°)"]).style.format("{:.4f}"),
                    use_container_width=True)
                st.write(f"Sprung mass {mp['ms']:.1f} kg, sprung CG hs = {mp['hs']*1000:.2f} mm, "
                         f"sprung front dist χs = {mp['chi_s']:.4f}, q̂ = {r['qhat']*1000:.2f} mm")
                st.write(f"Wheel rates: F {r['kw_f']:,.1f} N/m, R {r['kw_r']:,.1f} N/m  →  "
                         f"spring rates: F {r['spring_f']:,.0f} N/m, R {r['spring_r']:,.0f} N/m")

    # ============================== PITCH TAB =============================
    with tab_pitch:
        ci, co = st.columns([1, 3])
        with ci:
            st.subheader("Pitch inputs")
            p["ax_g"] = st.number_input("Braking deceleration (g)", value=DEFAULTS["ax_g"], step=0.1)
            p["anti_dive"] = st.slider("Front anti-dive (%)", 0, 150, int(DEFAULTS["anti_dive"] * 100)) / 100
            p["anti_lift"] = st.slider("Rear anti-lift (%)", 0, 150, int(DEFAULTS["anti_lift"] * 100)) / 100
            p["x_wing"] = st.number_input("Front wing ahead of front axle (m)", value=DEFAULTS["x_wing"], step=0.05)
            p["wing_lim"] = st.number_input("Max allowed wing drop (mm)", value=DEFAULTS["wing_lim"], step=1.0)

        pr = pitch_model(p)
        with co:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Pitch angle", f"{pr['theta']:.3f}°")
            m2.metric("Pitch gradient", f"{pr['grad']:.3f} °/g")
            m3.metric("Front wing drop", f"{-pr['wing']:.1f} mm",
                      delta=f"{pr['margin']:+.1f} mm vs limit", delta_color="normal")
            m4.metric("Anti-dive required", f"{pr['ad_req']*100:.1f} %")
            m5.metric("Total ΔZ", f"{pr['dZ']:.1f} N")

            st.markdown("**Pitch stiffness (Nm/deg)**")
            st.dataframe(pd.DataFrame({"Value": [pr["Ks"], pr["Kt"], pr["Ktot"]]},
                                      index=["Springs  l²·kf·kr/(kf+kr)", "Tyres", "Total (series)"]).style.format("{:,.1f}"),
                         use_container_width=True)

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Longitudinal load transfer (N)**")
                st.dataframe(pd.DataFrame({
                    "Front": [pr["geo_f"], pr["el_f"]],
                    "Rear": [pr["geo_r"], pr["el_r"]],
                }, index=["Geometric (anti)", "Elastic (springs)"]).style.format("{:,.1f}"),
                    use_container_width=True)
                st.write(f"Sprung {pr['dZs']:.1f} N + unsprung {pr['dZu']:.1f} N = **{pr['dZ']:.1f} N** "
                         f"(check m·aₓ·h/l = {pr['check']:.1f} N)")
            with g2:
                st.markdown("**Ride height changes (mm)**")
                st.dataframe(pd.DataFrame({
                    "Front": [pr["sf"], pr["tf"], pr["zf"]],
                    "Rear": [pr["sr"], pr["tr"], pr["zr"]],
                }, index=["Spring (comp F / ext R)", "Tyre", "Ride height Δ (at axle)"]).style.format("{:.2f}"),
                    use_container_width=True)
                st.write(f"Heave at CG {pr['heave_cg']:.2f} mm  |  wing height Δ **{pr['wing']:.2f} mm**")

            l = pr["mp"]["l"]; xw = p["x_wing"]
            fig = go.Figure()
            fig.add_scatter(x=[-xw, l], y=[0, 0], mode="lines", name="Static", line=dict(dash="dot"))
            fig.add_scatter(x=[-xw, 0, l], y=[pr["wing"], pr["zf"], pr["zr"]], mode="lines+markers+text",
                            name="Braking", text=["Wing", "Front axle", "Rear axle"], textposition="top center")
            fig.add_hline(y=-p["wing_lim"], line_color="red", line_dash="dash", annotation_text="Wing limit")
            fig.update_layout(height=320, xaxis_title="Distance from front axle (m, rearward +)",
                              yaxis_title="Ride height change (mm)", margin=dict(t=30, b=10))

            ads = np.linspace(0, 1.5, 61)
            wd = [-pitch_model({**p, "anti_dive": x})["wing"] for x in ads]
            fig2 = go.Figure()
            fig2.add_scatter(x=ads * 100, y=wd, name="Wing drop")
            fig2.add_hline(y=p["wing_lim"], line_color="red", line_dash="dash", annotation_text="Limit")
            fig2.add_vline(x=p["anti_dive"] * 100, line_dash="dash")
            fig2.update_layout(height=320, xaxis_title="Front anti-dive (%)", yaxis_title="Wing drop (mm)",
                               margin=dict(t=30, b=10))
            h1, h2 = st.columns(2)
            h1.plotly_chart(fig, use_container_width=True)
            h2.plotly_chart(fig2, use_container_width=True)

            st.caption("Anti % is the effective value (includes brake bias) and applies to sprung-mass "
                       "load transfer only. Tyres see the full ΔZ. Aero load change not included.")

    # ============================ NONLINEAR TAB ===========================
    with tab_nl:
        mp = mass_props(p)
        Wsf, Wsr = (p["m_fc"] - p["mu_fc"]) * G, (p["m_rc"] - p["mu_rc"]) * G
        st.caption("Corners are named outer / inner for the turn. Travel and heave are wheel travel from "
                   "static ride height (bump +). Roll + = outer side down, pitch + = nose down. "
                   "With flat curves the solver reproduces the linear tabs.")
        t_spr, t_tyr, t_rc, t_pc, t_run = st.tabs(
            ["Springs & bump stops", "Tyres", "Roll centres", "Pitch centre", "Run & results"])

        with t_spr:
            cf, cr = st.columns(2)
            with cf:
                wheel_f = spring_input("f", "Front", DEFAULTS["kw_f"], DEFAULTS["mr_f"], Wsf)
            with cr:
                wheel_r = spring_input("r", "Rear", DEFAULTS["kw_r"], DEFAULTS["mr_r"], Wsr)
            c1, c2 = st.columns(2)
            karb_f = c1.number_input("Front ARB wheel rate in roll (N/mm)", value=round(DEFAULTS["karb_f"] / 1000, 3),
                                     step=0.5, key="nl_karb_f")
            karb_r = c2.number_input("Rear ARB wheel rate in roll (N/mm)", value=round(DEFAULTS["karb_r"] / 1000, 3),
                                     step=0.5, key="nl_karb_r")

        with t_tyr:
            cf, cr = st.columns(2)
            tyres = {}
            for col, key, lab in [(cf, "f", "Front"), (cr, "r", "Rear")]:
                with col:
                    st.markdown(f"### {lab}")
                    d = curve_input(f"ty_{key}", pd.DataFrame({"load_N": [0.0, 500.0, 1000.0, 2000.0, 3000.0],
                                                               "rate_Nmm": [round(DEFAULTS[f"kt_{key}"] / 1000, 3)] * 5}),
                                    "Tyre vertical rate vs load", "Rate held flat outside the table.")
                    tyres[key] = dict(load=d.iloc[:, 0].values, rate=d.iloc[:, 1].values)
                    fig = go.Figure(go.Scatter(x=d.iloc[:, 0], y=d.iloc[:, 1], mode="lines+markers"))
                    fig.update_layout(height=260, xaxis_title="Load (N)", yaxis_title="Rate (N/mm)",
                                      margin=dict(t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        with t_rc:
            cf, cr = st.columns(2)
            rc_help = ("Roll centre height above ground (mm) vs suspension roll angle of that axle "
                       "(deg, outer down +) and axle heave (mean wheel travel, mm, bump +).")
            with cf:
                rc_f = grid_input("rcf", "Front roll centre height (mm)", "Roll angle", DEFAULTS["q_f"] * 1000, rc_help)
            with cr:
                rc_r = grid_input("rcr", "Rear roll centre height (mm)", "Roll angle", DEFAULTS["q_r"] * 1000, rc_help)

        with t_pc:
            pc_help = ("Pitch centre height above ground (mm) vs suspension pitch angle (deg, nose down +) "
                       "and heave at the sprung CG (wheel travel, mm, bump +). "
                       "Effective anti % = pitch centre height / sprung CG height.")
            st.write(f"Sprung CG height hs = {mp['hs']*1000:.1f} mm  →  30 % anti = {0.3*mp['hs']*1000:.1f} mm")
            pc = grid_input("pc", "Pitch centre height (mm)", "Pitch angle", round(0.3 * mp["hs"] * 1000, 2), pc_help)

        with t_run:
            c1, c2, c3, c4 = st.columns(4)
            ay_t = c1.number_input("Target lateral g", value=1.5, step=0.1, key="nl_ay")
            ax_t = c2.number_input("Target braking g (accel -)", value=0.0, step=0.1, key="nl_ax")
            nst = c3.number_input("Steps", min_value=5, max_value=200, value=30, step=5, key="nl_n")
            x_wing = c4.number_input("Front wing ahead of front axle (m)", value=DEFAULTS["x_wing"], step=0.05,
                                     key="nl_xw")
            c1, c2, c3, c4 = st.columns(4)
            FL = c1.number_input("Aero downforce (N, constant)", value=0.0, step=100.0, key="nl_FL")
            bal = c2.number_input("Aero balance front (%)", value=45.0, step=1.0, key="nl_bal") / 100
            so2 = c3.checkbox("2nd-order CG shift", value=False, key="nl_so2",
                              help="Adds the gravity moment from the CG moving sideways/forwards as the body "
                                   "rolls/pitches: ms·g·(hs − RC)·angle.")
            wing_lim = c4.number_input("Max allowed wing drop (mm)", value=DEFAULTS["wing_lim"], step=1.0,
                                       key="nl_wl")
            st.caption("The sweep ramps lateral and braking g together from 0 to the targets "
                       "(set one to 0 for a pure lateral or pure braking sweep).")

            cfg = dict(wheel_f=wheel_f, wheel_r=wheel_r, tyre_f=tyres["f"], tyre_r=tyres["r"],
                       rc_f=rc_f, rc_r=rc_r, pc=pc, karb_f=karb_f, karb_r=karb_r,
                       FL=FL, aero_bal=bal, x_wing=x_wing, second_order=so2)
            try:
                S = nl_build(mp, p, cfg)
                rows, ok = nl_sweep(S, ay_t, ax_t, int(nst))
            except Exception as e:
                st.error(f"Solver error: {e}")
                st.stop()
            res = pd.DataFrame(rows)

            if not ok:
                bad = res.loc[~res["converged"]].iloc[0]
                st.warning(f"Solver did not converge from {bad['comb_g']:.2f} g. Results after that point "
                           "are not valid - check curve ranges.")
            fl = res.loc[res["flags"] != "", ["comb_g", "flags"]]
            if not fl.empty:
                seen = {}
                for _, rw in fl.iterrows():
                    for f in rw["flags"].split(", "):
                        seen.setdefault(f, rw["comb_g"])
                st.warning("  |  ".join(f"{f} from {g:.2f} g" for f, g in seen.items()))

            xopts = {"Lateral g": "ay_g", "Braking g": "ax_g", "Combined g": "comb_g"}
            default_x = "Lateral g" if ay_t != 0 else ("Braking g" if ax_t != 0 else "Combined g")
            xl = st.radio("X axis", list(xopts), index=list(xopts).index(default_x), horizontal=True, key="nl_x")
            xc = xopts[xl]

            last = res.iloc[-1]
            m = st.columns(6)
            m[0].metric("Roll", f"{last['roll']:.3f}°")
            m[1].metric("Pitch", f"{last['pitch']:.3f}°")
            m[2].metric("Front LLTD", "-" if pd.isna(last["lltd"]) else f"{last['lltd']:.1f} %")
            m[3].metric("Heave at CG", f"{last['heave']:.1f} mm")
            m[4].metric("Front wing drop", f"{-last['wing']:.1f} mm",
                        delta=f"{wing_lim + last['wing']:+.1f} mm vs limit", delta_color="normal")
            m[5].metric("Max bump stop force", f"{res[['bump_FO','bump_FI','bump_RO','bump_RI']].values.max():.0f} N")

            g1, g2 = st.columns(2)
            g1.plotly_chart(lines(res, xc, xl, ["roll", "roll_susp_f", "roll_susp_r", "pitch", "pitch_susp"],
                                  ["Body roll", "Susp. roll F", "Susp. roll R", "Body pitch", "Susp. pitch"],
                                  "deg", "Attitude"), use_container_width=True)
            g2.plotly_chart(lines(res, xc, xl, ["rh_f", "rh_r", "wing", "heave"],
                                  ["Front axle", "Rear axle", "Front wing", "Heave at CG"],
                                  "mm (+ up)", "Ride height change", hl=[(-wing_lim, "Wing limit")]),
                            use_container_width=True)
            g1, g2 = st.columns(2)
            g1.plotly_chart(lines(res, xc, xl, ["N_FO", "N_FI", "N_RO", "N_RI"],
                                  ["Front outer", "Front inner", "Rear outer", "Rear inner"],
                                  "N", "Tyre loads"), use_container_width=True)
            g2.plotly_chart(lines(res, xc, xl, ["s_FO", "s_FI", "s_RO", "s_RI"],
                                  ["Front outer", "Front inner", "Rear outer", "Rear inner"],
                                  "mm (bump +)", "Wheel travel"), use_container_width=True)
            g1, g2 = st.columns(2)
            g1.plotly_chart(lines(res, xc, xl, ["q_f", "q_r", "q_p"],
                                  ["Front RC", "Rear RC", "Pitch centre"], "mm", "Roll / pitch centre height"),
                            use_container_width=True)
            g2.plotly_chart(lines(res, xc, xl, ["lltd", "anti"], ["Front LLTD", "Effective anti"],
                                  "%", "LLTD & anti"), use_container_width=True)
            g1, g2 = st.columns(2)
            g1.plotly_chart(lines(res, xc, xl, ["geo_f", "el_f", "uns_f", "geo_r", "el_r", "uns_r"],
                                  ["Geo F", "Elastic F", "Unsprung F", "Geo R", "Elastic R", "Unsprung R"],
                                  "N (per wheel)", "Lateral load transfer split"), use_container_width=True)
            g2.plotly_chart(lines(res, xc, xl, ["k_FO", "k_FI", "k_RO", "k_RI"],
                                  ["Front outer", "Front inner", "Rear outer", "Rear inner"],
                                  "N/mm", "Wheel rate at operating point"), use_container_width=True)
            if wheel_f["mode"] == "spring" or wheel_r["mode"] == "spring":
                hl = []
                if wheel_f["mode"] == "spring":
                    hl.append((wheel_f["gap"], "Front gap"))
                if wheel_r["mode"] == "spring":
                    hl.append((wheel_r["gap"], "Rear gap"))
                g1, g2 = st.columns(2)
                g1.plotly_chart(lines(res, xc, xl, ["x_FO", "x_FI", "x_RO", "x_RI"],
                                      ["Front outer", "Front inner", "Rear outer", "Rear inner"],
                                      "mm (comp +)", "Spring / damper travel", hl=hl), use_container_width=True)
                g2.plotly_chart(lines(res, xc, xl, ["bump_FO", "bump_FI", "bump_RO", "bump_RI"],
                                      ["Front outer", "Front inner", "Rear outer", "Rear inner"],
                                      "N", "Bump stop force (at damper)"), use_container_width=True)

            with st.expander("Results table"):
                st.dataframe(res.style.format(precision=3), use_container_width=True)
            st.download_button("Download results CSV", res.to_csv(index=False), "nonlinear_sweep.csv",
                               key="dl_results")
            st.caption("Not included: jacking from unequal left/right tyre lateral forces, lateral RC "
                       "migration, compliance steer/camber, aero map vs ride height. Results after "
                       "'wheel lift' are not valid (tyre is not allowed to leave the ground).")
