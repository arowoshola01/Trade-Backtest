"""
hmm_model.py
Ports of:
  - MVT HMM Regime Classifier: 5-state, full 4x4 covariance Gaussian HMM
    (Bull Extreme, Bull Strong, Neutral/Chop, Bear Strong, Bear Extreme)
  - MoVol HMM Multi-S: generic bivariate-Gaussian HMM, S=4/5/6 state models,
    trained separately for M5 and M15.

Both are Bayesian forward filters (predict -> update) run bar-by-bar,
exactly mirroring the recursive `var float prob_x` pattern in Pine.
"""

import numpy as np

STATES_5 = ["be", "bs", "nc", "brs", "bre"]  # Bull Extreme..Bear Extreme


# ═══════════════════════════════════════════════════════════════════════════
# MVT HMM — 5-state, full covariance
# ═══════════════════════════════════════════════════════════════════════════

# Transition matrix and emission parameters transplanted verbatim from Pine.
# Structure: MVT_PARAMS[tf_cal][state] = {...}
MVT_PARAMS = {
    "M5 Trained": {
        "trans": {  # from -> {to: p}
            "be":  {"be": 0.49909676, "bs": 0.00150444, "nc": 0.46244246, "brs": 0.03695635, "bre": 0.0},
            "bs":  {"be": 0.25357309, "bs": 0.15451822, "nc": 0.55752327, "brs": 0.03386534, "bre": 0.00052008},
            "nc":  {"be": 0.04225314, "bs": 0.15239004, "nc": 0.61616399, "brs": 0.14776250, "bre": 0.04143034},
            "brs": {"be": 0.00057480, "bs": 0.03319735, "nc": 0.55368699, "brs": 0.15531996, "bre": 0.25722090},
            "bre": {"be": 0.0,        "bs": 0.03668776, "nc": 0.46746740, "brs": 0.00171012, "bre": 0.49413472},
        },
        "emit": {
            "be":  dict(mu=(99.99750, 0.86470, 0.37941, 1.28467),
                        sg=(0.00316, 0.15484, 0.45590, 0.85579),
                        rho12=0.10091, rho13=0.07159, rho14=0.23829, rho23=0.21741, rho24=0.50314, rho34=0.39254),
            "bs":  dict(mu=(54.93915, 0.65412, 0.44982, 1.00053),
                        sg=(34.23373, 0.22891, 0.48535, 0.85770),
                        rho12=0.76718, rho13=0.40644, rho14=0.31328, rho23=0.47317, rho24=0.48536, rho34=0.45256),
            "nc":  dict(mu=(-1.44603, -0.00921, 0.00000, 0.00187),
                        sg=(57.29204, 0.46887, 0.00100, 0.78929),
                        rho12=0.78173, rho13=-0.00007, rho14=0.51603, rho23=-0.00009, rho24=0.65609, rho34=0.00029),
            "brs": dict(mu=(-59.28566, -0.68674, -0.48825, -1.03579),
                        sg=(32.08787, 0.20866, 0.50572, 0.87583),
                        rho12=0.74905, rho13=0.40948, rho14=0.30187, rho23=0.48184, rho24=0.49288, rho34=0.44446),
            "bre": dict(mu=(-99.99777, -0.88321, -0.41339, -1.32850),
                        sg=(0.00289, 0.13569, 0.49981, 0.86139),
                        rho12=0.14167, rho13=0.07946, rho14=0.23165, rho23=0.21593, rho24=0.49939, rho34=0.39065),
        },
    },
    "M15 Trained": {
        "trans": {
            "be":  {"be": 0.48345462, "bs": 0.00042008, "nc": 0.47728001, "brs": 0.03884530, "bre": 0.0},
            "bs":  {"be": 0.24886932, "bs": 0.14816132, "nc": 0.56861973, "brs": 0.03342132, "bre": 0.00092832},
            "nc":  {"be": 0.04563206, "bs": 0.14188897, "nc": 0.62586145, "brs": 0.14239248, "bre": 0.04422504},
            "brs": {"be": 0.00108132, "bs": 0.03482600, "nc": 0.56696791, "brs": 0.14579555, "bre": 0.25132921},
            "bre": {"be": 0.0,        "bs": 0.03857977, "nc": 0.48244987, "brs": 0.00099963, "bre": 0.47797073},
        },
        "emit": {
            "be":  dict(mu=(99.99944, 0.85555, 0.35567, 1.26601),
                        sg=(0.00135, 0.17044, 0.46474, 0.93350),
                        rho12=0.09506, rho13=0.07174, rho14=0.20711, rho23=0.22439, rho24=0.48781, rho34=0.39154),
            "bs":  dict(mu=(56.10524, 0.66210, 0.43651, 1.05437),
                        sg=(31.85612, 0.21938, 0.46170, 0.90979),
                        rho12=0.75990, rho13=0.41665, rho14=0.29807, rho23=0.50113, rho24=0.48424, rho34=0.44094),
            "nc":  dict(mu=(-0.98727, -0.01039, 0.00000, 0.00132),
                        sg=(55.96393, 0.45749, 0.00100, 0.81231),
                        rho12=0.78919, rho13=0.00020, rho14=0.51371, rho23=0.00009, rho24=0.65275, rho34=0.00019),
            "brs": dict(mu=(-59.43409, -0.68765, -0.46156, -1.07165),
                        sg=(29.99878, 0.20148, 0.48705, 0.94172),
                        rho12=0.73966, rho13=0.40356, rho14=0.29134, rho23=0.49532, rho24=0.48090, rho34=0.43684),
            "bre": dict(mu=(-99.99949, -0.86692, -0.36661, -1.28979),
                        sg=(0.00131, 0.15550, 0.47258, 0.94905),
                        rho12=0.10404, rho13=0.05946, rho14=0.20166, rho23=0.23162, rho24=0.49568, rho34=0.38675),
        },
    },
}


def _build_cov(sg, rho12, rho13, rho14, rho23, rho24, rho34):
    s1, s2, s3, s4 = sg
    cov = np.zeros((4, 4))
    cov[0, 0], cov[1, 1], cov[2, 2], cov[3, 3] = s1*s1, s2*s2, s3*s3, s4*s4
    cov[0, 1] = cov[1, 0] = rho12 * s1 * s2
    cov[0, 2] = cov[2, 0] = rho13 * s1 * s3
    cov[0, 3] = cov[3, 0] = rho14 * s1 * s4
    cov[1, 2] = cov[2, 1] = rho23 * s2 * s3
    cov[1, 3] = cov[3, 1] = rho24 * s2 * s4
    cov[2, 3] = cov[3, 2] = rho34 * s3 * s4
    return cov


def _mvgauss_loglike_batch(x, mu, inv_cov, det):
    """x: (N,4) array. Returns (N,) likelihood (not log) for numerical parity with Pine."""
    d = x - np.array(mu)
    quad = np.einsum("ij,jk,ik->i", d, inv_cov, d)
    norm = 1.0 / np.sqrt((2.0 * np.pi) ** 4 * abs(det))
    return norm * np.exp(np.clip(-0.5 * quad, -700.0, None))


class MVTHmm:
    def __init__(self, tf_cal="M5 Trained"):
        assert tf_cal in MVT_PARAMS
        self.tf_cal = tf_cal
        params = MVT_PARAMS[tf_cal]
        self.trans = np.array([[params["trans"][a][b] for b in STATES_5] for a in STATES_5])
        self.mu, self.inv_cov, self.det = {}, {}, {}
        for s in STATES_5:
            e = params["emit"][s]
            cov = _build_cov(e["sg"], e["rho12"], e["rho13"], e["rho14"], e["rho23"], e["rho24"], e["rho34"])
            self.mu[s] = e["mu"]
            self.inv_cov[s] = np.linalg.inv(cov)
            self.det[s] = np.linalg.det(cov)
        self.prob = np.full(5, 0.2)  # var float prob_x = 0.20 each

    def reset(self):
        self.prob = np.full(5, 0.2)

    def step(self, zio, vpmo, prt, vel):
        """Process one bar's observables. Returns dict with probs + dominant state."""
        prior = self.prob @ self.trans  # prior_x = sum_i prob_i * p_i_x
        x = np.array([zio, vpmo, prt, vel])
        like = np.array([
            _mvgauss_loglike_batch(x[None, :], self.mu[s], self.inv_cov[s], self.det[s])[0]
            for s in STATES_5
        ])
        post = prior * like
        total = post.sum()
        if total > 0:
            self.prob = post / total
        pct = dict(zip(STATES_5, self.prob * 100))
        dom = max(pct, key=pct.get)
        dom_code = {"be": 2, "bs": 1, "nc": 0, "brs": -1, "bre": -2}[dom]
        return {"pct": pct, "dominant": dom, "dominant_code": dom_code, "confidence": pct[dom]}

    def run(self, obs_df):
        """obs_df: DataFrame with columns zio_hmm, vpmo_hmm, prt_hmm, vel_hmm. Returns list of step results."""
        self.reset()
        out = []
        for _, row in obs_df.iterrows():
            if row.isna().any():
                out.append(None)
                continue
            out.append(self.step(row["zio_hmm"], row["vpmo_hmm"], row["prt_hmm"], row["vel_hmm"]))
        return out


# ═══════════════════════════════════════════════════════════════════════════
# MOVOL HMM — generic S=4/5/6 bivariate Gaussian, M5/M15
# ═══════════════════════════════════════════════════════════════════════════

def _S4_M5():
    mu_mom = [1.094365, 0.966291, -1.038360, -0.960873]
    mu_vol = [1.339331, -1.562983, 1.315524, -1.524380]
    sg_mom = [0.841906, 0.694730, 0.857522, 0.727373]
    sg_vol = [0.953884, 0.466267, 0.929667, 0.510236]
    rho = [0.217890, 0.207057, -0.284901, -0.210614]
    trans = [
        [0.945301, 0.012294, 0.042277, 0.000128],
        [0.010411, 0.953357, 0.002420, 0.033811],
        [0.040031, 0.000232, 0.945898, 0.013840],
        [0.003873, 0.032021, 0.011767, 0.952340],
    ]
    labels = ["Bull Breakout", "Bull Trend", "Bear Breakdown", "Bear Drift"]
    dir_sign = [1, 1, -1, -1]
    return mu_mom, mu_vol, sg_mom, sg_vol, rho, trans, labels, dir_sign


def _S4_M15():
    mu_mom = [0.979561, 0.909258, -0.983226, -0.877724]
    mu_vol = [1.192189, -1.421229, 1.220485, -1.387871]
    sg_mom = [0.789413, 0.660899, 0.797629, 0.677851]
    sg_vol = [0.835981, 0.434235, 0.812388, 0.458959]
    rho = [0.319213, 0.247365, -0.313929, -0.210382]
    trans = [
        [0.855165, 0.038106, 0.103303, 0.003427],
        [0.027838, 0.882249, 0.010772, 0.079141],
        [0.103825, 0.000343, 0.857212, 0.038620],
        [0.014120, 0.076738, 0.030597, 0.878546],
    ]
    labels = ["Bull Breakout", "Bull Trend", "Bear Breakdown", "Bear Drift"]
    dir_sign = [1, 1, -1, -1]
    return mu_mom, mu_vol, sg_mom, sg_vol, rho, trans, labels, dir_sign


def _S5_M5():
    mu_mom = [1.343704, 0.978187, -0.427193, -0.949466, -1.077247]
    mu_vol = [1.581866, -1.543174, 0.335369, -1.598776, 1.903255]
    sg_mom = [0.808725, 0.688423, 0.894096, 0.726246, 0.953053]
    sg_vol = [0.875777, 0.469338, 0.548364, 0.460181, 0.644431]
    rho = [0.090546, 0.174192, -0.001272, -0.272745, -0.222851]
    trans = [
        [0.940252, 0.004519, 0.022912, 0.000000, 0.032316],
        [0.007019, 0.952904, 0.008609, 0.031362, 0.000106],
        [0.020257, 0.016076, 0.921297, 0.020721, 0.021648],
        [0.000626, 0.031963, 0.013615, 0.953141, 0.000655],
        [0.033856, 0.000000, 0.023691, 0.000000, 0.942453],
    ]
    labels = ["Bull Breakout", "Bull Trend", "Neutral/Chop", "Bear Drift", "Bear Breakdown"]
    dir_sign = [1, 1, 0, -1, -1]
    return mu_mom, mu_vol, sg_mom, sg_vol, rho, trans, labels, dir_sign


def _S5_M15():
    mu_mom = [1.019282, 0.894619, 0.002950, -0.857207, -1.049920]
    mu_vol = [1.532828, -1.511094, -0.117300, -1.507009, 1.534847]
    sg_mom = [0.801843, 0.636356, 1.168433, 0.655170, 0.801432]
    sg_vol = [0.613928, 0.346463, 0.510735, 0.358194, 0.612645]
    rho = [0.368299, 0.217981, 0.005588, -0.225898, -0.358155]
    trans = [
        [0.849220, 0.000000, 0.056934, 0.000000, 0.093846],
        [0.001725, 0.881064, 0.047265, 0.068862, 0.001084],
        [0.055388, 0.061637, 0.757581, 0.065428, 0.059966],
        [0.001338, 0.068516, 0.049634, 0.877911, 0.002602],
        [0.097424, 0.000000, 0.055073, 0.000000, 0.847502],
    ]
    labels = ["Bull Breakout", "Bull Trend", "Neutral/Chop", "Bear Drift", "Bear Breakdown"]
    dir_sign = [1, 1, 0, -1, -1]
    return mu_mom, mu_vol, sg_mom, sg_vol, rho, trans, labels, dir_sign


def _S6_M5():
    mu_mom = [1.387816, 0.943291, 0.008411, -0.358556, -0.913564, -1.141505]
    mu_vol = [1.876957, -1.659369, -0.345071, 0.937301, -1.696093, 2.141487]
    sg_mom = [0.812949, 0.647842, 1.383595, 0.912133, 0.682342, 0.991435]
    sg_vol = [0.712654, 0.363497, 0.448248, 0.320785, 0.395249, 0.598265]
    rho = [0.198274, 0.107729, 0.045903, -0.101220, -0.243634, -0.228818]
    trans = [
        [0.934694, 0.000000, 0.002299, 0.031507, 0.000000, 0.031500],
        [0.000040, 0.950913, 0.018944, 0.000000, 0.029969, 0.000134],
        [0.014971, 0.024570, 0.905226, 0.025412, 0.024654, 0.005167],
        [0.015054, 0.000000, 0.043851, 0.920014, 0.000000, 0.021082],
        [0.000206, 0.029641, 0.018145, 0.000000, 0.951829, 0.000178],
        [0.035229, 0.000000, 0.000000, 0.032948, 0.000000, 0.931822],
    ]
    labels = ["Bull Breakout", "Bull Trend", "Quiet Chop", "Volatile Chop", "Bear Drift", "Bear Breakdown"]
    dir_sign = [1, 1, 0, 0, -1, -1]
    return mu_mom, mu_vol, sg_mom, sg_vol, rho, trans, labels, dir_sign


def _S6_M15():
    mu_mom = [1.380570, 0.892096, -0.003407, -0.028522, -0.847390, -1.431876]
    mu_vol = [1.478825, -1.537799, -0.327468, 1.359798, -1.541167, 1.551827]
    sg_mom = [0.653796, 0.625575, 1.203971, 0.560305, 0.645878, 0.652033]
    sg_vol = [0.679200, 0.324565, 0.480438, 0.592614, 0.334730, 0.669196]
    rho = [0.507800, 0.208405, 0.007754, 0.138397, -0.227296, -0.411765]
    trans = [
        [0.777017, 0.000000, 0.045434, 0.165039, 0.000000, 0.012510],
        [0.001378, 0.879586, 0.052394, 0.000173, 0.065637, 0.000832],
        [0.050605, 0.067078, 0.743090, 0.013558, 0.071237, 0.054432],
        [0.139261, 0.000000, 0.073116, 0.660755, 0.000001, 0.126869],
        [0.001392, 0.065715, 0.053685, 0.000001, 0.876802, 0.002405],
        [0.009018, 0.000000, 0.030119, 0.186409, 0.000000, 0.774454],
    ]
    labels = ["Bull Breakout", "Bull Trend", "Quiet Chop", "Volatile Chop", "Bear Drift", "Bear Breakdown"]
    dir_sign = [1, 1, 0, 0, -1, -1]
    return mu_mom, mu_vol, sg_mom, sg_vol, rho, trans, labels, dir_sign


_MOVOL_SELECTOR = {
    ("S=4 (Dir. x Vol)", "M5 Trained"): _S4_M5,
    ("S=4 (Dir. x Vol)", "M15 Trained"): _S4_M15,
    ("S=5 (Neutral/Chop)", "M5 Trained"): _S5_M5,
    ("S=5 (Neutral/Chop)", "M15 Trained"): _S5_M15,
    ("S=6 (Quiet/Volatile Chop)", "M5 Trained"): _S6_M5,
    ("S=6 (Quiet/Volatile Chop)", "M15 Trained"): _S6_M15,
}


def _pdf2d(x1, x2, mu1, mu2, s1, s2, rho):
    z1 = (x1 - mu1) / s1
    z2 = (x2 - mu2) / s2
    omr2 = 1.0 - rho * rho
    expo = -(z1*z1 - 2.0*rho*z1*z2 + z2*z2) / (2.0 * omr2)
    den = 2.0 * np.pi * s1 * s2 * np.sqrt(omr2)
    return np.exp(expo) / den


class MoVolHmm:
    def __init__(self, model_sel="S=4 (Dir. x Vol)", tf_cal="M5 Trained"):
        key = (model_sel, tf_cal)
        assert key in _MOVOL_SELECTOR, f"Unknown MoVol model/tf combo: {key}"
        (self.mu_mom, self.mu_vol, self.sg_mom, self.sg_vol,
         self.rho, self.trans, self.labels, self.dir_sign) = _MOVOL_SELECTOR[key]()
        self.n = len(self.mu_mom)
        self.trans = np.array(self.trans)
        self.prob = np.full(self.n, 1.0 / self.n)

    def reset(self):
        self.prob = np.full(self.n, 1.0 / self.n)

    def step(self, obs_mom, obs_vol):
        prior = self.prob @ self.trans
        like = np.array([
            _pdf2d(obs_mom, obs_vol, self.mu_mom[s], self.mu_vol[s], self.sg_mom[s], self.sg_vol[s], self.rho[s])
            for s in range(self.n)
        ])
        post = prior * like
        total = post.sum()
        if total > 0:
            self.prob = post / total
        pct = self.prob * 100
        dom_idx = int(np.argmax(pct))
        return {
            "pct": pct, "dominant_idx": dom_idx,
            "dominant_label": self.labels[dom_idx],
            "dominant_dir": self.dir_sign[dom_idx],
            "confidence": pct[dom_idx],
        }

    def run(self, obs_mom_series, obs_vol_series):
        self.reset()
        out = []
        for m, v in zip(obs_mom_series, obs_vol_series):
            if np.isnan(m) or np.isnan(v):
                out.append(None)
                continue
            out.append(self.step(m, v))
        return out


def compute_movol_observables(df, length):
    """obs_mom / obs_vol — z-scored ROC and z-scored ATR, matching Pine exactly."""
    import pandas as pd
    from indicators import ema, sma, stdev, atr

    mom_raw = df["close"].pct_change() * 100.0  # ta.roc(close,1) == % change
    mom_smooth = ema(mom_raw, length)
    mom_std = stdev(mom_smooth, length)
    msl = sma(mom_smooth, length)
    obs_mom = ((mom_smooth - msl) / mom_std.replace(0, np.nan)).fillna(0.0)

    vol_raw = atr(df, length)
    vol_std = stdev(vol_raw, length)
    vrl = sma(vol_raw, length)
    obs_vol = ((vol_raw - vrl) / vol_std.replace(0, np.nan)).fillna(0.0)

    return pd.DataFrame({"obs_mom": obs_mom, "obs_vol": obs_vol})
