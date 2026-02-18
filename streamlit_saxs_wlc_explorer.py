import io
import importlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gamma, laplace, linregress, lognorm, norm, triang, uniform

HAS_SASMODELS = importlib.util.find_spec("sasmodels") is not None
if HAS_SASMODELS:
    from sasmodels.core import load_model
    from sasmodels.data import empty_data1D
    from sasmodels.direct_model import DirectModel


@dataclass
class DistributionResult:
    values: np.ndarray
    weights: np.ndarray
    pdf: np.ndarray


def benoit_doty_rg(length: float, kuhn_length: float) -> float:
    """Benoit-Doty relation for ideal worm-like chain using contour length L and Kuhn length b."""
    L = max(length, 1e-12)
    b = max(kuhn_length, 1e-12)
    x = L / b
    rg2 = (L * b / 6.0) - (b**2 / 4.0) + (b**3 / (4.0 * L)) - (b**4 / (8.0 * L**2)) * (1.0 - np.exp(-2.0 * x))
    return float(np.sqrt(max(rg2, 0.0)))


def _safe_normalize(pdf: np.ndarray) -> np.ndarray:
    s = np.trapz(pdf) if pdf.ndim == 1 else np.sum(pdf)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(pdf) / len(pdf)
    return pdf / np.sum(pdf)


def make_distribution(mean: float, pd_index: float, dist_name: str, n_bins: int) -> DistributionResult:
    mean = max(float(mean), 1e-9)
    pd_index = max(float(pd_index), 0.0)
    n_bins = max(int(n_bins), 1)

    if pd_index == 0 or n_bins == 1:
        vals = np.array([mean])
        return DistributionResult(vals, np.array([1.0]), np.array([1.0]))

    sigma = max(pd_index * mean, 1e-12)

    if dist_name == "Gaussian":
        x = np.linspace(mean - 4 * sigma, mean + 4 * sigma, n_bins)
        x = x[x > 1e-12]
        if x.size == 0:
            x = np.array([mean])
        pdf = norm.pdf(x, loc=mean, scale=sigma)
    elif dist_name == "Lognormal":
        cv = max(pd_index, 1e-12)
        sigma_ln2 = np.log1p(cv**2)
        sigma_ln = np.sqrt(sigma_ln2)
        mu_ln = np.log(mean) - 0.5 * sigma_ln2
        dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))
        q_low, q_high = dist.ppf([0.001, 0.999])
        x = np.linspace(max(q_low, 1e-12), max(q_high, 2e-12), n_bins)
        pdf = dist.pdf(x)
    elif dist_name == "Schulz":
        cv = max(pd_index, 1e-12)
        k = 1.0 / (cv**2)
        theta = mean / k
        dist = gamma(a=k, scale=theta)
        q_low, q_high = dist.ppf([0.001, 0.999])
        x = np.linspace(max(q_low, 1e-12), max(q_high, 2e-12), n_bins)
        pdf = dist.pdf(x)
    elif dist_name == "Triangular":
        half_width = np.sqrt(6.0) * sigma
        left, right = max(mean - half_width, 1e-12), mean + half_width
        x = np.linspace(left, right, n_bins)
        c = np.clip((mean - left) / max(right - left, 1e-12), 0.0, 1.0)
        pdf = triang.pdf(x, c=c, loc=left, scale=right - left)
    elif dist_name == "Uniform":
        half_width = np.sqrt(3.0) * sigma
        left, right = max(mean - half_width, 1e-12), mean + half_width
        x = np.linspace(left, right, n_bins)
        pdf = uniform.pdf(x, loc=left, scale=max(right - left, 1e-12))
    elif dist_name == "Boltzmann":
        bscale = max((pd_index * mean) / np.sqrt(2.0), 1e-12)
        x = np.linspace(max(mean - 8 * bscale, 1e-12), mean + 8 * bscale, n_bins)
        pdf = laplace.pdf(x, loc=mean, scale=bscale)
    else:
        x = np.array([mean])
        pdf = np.array([1.0])

    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    w = _safe_normalize(pdf)
    return DistributionResult(values=x, weights=w, pdf=pdf)


@st.cache_resource(show_spinner=False)
def get_flexible_cylinder_calculator():
    if not HAS_SASMODELS:
        return None
    model = load_model("flexible_cylinder")
    data = empty_data1D(np.logspace(-4, 0, 16))
    return DirectModel(data, model)


def compute_intensity(
    q: np.ndarray,
    base_params: Dict[str, float],
    contour_dist: DistributionResult,
    kuhn_dist: DistributionResult,
) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_SASMODELS:
        return np.full_like(q, np.nan), np.full_like(q, np.nan)

    calc = get_flexible_cylinder_calculator()

    mean_i = calc(q=q, **base_params)

    weights_2d = np.outer(contour_dist.weights, kuhn_dist.weights)
    weights_2d = weights_2d / np.sum(weights_2d)

    smeared = np.zeros_like(q, dtype=float)
    for i, L in enumerate(contour_dist.values):
        for j, B in enumerate(kuhn_dist.values):
            wij = weights_2d[i, j]
            if wij <= 0:
                continue
            p = dict(base_params)
            p["length"] = float(L)
            p["kuhn_length"] = float(B)
            Iij = calc(q=q, **p)
            smeared += wij * Iij

    return mean_i, smeared


def build_csv_bytes(df: pd.DataFrame, metadata: Dict[str, float]) -> bytes:
    sio = io.StringIO()
    sio.write("# SAXS WLC simulation metadata\n")
    for k, v in metadata.items():
        sio.write(f"# {k},{v}\n")
    sio.write("\n")
    df.to_csv(sio, index=False)
    return sio.getvalue().encode("utf-8")


def transform_for_mode(mode: str, q: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, str]:
    safe_i = np.clip(intensity, 1e-300, None)
    if mode == "Log-Log":
        return q, safe_i, "q", "I(q)"
    if mode == "Lin-Lin":
        return q, intensity, "q", "I(q)"
    if mode == "Guinier":
        return q**2, np.log(safe_i), "q²", "ln I(q)"
    if mode == "Kratky":
        return q, (q**2) * intensity, "q", "q² I(q)"
    if mode == "Porod":
        return q, (q**4) * intensity, "q", "q⁴ I(q)"
    return q, intensity, "q", "I(q)"


def main():
    st.set_page_config(page_title="SAXS WLC Explorer", layout="wide")
    st.title("Streamlit SAXS Worm-like Chain Explorer")

    if not HAS_SASMODELS:
        st.error("sasmodels is not available. Install `sasmodels` to compute scattering curves.")

    with st.sidebar:
        st.header("Q-Range")
        q_min = st.number_input("q min", min_value=1e-6, value=1e-3, format="%.6f")
        q_max = st.number_input("q max", min_value=1e-6, value=0.5, format="%.6f")
        n_q = st.number_input("Number of q points", min_value=10, max_value=20000, value=500, step=10)
        log_q = st.toggle("Log spacing", value=True)

        st.header("Physical Properties")
        radius = st.number_input("Radius", min_value=1e-5, value=20.0)
        sld = st.number_input("SLD (polymer)", value=1.0)
        sld_solvent = st.number_input("SLD (solvent)", value=6.3)
        scale = st.number_input("Scale", min_value=0.0, value=1.0)
        background = st.number_input("Background", min_value=0.0, value=0.0, format="%.8f")

        dist_names = ["Gaussian", "Lognormal", "Schulz", "Triangular", "Uniform", "Boltzmann"]

        st.header("Contour Length L")
        length_mean = st.number_input("Mean L", min_value=1e-4, value=500.0)
        length_dist = st.selectbox("L distribution", dist_names)
        length_pd = st.slider("L polydispersity index (σ/μ)", min_value=0.0, max_value=20.0, value=0.0, step=0.05)
        length_bins = st.number_input("L integration bins", min_value=1, max_value=300, value=25)

        st.header("Kuhn Length b")
        kuhn_mean = st.number_input("Mean b", min_value=1e-4, value=100.0)
        kuhn_dist_name = st.selectbox("b distribution", dist_names, index=0)
        kuhn_pd = st.slider("b polydispersity index (σ/μ)", min_value=0.0, max_value=20.0, value=0.0, step=0.05)
        kuhn_bins = st.number_input("b integration bins", min_value=1, max_value=300, value=25)

    q = np.logspace(np.log10(q_min), np.log10(q_max), int(n_q)) if log_q else np.linspace(q_min, q_max, int(n_q))
    q = np.clip(q, 1e-12, None)

    contour = make_distribution(length_mean, length_pd, length_dist, int(length_bins))
    kuhn = make_distribution(kuhn_mean, kuhn_pd, kuhn_dist_name, int(kuhn_bins))

    base_params = {
        "radius": float(radius),
        "sld": float(sld),
        "sld_solvent": float(sld_solvent),
        "scale": float(scale),
        "background": float(background),
        "length": float(length_mean),
        "kuhn_length": float(kuhn_mean),
    }

    if HAS_SASMODELS:
        with st.spinner("Computing scattering..."):
            i_mean, i_smeared = compute_intensity(q, base_params, contour, kuhn)
    else:
        i_mean = np.full_like(q, np.nan)
        i_smeared = np.full_like(q, np.nan)

    plot_mode = st.radio("Plot mode", ["Log-Log", "Lin-Lin", "Guinier", "Kratky", "Porod"], horizontal=True)

    fig = go.Figure()
    x_mean, y_mean, xl, yl = transform_for_mode(plot_mode, q, i_mean)
    x_sm, y_sm, _, _ = transform_for_mode(plot_mode, q, i_smeared)

    pd_active = (length_pd > 0 and len(contour.values) > 1) or (kuhn_pd > 0 and len(kuhn.values) > 1)
    if pd_active:
        fig.add_trace(go.Scatter(x=x_mean, y=y_mean, mode="lines", name="Mean", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=x_sm, y=y_sm, mode="lines", name="Smeared"))
    else:
        fig.add_trace(go.Scatter(x=x_mean, y=y_mean, mode="lines", name="I(q)"))

    if plot_mode == "Log-Log":
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    analysis_text = ""
    if plot_mode == "Guinier" and HAS_SASMODELS:
        rg_theory = benoit_doty_rg(length_mean, kuhn_mean)
        q_limit = 1.3 / max(rg_theory, 1e-12)
        mask = (q <= q_limit) & np.isfinite(i_smeared) & (i_smeared > 0)

        if np.count_nonzero(mask) >= 3:
            res = linregress(q[mask] ** 2, np.log(i_smeared[mask]))
            rg_app = np.sqrt(max(-3.0 * res.slope, 0.0))
            i0_app = float(np.exp(res.intercept))
            analysis_text = (
                f"Theoretical Rg (Benoit-Doty): {rg_theory:.4g} | "
                f"Validity limit q·Rg ≤ 1.3 → q ≤ {q_limit:.4g} | "
                f"Apparent Rg: {rg_app:.4g} | Apparent I0: {i0_app:.4g}"
            )

            if st.button("Zoom to Fit Region"):
                y_zoom = np.log(np.clip(i_smeared[mask], 1e-300, None))
                fig.update_xaxes(range=[0, float((q_limit ** 2) * 1.05)])
                fig.update_yaxes(range=[float(np.min(y_zoom) * 0.98), float(np.max(y_zoom) * 1.02)])
        else:
            analysis_text = "Not enough points in Guinier validity region for regression."

    fig.update_layout(xaxis_title=xl, yaxis_title=yl, template="plotly_white", height=520)
    st.plotly_chart(fig, use_container_width=True)
    if analysis_text:
        st.info(analysis_text)

    c1, c2 = st.columns(2)
    with c1:
        fig_l = go.Figure(go.Scatter(x=contour.values, y=contour.pdf, mode="lines", name="L PDF"))
        fig_l.update_layout(template="plotly_white", title="Contour Length Distribution", xaxis_title="L", yaxis_title="PDF")
        st.plotly_chart(fig_l, use_container_width=True)

    with c2:
        fig_b = go.Figure(go.Scatter(x=kuhn.values, y=kuhn.pdf, mode="lines", name="b PDF"))
        fig_b.update_layout(template="plotly_white", title="Kuhn Length Distribution", xaxis_title="b", yaxis_title="PDF")
        st.plotly_chart(fig_b, use_container_width=True)

    out_df = pd.DataFrame(
        {
            "q": q,
            "I_mean": i_mean,
            "I_smeared": i_smeared,
        }
    )

    metadata = {
        "q_min": q_min,
        "q_max": q_max,
        "n_q": n_q,
        "log_q": log_q,
        "radius": radius,
        "sld": sld,
        "sld_solvent": sld_solvent,
        "scale": scale,
        "background": background,
        "length_mean": length_mean,
        "length_distribution": length_dist,
        "length_pd": length_pd,
        "length_bins": length_bins,
        "kuhn_mean": kuhn_mean,
        "kuhn_distribution": kuhn_dist_name,
        "kuhn_pd": kuhn_pd,
        "kuhn_bins": kuhn_bins,
    }
    st.download_button(
        "Download CSV",
        data=build_csv_bytes(out_df, metadata),
        file_name="saxs_wlc_simulation.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
