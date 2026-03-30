"""
Strength prediction module for screw-controlled yielding / CRSS.

This module provides:
- model parameters
- core strength equations
- temperature grid generation
- prediction + CSV/PNG export utilities

Output mode:
- CRSS   -> computes total_y (MPa)
- SIGMAY -> computes sigma_y (MPa)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ModelParameters:
    a0: float = 3.3331                # lattice constant
    Ek: float = 0.58425              # kink formation energy
    DEp: float = 0.100989            # solute/screw interaction energy parameter
    Ev: float = 2.4193               # vacancy formation energy
    Esi: float = 4.75695             # interstitial formation energy
    epsilon_0_dot: float = 10000     # reference strain rate
    epsilon_dot: float = 0.001       # experimental strain rate
    wk: float = 10                   # kink width (in units of b)
    stress_convert_factor: float = 160000
    sigma_factor: float = 2.74       # Taylor/orientation conversion to sigma_y


def validate_mode(CRSS_SIGMAY: str) -> str:
    """
    Validate and normalize output mode.

    Allowed values:
    - 'CRSS'
    - 'SIGMAY'
    """
    mode = str(CRSS_SIGMAY).strip().upper()
    if mode not in {"CRSS", "SIGMAY"}:
        raise ValueError("CRSS_SIGMAY must be either 'CRSS' or 'SIGMAY'.")
    return mode


def generate_temperature_range(max_T: float = 1000, T_interval: float = 50) -> np.ndarray:
    """
    Generate temperatures from 0 to max_T inclusive when possible.

    Parameters
    ----------
    max_T : float
        Maximum temperature in K.
    T_interval : float
        Temperature interval in K.

    Returns
    -------
    np.ndarray
        1D array of temperatures in K.
    """
    if max_T < 0:
        raise ValueError("max_T must be non-negative.")
    if T_interval <= 0:
        raise ValueError("T_interval must be positive.")

    n_steps = int(np.floor(max_T / T_interval))
    T_values = np.arange(n_steps + 1, dtype=float) * T_interval

    if len(T_values) == 0 or T_values[-1] < max_T:
        T_values = np.append(T_values, float(max_T))

    return T_values


def compute_derived_quantities(params: ModelParameters) -> dict:
    """
    Compute derived quantities used across the model.
    """
    b = np.sqrt(3) / 2 * params.a0
    zeta_c = (1.083 * params.Ek / params.DEp) ** 2
    tau_c = params.DEp / (0.943 * b ** 3)
    tau_b = 1.08 * params.Ek / (0.943 * b ** 3 * zeta_c)
    tau_kov = np.pi / 2 * params.Ev / (0.943 * b ** 3 * 7.5 * zeta_c)
    tau_kosi = np.pi / 2 * params.Esi / (0.943 * b ** 3 * 15 * zeta_c)
    tau_kov_ath = params.Ev / (0.943 * b ** 3 * 7.5 * zeta_c) * params.stress_convert_factor
    tau_kosi_ath = params.Esi / (0.943 * b ** 3 * 15 * zeta_c) * params.stress_convert_factor

    return {
        "b": b,
        "zeta_c": zeta_c,
        "tau_c": tau_c,
        "tau_b": tau_b,
        "tau_kov": tau_kov,
        "tau_kosi": tau_kosi,
        "tau_kov_ath": tau_kov_ath,
        "tau_kosi_ath": tau_kosi_ath,
    }


def dh(T: float, params: ModelParameters) -> float:
    K = 8.617e-5
    return K * T * np.log(params.epsilon_0_dot / params.epsilon_dot)


def tau_kh(T: float, params: ModelParameters, q: dict) -> float:
    return params.stress_convert_factor * (
        q["tau_b"] + q["tau_c"] * (
            3.26 * (
                dh(T, params) / params.DEp
                - 0.038 * params.Ek / params.DEp
                + 1.07 * np.sqrt(params.wk)
            ) ** (-1)
            - 2.493 * params.DEp / params.Ek
        )
    )


def tau_kl(T: float, params: ModelParameters, q: dict) -> float:
    return params.stress_convert_factor * (
        q["tau_b"] - q["tau_c"] * (
            params.DEp ** 2 / (0.635 * params.Ek ** 2) * (
                dh(T, params) / params.DEp
                - 1.345 * params.Ek / params.DEp
                + 1.07 * np.sqrt(params.wk)
            )
        )
    )


def tau_k(T: float, params: ModelParameters, q: dict) -> float:
    tau_kh_val = tau_kh(T, params, q)
    tau_b_mpa = q["tau_b"] * params.stress_convert_factor
    if tau_kh_val > tau_b_mpa:
        return tau_kh_val
    return max(tau_kl(T, params, q), 0.0)


def tau_xk_v(T: float, params: ModelParameters, q: dict) -> float:
    ratio = dh(T, params) / params.Ev
    return params.stress_convert_factor * q["tau_kov"] * (1 - ratio ** (2 / 3))


def tau_xk_si(T: float, params: ModelParameters, q: dict) -> float:
    ratio = dh(T, params) / params.Esi
    return params.stress_convert_factor * q["tau_kosi"] * (1 - ratio ** (2 / 3))


def tau_xk_v_ath(T: float, params: ModelParameters, q: dict) -> float:
    return max(tau_xk_v(T, params, q), q["tau_kov_ath"])


def tau_xk_si_ath(T: float, params: ModelParameters, q: dict) -> float:
    return max(tau_xk_si(T, params, q), q["tau_kosi_ath"])


def total_y(T: float, params: ModelParameters, q: dict) -> float:
    """
    CRSS-like total strength in MPa.
    """
    return tau_k(T, params, q) + max(
        tau_xk_v_ath(T, params, q),
        tau_xk_si_ath(T, params, q),
    )


def sigma_y(T: float, params: ModelParameters, q: dict) -> float:
    """
    Yield stress in MPa.
    """
    return total_y(T, params, q) * params.sigma_factor


def predict_strength_dataframe(
    CRSS_SIGMAY: str = "SIGMAY",
    max_T: float = 1000,
    T_interval: float = 50,
    params: ModelParameters | None = None,
) -> pd.DataFrame:
    """
    Build prediction DataFrame over temperature.
    """
    mode = validate_mode(CRSS_SIGMAY)
    params = ModelParameters() if params is None else params
    q = compute_derived_quantities(params)
    T_values = generate_temperature_range(max_T=max_T, T_interval=T_interval)

    if mode == "CRSS":
        y_values = [total_y(T, params, q) for T in T_values]
        value_col = "CRSS (MPa)"
    else:
        y_values = [sigma_y(T, params, q) for T in T_values]
        value_col = "sigma_y (MPa)"

    return q['zeta_c'], pd.DataFrame({
        "Temperature (K)": T_values,
        value_col: y_values,
    })


def save_results(
    df: pd.DataFrame,
    CRSS_SIGMAY: str = "SIGMAY",
    csv_output_name: str = "my_prediction.csv",
    png_output_name: str = "my_prediction.png",
) -> None:
    """
    Save prediction DataFrame to CSV and PNG.
    """
    mode = validate_mode(CRSS_SIGMAY)

    if mode == "CRSS":
        y_col = "CRSS (MPa)"
        y_label = "CRSS (MPa)"
    else:
        y_col = "sigma_y (MPa)"
        y_label = r"$\sigma_y$ (MPa)"

    df.to_csv(csv_output_name, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df["Temperature (K)"], df[y_col], marker="o")
    plt.xlabel("Temperature (K)")
    plt.ylabel(y_label)
    plt.title(f"{mode} vs Temperature")
    plt.xlim(left=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_output_name, dpi=300)
    plt.close()


def screw_strength_prediction(
    CRSS_SIGMAY: str = "SIGMAY",
    max_T: float = 1000,
    T_interval: float = 50,
    csv_output_name: str = "my_prediction.csv",
    png_output_name: str = "my_prediction.png",
    params: ModelParameters | None = None,
) -> pd.DataFrame:
    """
    Main user-facing function.

    Parameters
    ----------
    CRSS_SIGMAY : str
        Either 'CRSS' or 'SIGMAY'. Default is 'SIGMAY'.
    max_T : float
        Maximum temperature in K. Default is 1000.
    T_interval : float
        Temperature interval in K. Default is 50.
    csv_output_name : str
        CSV output file name. Default is 'my_prediction.csv'.
    png_output_name : str
        PNG output file name. Default is 'my_prediction.png'.
    params : ModelParameters or None
        Model parameters. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        Prediction DataFrame.
    """
    zeta_c, df = predict_strength_dataframe(
        CRSS_SIGMAY=CRSS_SIGMAY,
        max_T=max_T,
        T_interval=T_interval,
        params=params,
    )
    save_results(
        df=df,
        CRSS_SIGMAY=CRSS_SIGMAY,
        csv_output_name=csv_output_name,
        png_output_name=png_output_name,
    )

    print(f"zeta_c/b: {zeta_c}")
    print(f"Saved CSV: {csv_output_name}")
    print(f"Saved PNG: {png_output_name}")
    return df
