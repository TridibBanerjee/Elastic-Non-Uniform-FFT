# Module_Ems.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# This module stores the reusable elastic mode selection algorithm.

import numpy as np


delta = 0.02
w1 = 0.5
w2 = 0.5
alpha_min = 0.0
alpha_max = 0.7
k_min = 1
k_max = 12


# Compute the EMS retained count and all intermediate diagnostics for one unordered spectrum.
def elastic_mode_selection(e_values, k_min=k_min, k_max=k_max, alpha_min=alpha_min, alpha_max=alpha_max, delta=delta, w1=w1, w2=w2):
    e_sorted = np.sort(e_values)[::-1]
    j_star = len(e_values)
    sum_e = np.sum(e_sorted)
    n_eff = sum_e ** 2 / np.sum(e_sorted ** 2)
    n_eff_clip = min(n_eff, k_max)
    j_window = min(k_max, j_star)
    g = e_sorted[:j_window - 1] / e_sorted[1:j_window]
    if j_window > 1:
        s_delta = (1 / (j_window - 1)) * np.sum(np.exp(-(g - 1) / delta))
    else:
        s_delta = 1.0
    c = w1 * (n_eff_clip / k_max) + w2 * s_delta
    alpha_c = alpha_min + (alpha_max - alpha_min) * c
    f_k = np.cumsum(e_sorted) / sum_e
    k_candidates = np.where(f_k >= alpha_c)[0] + 1
    if len(k_candidates) == 0:
        k_star = k_max
    else:
        k_star = min(k_max, max(k_min, k_candidates[0]))
    alpha_c_final = f_k[k_star - 1]
    return {
        "e_sorted": e_sorted,
        "j_star": j_star,
        "j_max": j_star,
        "j_window": j_window,
        "sum_e": sum_e,
        "n_eff": n_eff,
        "n_eff_clip": n_eff_clip,
        "n_eff_norm": n_eff_clip / k_max,
        "g": g,
        "s_delta": s_delta,
        "c": c,
        "alpha_c": alpha_c,
        "f_k": f_k,
        "k_star": k_star,
        "alpha_c_final": alpha_c_final,
    }


# Select a sparse conjugate-symmetric Fourier spectrum from Parseval pair energies.
def select_sparse_conjugate_modes(spectrum, m_values, n_values, k_min=1, k_max=None, alpha_min=0.0, alpha_max=0.7, delta=0.02, w1=0.5, w2=0.5):
    m_values = np.asarray(m_values, dtype=int)
    n_values = np.asarray(n_values, dtype=int)
    m_lookup = {int(value): index for index, value in enumerate(m_values)}
    n_lookup = {int(value): index for index, value in enumerate(n_values)}
    mode_pairs = []
    energy = []
    for m_mode in m_values:
        for n_mode in n_values:
            if m_mode == 0 and n_mode == 0:
                continue
            if (m_mode > 0) or (m_mode == 0 and n_mode > 0):
                mode_pairs.append((int(m_mode), int(n_mode)))
                energy.append(np.abs(spectrum[m_lookup[int(m_mode)], n_lookup[int(n_mode)]]) ** 2 + np.abs(spectrum[m_lookup[int(-m_mode)], n_lookup[int(-n_mode)]]) ** 2)
    mode_pairs = np.asarray(mode_pairs, dtype=int)
    energy = np.asarray(energy, dtype=float)
    if len(energy) == 0 or float(np.sum(energy)) <= 1e-15:
        return {"spectrum": np.zeros_like(spectrum), "selected_modes": np.zeros((0, 2), dtype=int), "k_star": 0, "k_max": 0, "n_eff": 0.0, "s_delta": 0.0, "c_measure": 0.0, "alpha_c": alpha_min, "power_retained": 0.0, "energy_sorted": energy}
    order = np.argsort(-energy)
    mode_pairs = mode_pairs[order]
    energy = energy[order]
    mode_limit = max(int(np.max(np.abs(m_values))), int(np.max(np.abs(n_values))))
    k_max = mode_limit if k_max is None else int(k_max)
    k_max = int(np.clip(k_max, 1, len(energy)))
    k_min = int(np.clip(k_min, 1, k_max))
    total_energy = float(np.sum(energy))
    n_eff = total_energy ** 2 / max(float(np.sum(energy ** 2)), 1e-15)
    j_window = min(k_max, len(energy))
    if j_window > 1:
        gaps = energy[:j_window - 1] / np.maximum(energy[1:j_window], 1e-15)
        s_delta = float(np.mean(np.exp(-(gaps - 1.0) / max(delta, 1e-6))))
    else:
        s_delta = 1.0
    c_measure = w1 * (min(n_eff, k_max) / k_max) + w2 * s_delta
    alpha_c = float(np.clip(alpha_min + (alpha_max - alpha_min) * c_measure, 0.0, 1.0))
    cumulative_energy = np.cumsum(energy) / total_energy
    k_star = k_max
    for k_value in range(k_min, k_max + 1):
        if cumulative_energy[k_value - 1] >= alpha_c:
            k_star = k_value
            break
    sparse_spectrum = np.zeros_like(spectrum)
    selected_modes = mode_pairs[:k_star]
    for m_mode, n_mode in selected_modes:
        sparse_spectrum[m_lookup[int(m_mode)], n_lookup[int(n_mode)]] = spectrum[m_lookup[int(m_mode)], n_lookup[int(n_mode)]]
        sparse_spectrum[m_lookup[int(-m_mode)], n_lookup[int(-n_mode)]] = spectrum[m_lookup[int(-m_mode)], n_lookup[int(-n_mode)]]
    return {"spectrum": sparse_spectrum, "selected_modes": selected_modes, "k_star": int(k_star), "k_max": int(k_max), "n_eff": float(n_eff), "s_delta": float(s_delta), "c_measure": float(c_measure), "alpha_c": alpha_c, "power_retained": float(cumulative_energy[k_star - 1]), "energy_sorted": energy}
