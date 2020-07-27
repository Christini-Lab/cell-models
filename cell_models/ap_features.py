def get_ap_duration(t, v_m):
    """
    Accepts t and v_m as Pandas Series objects
    """
    dvdt_max_idx = (v_m.diff() / t.diff()).idxmax()

    v_min = v_m.min()
    v_max = v_m.max()

    v_90 = v_max - ((v_max - v_min) * .9)

    v_idxmax = v_m.idxmax()
    v_90_idx = (v_m[v_idxmax:] - v_90).abs().idxmin()

    apd_90 = t[v_90_idx] - t[dvdt_max_idx]

    return apd_90

def get_ap_amplitude(t, v_m):
    v_min = v_m.min()
    v_max = v_m.max()

    return v_max - v_min

