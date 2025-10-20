import numpy as np

def four_vector_from_pt_eta_phi_e(pt, eta, phi, e, padding_value = -999):
    """
    Computes the four-momentum vector components from pt, eta, phi, and energy.

    Args:   
        pt (np.ndarray): Transverse momentum.
        eta (np.ndarray): Pseudorapidity.
        phi (np.ndarray): Azimuthal angle.
        e (np.ndarray): Energy.
    Returns:
        tuple: A tuple containing the four-momentum components (px, py, pz, e).
    """
    mask = (pt == padding_value) | (eta == padding_value) | (phi == padding_value) | (e == padding_value)
    phi = np.where(mask, 0, phi)
    eta = np.where(mask, 0, eta)
    pt = np.where(mask, 0, pt)
    e = np.where(mask, padding_value, e)
    px = np.where(mask, padding_value, pt * np.cos(phi))
    py = np.where(mask, padding_value, pt * np.sin(phi))
    pz = np.where(mask, padding_value, pt * np.sinh(eta))
    return px, py, pz, e

def compute_mass_from_four_vector(px, py, pz, e, padding_value = -999):
    """
    Computes the invariant mass from four-momentum components.

    Args:
        px (np.ndarray): x-component of momentum.
        py (np.ndarray): y-component of momentum.
        pz (np.ndarray): z-component of momentum.
        e (np.ndarray): Energy.
    Returns:
        np.ndarray: The invariant mass.
    """
    mask = (px == padding_value) | (py == padding_value) | (pz == padding_value) | (e == padding_value)
    px = np.where(mask, 0, px)
    py = np.where(mask, 0, py)
    pz = np.where(mask, 0, pz)
    mass_squared = np.where(mask, padding_value, e**2 - (px**2 + py**2 + pz**2))
    mass_squared = np.maximum(mass_squared, 0)  # Prevent negative values due to numerical errors
    return np.sqrt(mass_squared)