from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor

from .elasticity import IsotropicElasticity3D, OrthotropicElasticity3D


class IsotropicDamage3D(IsotropicElasticity3D):
    """Isotropic damage material model in 3D.

    This class extends `IsotropicElasticity3D` to incorporate isotropic damage
    with a scalar damage variable $D \\in [0, 1]$.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        d (Callable): Damage evolution function $D(\\kappa, l_c)$.
        d_prime (Callable): Derivative of the damage evolution
            $D'(\\kappa, l_c)$.
        eq_strain (Literal["rankine", "mises"]): Type of equivalent strain
            measure used for damage driving.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption.
        - Two internal state variables (``n_state = 2``):
          $\\kappa$ (damage driving variable) and $D$ (damage variable).
        - Supports batched/vectorized material parameters.

    Info: Isotropic damage model
        The stress is degraded by a scalar damage variable $D$ as

        $$
            \\pmb{\\sigma} = (1 - D) \\, \\mathbb{C} : \\pmb{\\varepsilon}
        $$

        where $\\mathbb{C}$ is the undamaged elastic stiffness tensor.

        The damage is driven by an equivalent strain measure
        $\\tilde{\\varepsilon}$. For ``eq_strain="rankine"``, this is the
        largest principal strain. The history variable $\\kappa$ tracks the
        maximum equivalent strain ever reached:

        $$
            \\kappa_{n+1} = \\max(\\kappa_n,\\, \\tilde{\\varepsilon}_{n+1})
        $$

        and the damage evolves irreversibly as $D = d(\\kappa, l_c)$, where
        the characteristic length $l_c$ is used for fracture energy
        regularization.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        d: Callable,
        d_prime: Callable,
        eq_strain: Literal["rankine", "mises"],
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self.d = d
        self.d_prime = d_prime
        self.n_state = 2
        self.eq_strain: Literal["rankine", "mises"] = eq_strain

    def vectorize(self, n_elem: int) -> IsotropicDamage3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicDamage3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicDamage3D(E, nu, self.d, self.d_prime, self.eq_strain, rho)

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        de0: Tensor,
        cl: Tensor,
        iter: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Performs a strain increment with the isotropic damage model.

        The stress is computed as
        $\\pmb{\\sigma} = (1 - D) \\, \\mathbb{C} : \\pmb{\\varepsilon}$
        and the algorithmic tangent stiffness is

        $$
            C^{\\text{alg}}_{ijkl} = (1 - D)  C_{ijkl}
                - D'(\\kappa, l_c) \\, \\sigma^{\\text{trial}}_{ij} \\,
                  n_k \\, n_l
        $$

        where $\\mathbf{n}$ is the direction of the damage-driving
        principal strain.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 3, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 3, 3)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                - Shape: `(..., 3, 3)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain.
                - Shape: `(..., 1)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                - Shape: `(..., 3, 3)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Newton iteration.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 3, 3, 3, 3)`.
        """
        # Compute total strain
        H_new = (F - torch.eye(H_inc.shape[-1])) + H_inc
        eps_new = 0.5 * (H_new.transpose(-1, -2) + H_new)

        # Extract state variables
        kappa = state[..., 0]
        D = state[..., 1]

        # Initialize solution variables
        stress_new = stress.clone()
        state_new = state.clone()

        # Calculate equivalent strain
        if self.eq_strain == "rankine":
            L, Q = torch.linalg.eigh(eps_new)
            # Find largest eigenvalue by magnitude
            idx = L.abs().argmax(dim=-1, keepdim=True)
            eps_eq = torch.take_along_dim(L, idx, dim=-1).squeeze(-1)
            n = torch.take_along_dim(Q, idx.unsqueeze(-2), dim=-1).squeeze(-1)
        else:
            raise NotImplementedError(
                f"Equivalent strain type '{self.eq_strain}' is not implemented."
            )

        # Update kappa and damage
        kappa_new = torch.maximum(kappa, eps_eq)
        D_new = self.d(kappa_new, cl)
        D_prime = self.d_prime(kappa_new, cl)

        # Update stress
        sigma_trial = torch.einsum("...ijkl,...kl->...ij", self.C, eps_new - de0)
        stress_new = (1 - D_new)[:, None, None] * sigma_trial

        # Update state variables
        state_new[..., 0] = kappa_new
        state_new[..., 1] = D_new

        # Update tangent stiffness
        ddsdde = (1.0 - D_new)[..., None, None, None, None] * self.C
        if iter > 0:
            active = D_new > D
            ddsdde[active] -= D_prime[active, None, None, None, None] * torch.einsum(
                "...ij,...k,...l->...ijkl", sigma_trial[active], n[active], n[active]
            )
        return stress_new, state_new, ddsdde


class AnisotropicDamage3D(OrthotropicElasticity3D):
    """
    Anisotropic damage model for UD composites
    (Hashin-type initiation + Abaqus-like linear softening).

    State variables:
    [delta_ft_max, d_ft, delta_fc_max, d_fc,
        delta_mt_max, d_mt, delta_mc_max, d_mc]
    """

    def __init__(
        self,
        E1,
        E2,
        E3,
        G12,
        G13,
        G23,
        nu12,
        nu13,
        nu23,
        rho,
        # strengths
        Xt,
        Xc,
        Yt,
        Yc,
        S12,
        S13=None,
        S23=None,
        # fracture energies
        G_ft=None,
        G_fc=None,
        G_mt=None,
        G_mc=None,
    ):
        super().__init__(
            E_1=E1,
            E_2=E2,
            E_3=E3,
            nu_12=nu12,
            nu_13=nu13,
            nu_23=nu23,
            G_12=G12,
            G_13=G13,
            G_23=G23,
            rho=rho,
        )

        self.Xt = Xt
        self.Xc = Xc
        self.Yt = Yt
        self.Yc = Yc
        self.S12 = S12

        if S13 is None:
            S13 = S12
        if S23 is None:
            S23 = S12

        self.S13 = S13
        self.S23 = S23

        self.G_ft = G_ft if G_ft is not None else 1.0
        self.G_fc = G_fc if G_fc is not None else 1.0
        self.G_mt = G_mt if G_mt is not None else 1.0
        self.G_mc = G_mc if G_mc is not None else 1.0

        self.n_state = 8

    def vectorize(self, n_elem: int):
        if getattr(self, "is_vectorized", False):
            return self

        def broadcast(x):
            t = torch.as_tensor(x)
            if t.ndim == 0:
                return t.repeat(n_elem)
            return t

        mat = AnisotropicDamage3D(
            E1=broadcast(self.E_1),
            E2=broadcast(self.E_2),
            E3=broadcast(self.E_3),
            G12=broadcast(self.G_12),
            G13=broadcast(self.G_13),
            G23=broadcast(self.G_23),
            nu12=broadcast(self.nu_12),
            nu13=broadcast(self.nu_13),
            nu23=broadcast(self.nu_23),
            rho=broadcast(self.rho),
            Xt=broadcast(self.Xt),
            Xc=broadcast(self.Xc),
            Yt=broadcast(self.Yt),
            Yc=broadcast(self.Yc),
            S12=broadcast(self.S12),
            S13=broadcast(self.S13),
            S23=broadcast(self.S23),
            G_ft=broadcast(self.G_ft),
            G_fc=broadcast(self.G_fc),
            G_mt=broadcast(self.G_mt),
            G_mc=broadcast(self.G_mc),
        )

        mat.is_vectorized = True
        return mat

    def step(
        self,
        H_inc: torch.Tensor,
        F: torch.Tensor,
        sigma: torch.Tensor,
        state: torch.Tensor,
        de0: torch.Tensor,
        cl: torch.Tensor,
        iter: int,
        dl: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        I = torch.eye(H_inc.shape[-1], device=F.device, dtype=F.dtype)
        H_new = (F - I) + H_inc
        eps_new = 0.5 * (H_new.transpose(-1, -2) + H_new)

        delta_ft_max = state[..., 0]
        d_ft_prev = state[..., 1]

        delta_fc_max = state[..., 2]
        d_fc_prev = state[..., 3]

        delta_mt_max = state[..., 4]
        d_mt_prev = state[..., 5]

        delta_mc_max = state[..., 6]
        d_mc_prev = state[..., 7]

        sigma_hat = torch.einsum("...ijkl,...kl->...ij", self.C, eps_new - de0)

        # Per-mode Hashin initiation + linear softening (rate-independent).
        # Each mode: (kind, tension, strength, axial modulus, fracture energy,
        # previous delta_max, previous damage).
        modes = [
            ("fiber", True, self.Xt, self.E_1, self.G_ft, delta_ft_max, d_ft_prev),
            ("fiber", False, self.Xc, self.E_1, self.G_fc, delta_fc_max, d_fc_prev),
            ("matrix", True, self.Yt, self.E_2, self.G_mt, delta_mt_max, d_mt_prev),
            ("matrix", False, self.Yc, self.E_2, self.G_mc, delta_mc_max, d_mc_prev),
        ]
        out = [self.damage_update(m, sigma_hat, eps_new, cl) for m in modes]
        (f_ft, delta_ft, delta_ft_max_new, delta_0_ft, d_ft_new) = out[0]
        (f_fc, delta_fc, delta_fc_max_new, delta_0_fc, d_fc_new) = out[1]
        (f_mt, delta_mt, delta_mt_max_new, delta_0_mt, d_mt_new) = out[2]
        (f_mc, delta_mc, delta_mc_max_new, delta_0_mc, d_mc_new) = out[3]

        # Active fiber/matrix damage depends on the *sign* of the driving
        # stress (Camanho-Davila / Abaqus convention), not on max(d_t, d_c):
        # a specimen damaged in tension must keep its full compressive
        # stiffness (and vice versa) once the load reverses.
        sig11 = sigma_hat[..., 0, 0]
        sig_t2 = sigma_hat[..., 1, 1] + sigma_hat[..., 2, 2]

        d_f = torch.where(sig11 >= 0, d_ft_new, d_fc_new)
        d_m = torch.where(sig_t2 >= 0, d_mt_new, d_mc_new)

        # Shear damage accumulates all modes monotonically (Abaqus/Hashin
        # convention): once shear stiffness is lost to any cracking mode it must
        # not recover when the normal load reverses. Building d_s from the
        # sign-active pair (d_f, d_m) instead would let the shear modulus spring
        # back on reversal, which is unphysical. In-plane shear (12/13) sees all
        # four modes; transverse shear (23) is matrix-dominated.
        d_s12 = 1.0 - (1.0 - d_ft_new) * (1.0 - d_fc_new) * (1.0 - d_mt_new) * (
            1.0 - d_mc_new
        )
        d_s13 = d_s12
        d_s23 = 1.0 - (1.0 - d_mt_new) * (1.0 - d_mc_new)

        C_d, C33 = self.build_damaged_stiffness(d_f, d_m, d_s12, d_s13, d_s23)

        sigma_new = torch.einsum("...ijkl,...kl->...ij", C_d, eps_new - de0)

        state_new = state.clone()
        state_new[..., 0] = delta_ft_max_new
        state_new[..., 1] = d_ft_new
        state_new[..., 2] = delta_fc_max_new
        state_new[..., 3] = d_fc_new
        state_new[..., 4] = delta_mt_max_new
        state_new[..., 5] = d_mt_new
        state_new[..., 6] = delta_mc_max_new
        state_new[..., 7] = d_mc_new

        # ------------------------------------------------------------------
        # Consistent (algorithmic) tangent  d(sigma)/d(eps):
        #
        #     ddsdde = C_d + sum_k (d sigma / d d_k) (x) (d d_k / d eps)
        #
        # The secant C_d is only the first term; the second is the softening
        # contribution that restores ~quadratic Newton convergence. Derived by
        # hand below and checked against autodiff in the tests.
        # ------------------------------------------------------------------
        dtype = eps_new.dtype
        e = eps_new - de0

        def bc(x):  # broadcast a scalar field (...) to (..., 1, 1)
            return x.unsqueeze(-1).unsqueeze(-1)

        # Normal-block inverse C33 (sigma_ii = C33_ij eps_jj), reused from
        # build_damaged_stiffness, and its columns.
        c0, c1, c2 = C33[..., :, 0], C33[..., :, 1], C33[..., :, 2]
        e_n = torch.stack([e[..., 0, 0], e[..., 1, 1], e[..., 2, 2]], dim=-1)
        c0e = (c0 * e_n).sum(-1)
        c1e = (c1 * e_n).sum(-1)
        c2e = (c2 * e_n).sum(-1)

        # d(sigma)/d(damage) blocks. Normal: dC33/dd = -C33 (dS/dd) C33 is rank-1
        # in the affected compliance column. Shear: sigma_ab = 2 G_ab(1-d_s) e_ab.
        dS11 = 1.0 / ((1.0 - d_f) ** 2 * self.E_1)
        dS22 = 1.0 / ((1.0 - d_m) ** 2 * self.E_2)
        dS33 = 1.0 / ((1.0 - d_m) ** 2 * self.E_3)
        Sig_df = torch.diag_embed((-dS11 * c0e).unsqueeze(-1) * c0)
        Sig_dm = torch.diag_embed(
            (-dS22 * c1e).unsqueeze(-1) * c1 + (-dS33 * c2e).unsqueeze(-1) * c2
        )
        Sig_ds12 = torch.zeros_like(Sig_df)
        Sig_ds13 = torch.zeros_like(Sig_df)
        Sig_ds23 = torch.zeros_like(Sig_df)
        Sig_ds12[..., 0, 1] = -2.0 * self.G_12 * e[..., 0, 1]
        Sig_ds12[..., 1, 0] = Sig_ds12[..., 0, 1]
        Sig_ds13[..., 0, 2] = -2.0 * self.G_13 * e[..., 0, 2]
        Sig_ds13[..., 2, 0] = Sig_ds13[..., 0, 2]
        Sig_ds23[..., 1, 2] = -2.0 * self.G_23 * e[..., 1, 2]
        Sig_ds23[..., 2, 1] = Sig_ds23[..., 1, 2]

        # Chain d(d_f, d_m, d_s)/d(d_k) for the four fundamental damage variables.
        s11p = (sig11 >= 0).to(dtype)
        st2p = (sig_t2 >= 0).to(dtype)
        omft, omfc = 1.0 - d_ft_new, 1.0 - d_fc_new
        ommt, ommc = 1.0 - d_mt_new, 1.0 - d_mc_new
        Ss = Sig_ds12 + Sig_ds13  # in-plane shear sees all modes (d_s12 == d_s13)
        T_ft = bc(s11p) * Sig_df + bc(omfc * ommt * ommc) * Ss
        T_fc = bc(1.0 - s11p) * Sig_df + bc(omft * ommt * ommc) * Ss
        T_mt = bc(st2p) * Sig_dm + bc(omft * omfc * ommc) * Ss + bc(ommc) * Sig_ds23
        T_mc = (
            bc(1.0 - st2p) * Sig_dm + bc(omft * omfc * ommt) * Ss + bc(ommt) * Sig_ds23
        )

        # d(d_k)/d(eps) = d(damage_law)/d(delta) * d(delta_eq)/d(eps),
        # gated to the active + loading + unclamped regime.
        def dmg_factor(
            delta_cur, delta_prev, delta_max_new, delta_0, f, strength, Gp, d_new
        ):
            delta_f = 2.0 * Gp / strength
            dl = delta_f * delta_0 / ((delta_f - delta_0) * delta_max_new**2 + 1e-30)
            active = (delta_max_new >= delta_0) & (f >= 1.0) & (delta_f > delta_0)
            mask = (
                active & (delta_cur >= delta_prev) & (d_new > 0.0) & (d_new < 0.99)
            ).to(dtype)
            return dl * mask

        cf_ft = dmg_factor(
            delta_ft,
            delta_ft_max,
            delta_ft_max_new,
            delta_0_ft,
            f_ft,
            self.Xt,
            self.G_ft,
            d_ft_new,
        )
        cf_fc = dmg_factor(
            delta_fc,
            delta_fc_max,
            delta_fc_max_new,
            delta_0_fc,
            f_fc,
            self.Xc,
            self.G_fc,
            d_fc_new,
        )
        cf_mt = dmg_factor(
            delta_mt,
            delta_mt_max,
            delta_mt_max_new,
            delta_0_mt,
            f_mt,
            self.Yt,
            self.G_mt,
            d_mt_new,
        )
        cf_mc = dmg_factor(
            delta_mc,
            delta_mc_max,
            delta_mc_max_new,
            delta_0_mc,
            f_mc,
            self.Yc,
            self.G_mc,
            d_mc_new,
        )

        # d(delta_eq)/d(eps), symmetric. FT/FC axial; MT/MC use engineering shear
        # (gamma = 2 eps), so off-diagonal entries carry the factor 2.
        cl2 = cl**2
        Dmt = delta_mt + 1e-30
        Dmc = delta_mc + 1e-30
        gd_ft = torch.zeros_like(Sig_df)
        gd_fc = torch.zeros_like(Sig_df)
        gd_mt = torch.zeros_like(Sig_df)
        gd_mc = torch.zeros_like(Sig_df)
        gd_ft[..., 0, 0] = cl * (eps_new[..., 0, 0] > 0).to(dtype)
        gd_fc[..., 0, 0] = -cl * (eps_new[..., 0, 0] < 0).to(dtype)
        gd_mt[..., 1, 1] = cl2 * torch.relu(eps_new[..., 1, 1]) / Dmt
        gd_mt[..., 2, 2] = cl2 * torch.relu(eps_new[..., 2, 2]) / Dmt
        gd_mc[..., 1, 1] = -cl2 * torch.relu(-eps_new[..., 1, 1]) / Dmc
        gd_mc[..., 2, 2] = -cl2 * torch.relu(-eps_new[..., 2, 2]) / Dmc
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            gd_mt[..., a, b] = cl2 * 2.0 * eps_new[..., a, b] / Dmt
            gd_mt[..., b, a] = gd_mt[..., a, b]
            gd_mc[..., a, b] = cl2 * 2.0 * eps_new[..., a, b] / Dmc
            gd_mc[..., b, a] = gd_mc[..., a, b]

        # Secant tangent (C_d, positive definite) on the first Newton iteration
        # for robustness, then the consistent softening tangent afterwards. The
        # softening term makes C_alg indefinite, so using it on iteration 0 makes
        # the first step overshoot at sharp localization; this mirrors the
        # `if iter > 0` guard in IsotropicDamage3D.
        ddsdde = C_d
        if iter > 0:
            ddsdde = (
                ddsdde
                + torch.einsum("...ij,...kl->...ijkl", T_ft, bc(cf_ft) * gd_ft)
                + torch.einsum("...ij,...kl->...ijkl", T_fc, bc(cf_fc) * gd_fc)
                + torch.einsum("...ij,...kl->...ijkl", T_mt, bc(cf_mt) * gd_mt)
                + torch.einsum("...ij,...kl->...ijkl", T_mc, bc(cf_mc) * gd_mc)
            )

        return sigma_new, state_new, ddsdde

    # ============================================================
    # Per-mode damage update (Hashin initiation + linear softening)
    # ============================================================

    def damage_update(self, spec, sigma_hat, eps, cl):
        """Failure index, equivalent displacement and updated damage for one mode.

        `spec` is ``(kind, tension, strength, modulus, G, delta_max, d_prev)``:
        `kind` selects the fiber/matrix criteria, `tension` the tensile/compressive
        branch. Damage is rate-independent and monotonic (irreversible).
        Returns ``(f, delta, delta_max_new, delta_0, d_new)``.
        """
        kind, tension, strength, modulus, G, delta_max, d_prev = spec

        if kind == "fiber":
            f = self.hashin_fiber(sigma_hat, strength, tension)
            delta = self.eq_disp_fiber(eps, cl, tension)
        else:
            f = self.hashin_matrix(sigma_hat, strength, tension)
            delta = self.eq_disp_matrix(eps, cl, tension)

        delta_max_new = torch.maximum(delta_max, delta)
        delta_0 = cl * strength / modulus

        d_new = torch.maximum(
            d_prev, self.damage_law(delta_max_new, delta_0, f, strength, G)
        )
        d_new = torch.clamp(d_new, 0.0, 0.99)
        return f, delta, delta_max_new, delta_0, d_new

    # ============================================================
    # Hashin criteria
    # ============================================================

    def hashin_fiber(self, sigma_hat, strength, tension):
        """Fiber failure index. Tension adds the in-plane shear terms; each branch
        is gated on the sign of the fiber-direction stress."""
        sig11 = sigma_hat[..., 0, 0]
        if tension:
            tau12 = sigma_hat[..., 0, 1]
            tau13 = sigma_hat[..., 0, 2]
            f = (
                (sig11 / strength) ** 2
                + (tau12 / self.S12) ** 2
                + (tau13 / self.S13) ** 2
            )
            return torch.where(sig11 > 0, f, torch.zeros_like(f))
        f = (sig11 / strength) ** 2
        return torch.where(sig11 < 0, f, torch.zeros_like(f))

    def hashin_matrix(self, sigma_hat, strength, tension):
        """Matrix failure index. `tension` picks the positive/negative part of the
        transverse normal stresses; the shear terms are common to both branches."""
        sig22 = sigma_hat[..., 1, 1]
        sig33 = sigma_hat[..., 2, 2]
        tau12 = sigma_hat[..., 0, 1]
        tau13 = sigma_hat[..., 0, 2]
        tau23 = sigma_hat[..., 1, 2]
        if tension:
            sig_n = torch.sqrt(torch.relu(sig22) ** 2 + torch.relu(sig33) ** 2)
        else:
            sig_n = torch.sqrt(torch.relu(-sig22) ** 2 + torch.relu(-sig33) ** 2)
        return (
            (sig_n / strength) ** 2
            + (tau12 / self.S12) ** 2
            + (tau13 / self.S13) ** 2
            + (tau23 / self.S23) ** 2
        )

    # ============================================================
    # Equivalent displacement
    # ============================================================

    def eq_disp_fiber(self, eps, cl, tension):
        eps11 = eps[..., 0, 0]
        eps11 = torch.relu(eps11) if tension else torch.relu(-eps11)
        return cl * eps11

    def eq_disp_matrix(self, eps, cl, tension):
        eps22 = eps[..., 1, 1]
        eps33 = eps[..., 2, 2]
        gam12 = 2.0 * eps[..., 0, 1]
        gam13 = 2.0 * eps[..., 0, 2]
        gam23 = 2.0 * eps[..., 1, 2]
        if tension:
            n22, n33 = torch.relu(eps22), torch.relu(eps33)
        else:
            n22, n33 = torch.relu(-eps22), torch.relu(-eps33)
        eps_eq = torch.sqrt(n22**2 + n33**2 + gam12**2 + gam13**2 + gam23**2)
        return cl * eps_eq

    # ============================================================
    # Damage evolution laws
    # ============================================================

    def damage_law(self, delta_max, delta_0, f, strength, G):
        """Bilinear (linear-softening) damage, shared by all four modes.

        ``d = delta_f (delta_max - delta_0) / (delta_max (delta_f - delta_0))``,
        active only once the criterion is met (``f >= 1``) and the equivalent
        displacement has passed the onset value ``delta_0``.
        """
        eps = 1e-12 * (delta_0.abs() + 1.0)
        delta_f = 2.0 * G / (strength + eps)

        active = (delta_max >= delta_0) & (f >= 1.0) & (delta_f > delta_0 + eps)

        num = delta_f * (delta_max - delta_0)
        den = delta_max * (delta_f - delta_0 + eps)

        d = num / (den + eps)
        d = torch.where(active, d, torch.zeros_like(d))
        return torch.clamp(d, 0.0, 0.99)

    # ============================================================
    # Damaged stiffness
    # ============================================================

    def build_damaged_stiffness(
        self,
        d_f: torch.Tensor,
        d_m: torch.Tensor,
        d_s12: torch.Tensor,
        d_s13: torch.Tensor,
        d_s23: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble the damaged stiffness by degrading the *compliance* and
        re-inverting it (Camanho-Davila / Lapczyk-Hurtado formulation, the
        same one used by Abaqus' built-in Hashin damage model), rather than
        scaling the stiffness tensor entries directly.

        Only the direct (axial) compliance terms 1/E_i are reduced by
        damage; the Poisson (off-diagonal) compliance terms are left
        untouched. Re-inverting the 3x3 normal block then automatically
        produces the correct, coupled reduction of *all* normal stiffness
        terms -- including the 1/D re-normalization familiar from the
        Abaqus lamina formula -- which a naive entry-wise scaling of C
        cannot reproduce (it under/over-estimates stiffness by >15% already
        at 50% damage for typical Poisson ratios).
        """
        kmin = 1e-9
        d_f = torch.clamp(d_f, 0.0, 1.0 - kmin)
        d_m = torch.clamp(d_m, 0.0, 1.0 - kmin)

        S11 = 1.0 / ((1.0 - d_f) * self.E_1)
        S22 = 1.0 / ((1.0 - d_m) * self.E_2)
        S33 = 1.0 / ((1.0 - d_m) * self.E_3)
        S12 = -self.nu_12 / self.E_1
        S13 = -self.nu_13 / self.E_1
        S23 = -self.nu_23 / self.E_2

        S11, S22, S33, S12, S13, S23 = torch.broadcast_tensors(
            S11, S22, S33, S12, S13, S23
        )
        batch_shape = S11.shape

        S = torch.zeros(*batch_shape, 3, 3, dtype=S11.dtype, device=S11.device)
        S[..., 0, 0] = S11
        S[..., 1, 1] = S22
        S[..., 2, 2] = S33
        S[..., 0, 1] = S12
        S[..., 1, 0] = S12
        S[..., 0, 2] = S13
        S[..., 2, 0] = S13
        S[..., 1, 2] = S23
        S[..., 2, 1] = S23

        # Batched 3x3 inversion of the normal (axial) block
        C33 = torch.linalg.inv(S)

        G12_d = (1.0 - d_s12) * self.G_12
        G13_d = (1.0 - d_s13) * self.G_13
        G23_d = (1.0 - d_s23) * self.G_23
        G12_d, G13_d, G23_d = torch.broadcast_tensors(G12_d, G13_d, G23_d)

        C_d = torch.zeros(*batch_shape, 3, 3, 3, 3, dtype=S11.dtype, device=S11.device)
        for i in range(3):
            for j in range(3):
                C_d[..., i, i, j, j] = C33[..., i, j]

        C_d[..., 0, 1, 0, 1] = G12_d
        C_d[..., 1, 0, 1, 0] = G12_d
        C_d[..., 0, 1, 1, 0] = G12_d
        C_d[..., 1, 0, 0, 1] = G12_d

        C_d[..., 0, 2, 0, 2] = G13_d
        C_d[..., 2, 0, 2, 0] = G13_d
        C_d[..., 0, 2, 2, 0] = G13_d
        C_d[..., 2, 0, 0, 2] = G13_d

        C_d[..., 1, 2, 1, 2] = G23_d
        C_d[..., 2, 1, 2, 1] = G23_d
        C_d[..., 1, 2, 2, 1] = G23_d
        C_d[..., 2, 1, 1, 2] = G23_d

        # C33 (the inverted normal block) is returned as well so the consistent
        # tangent in step() can reuse it instead of re-extracting it from C_d.
        return C_d, C33
