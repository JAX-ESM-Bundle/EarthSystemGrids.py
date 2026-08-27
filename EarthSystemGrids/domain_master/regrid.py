"""
Regridder-callable builders: a flat-(nface,)-array regridder from a
pre-generated ESMF weight file, and a same-grid masking regridder for
transformations that are really just a mask, not an interpolation.
"""

from typing import Callable

import numpy as np
import xarray as xr


def _make_esmf_regridder(weight_file, *, frac_b_threshold: float = 0.0) -> Callable:
    """
    Build a shape-preserving regridder callable from an ESMF weight file,
    backed by EarthSystemGrids.base.esmf_regrid.ESMFRegridder.

    Applies the weight file's row/col/S to the flat (n_a,) array this
    repo's meshes already use (Domain.grid.face_lon/mask/area/etc.) -- a
    flat (nface,) input comes back flat (nface,). A 2D (n_lon, n_lat)
    input -- JEM's SlabGrid / ESMFRegridder.__call__ convention -- is also
    accepted, and comes back 2D (n_lon, n_lat): it's Fortran-flattened
    before regridding and Fortran-reshaped back to the weight file's own
    dst_shape after, which needs no transpose because the two conventions
    are the same underlying bytes -- this repo's SCRIP writer flattens
    each mesh's face arrays in C order over its own (nj, ni) = (lat, lon)
    logical shape, which is byte-identical to Fortran-order flattening a
    (ni, nj) = (n_lon, n_lat) array (lon varies fastest either way; see
    StructuredQuadMesh.write_to_SCRIP_grid_file's grid_dims=[ni,nj]). This
    makes the callable a drop-in for jem.utils.esmf_regrid.ESMFRegridder
    wherever a caller wants DomainMaster's registered transformation
    instead of constructing its own ESMFRegridder from the same weight
    file.

    frac_b isn't exposed on ESMFWeights, so it's read separately here.
    Destination cells with zero source coverage (frac_b <= frac_b_threshold)
    come back NaN, so check_coverage can see them as holes; partial
    coverage (0 < frac_b < 1) is kept as a legitimate area-weighted value.

    jax is imported lazily here (not at module top), so DomainMaster and
    this module import cleanly without jax installed as long as only
    custom (non-weight_file) regridders are actually used.
    """
    from EarthSystemGrids.base.esmf_regrid import ESMFRegridder
    import jax.numpy as jnp

    r = ESMFRegridder(weight_file)
    w = r.weights
    with xr.open_dataset(weight_file) as ds:
        frac_b = jnp.asarray(ds["frac_b"].values)

    def regridder(field_a):
        field_a = jnp.asarray(field_a)
        is_2d = field_a.ndim == 2
        flat_a = field_a.reshape(-1, order="F") if is_2d else field_a
        dst = jnp.zeros(w.dst_size, dtype=flat_a.dtype).at[w.row_indices].add(
            flat_a[w.col_indices] * w.weights
        )
        dst = jnp.where(frac_b <= frac_b_threshold, jnp.nan, dst)
        return dst.reshape(w.dst_shape, order="F") if is_2d else dst

    return regridder


def make_mask_regridder(mask, fill_value: float = np.nan) -> Callable:
    """
    Build a same-grid "masking" regridder: field_a -> where(mask, field_a, fill_value).

    For transformations where source and target share the same underlying
    grid (e.g. splitting a shared exchange grid into separate ocean/land
    views) -- no interpolation, just keeping the cells `mask` selects and
    replacing the rest with `fill_value`. `mask` is typically a domain's
    own `mask` field (1 = active, 0 = inactive; or any 0/1-or-bool array
    the same size as the grid); truthy entries are kept, falsy entries
    become `fill_value`.

    `mask` may also be a zero-argument callable returning the mask array,
    e.g. `lambda: dm.domains["ocn"].mask` -- resolved lazily, on the first
    call to the returned regridder rather than at make_mask_regridder()
    call time, and cached from then on. This lets a masking transformation
    be registered (`register_transformation(..., regridder=make_mask_regridder(
    lambda: dm.domains["ocn"].mask))`) before the domain it reads the mask
    from has one set yet -- the same "declare now, resolve later" pattern
    DomainMaster's own placeholder domains/transformations already follow.
    A callable that still returns None (or raises, e.g. because the domain
    isn't registered yet) only surfaces as an error at that first real
    call -- transform_scalar/transform_vector, or validate()'s functional
    check, not here.

    `fill_value` defaults to NaN, matching the "not covered by this
    source" convention check_coverage already reads (via each ESMF weight
    file's frac_b) -- so a set of masking transformations built this way
    (e.g. one per land/ocean view of a shared exchange grid) can be
    checked directly with check_coverage to confirm the masks partition
    the shared grid exactly, with no gaps or overlap.

    The returned regridder runs `jnp.where` (jax lazily imported here, as
    in `_make_esmf_regridder` above, so this module stays importable
    without jax as long as only eager/numpy callers are used) rather than
    `np.where`, so it works inside a jax.jit/grad trace -- needed for
    callers like a "source"->"exchange" transformation invoked from
    within a jitted coupling step. `mask` itself is still resolved and
    cached as plain numpy (it's a concrete, non-traced domain field, not
    something that varies under tracing).

    Shape-preserving like `_make_esmf_regridder`: `mask` is always flat
    (nface,) (a `Domain.mask`'s own convention), but `field_a` may be flat
    (nface,) or 2D (n_lon, n_lat) (e.g. a Veros field, which is natively
    2D) -- auto-detected from `field_a.ndim`. For a 2D `field_a`, the flat
    mask is reshaped to `field_a.shape` with `order="F"` before the
    `where` (byte-identical to the mask's own flat layout -- see
    `_make_esmf_regridder`'s docstring for why -- so this needs no
    transpose, just a shape wrapper matching `field_a`'s own).

    Raises ValueError if a concrete (non-callable) `mask` is None -- e.g.
    a domain's own `mask` field that was never set, passed directly rather
    than via a lazy callable. Silently accepting None would otherwise turn
    into a 0-d False mask that NaNs out every cell, which is exactly the
    kind of silent, all-cells-empty bug this should fail loudly on
    instead. A callable is exempt from this eager check by design -- that
    is exactly the deferred-resolution case above.
    """
    import jax.numpy as jnp

    if not callable(mask) and mask is None:
        raise ValueError("make_mask_regridder needs a mask, got None")

    resolved = {}

    def _resolved_mask():
        if "value" not in resolved:
            m = mask() if callable(mask) else mask
            if m is None:
                raise ValueError("make_mask_regridder's mask callable returned None")
            resolved["value"] = np.asarray(m).astype(bool)
        return resolved["value"]

    if not callable(mask):
        _resolved_mask()  # eager: resolve (and cache) immediately, as before

    def regridder(field_a):
        field_a = jnp.asarray(field_a)
        mask = _resolved_mask()
        if field_a.ndim == 2:
            mask = mask.reshape(field_a.shape, order="F")
        return jnp.where(mask, field_a, fill_value)

    return regridder
