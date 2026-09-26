import numpy as np

from kerr_sbi.radius import RadiusMesh, radius_prior_weights


def integrate_radius_slabs(
    mesh: RadiusMesh,
    observation: np.ndarray,
    sigma: float,
    radius_max: float,
    shape: list[int],
    collect_scores: bool = False,
) -> dict:
    if len(shape) != 3 or any(n < 2 for n in shape):
        raise ValueError("three quadrature sizes >= 2 required")
    axes = [
        mesh.bounds[0, j] + (np.arange(n) + 0.5) / n * mesh.width[j] for j, n in enumerate(shape)
    ]
    spin = np.zeros((3, 1, shape[0]))
    inclination = np.zeros((3, 1, shape[1]))
    joint = np.zeros((3, 1, shape[0], shape[2]))
    scores = np.zeros((3, len(mesh.gram))) if collect_scores else None
    ct = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 2)
    offset = -np.inf
    for index, a in enumerate(axes[0]):
        points = np.column_stack([np.full(len(ct), a), ct])
        likelihood = mesh.log_likelihood(points, observation[None], sigma)[0]
        next_offset = max(offset, float(likelihood.max()))
        rescale = np.exp(offset - next_offset)
        spin *= rescale
        inclination *= rescale
        joint *= rescale
        if scores is not None:
            scores *= rescale
        offset = next_offset
        weighted = radius_prior_weights(points, radius_max) * np.exp(likelihood - offset)
        slab = weighted.reshape(3, shape[1], shape[2])
        spin[:, 0, index] = slab.sum(axis=(1, 2))
        inclination[:, 0] += slab.sum(axis=2)
        joint[:, 0, index] = slab.sum(axis=1)
        if scores is not None:
            simplex, _ = mesh.weights(points)
            for prior in range(3):
                scores[prior] += np.bincount(
                    simplex, weights=weighted[prior], minlength=len(mesh.gram)
                )
    normalization = spin.sum(axis=-1, keepdims=True)
    if not np.isfinite(normalization).all() or np.any(normalization <= 0):
        raise ValueError("quadrature has no finite probability mass")
    spin /= normalization
    inclination /= normalization
    joint /= normalization[..., None]
    if scores is not None:
        scores /= normalization[:, 0]
    return {
        "axes": axes,
        "spin": spin,
        "inclination": inclination,
        "joint": joint,
        "simplex_scores": scores,
    }


def draw_radius_checks(
    mesh: RadiusMesh,
    observation: np.ndarray,
    sigma: float,
    integrated: dict,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a, c, t = integrated["axes"]
    joint = integrated["joint"].mean(axis=(0, 1))
    selected = rng.choice(joint.size, count, p=(joint / joint.sum()).ravel())
    result = []
    for index in selected:
        ia, it = np.unravel_index(index, joint.shape)
        points = np.column_stack([np.full(len(c), a[ia]), c, np.full(len(c), t[it])])
        likelihood = mesh.log_likelihood(points, observation[None], sigma)[0]
        probability = np.exp(likelihood - likelihood.max())
        ic = rng.choice(len(c), p=probability / probability.sum())
        result.append([a[ia], c[ic], t[it]])
    widths = mesh.width / np.array([len(axis) for axis in (a, c, t)])
    return np.array(result) + rng.uniform(-0.5, 0.5, (count, 3)) * widths


def targeted_radius_points(
    mesh: RadiusMesh,
    scores: np.ndarray,
    diagnostic_points: np.ndarray,
    diagnostic_errors: np.ndarray,
    count: int,
    boundary_fraction: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(mesh.gram),) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("finite nonnegative simplex scores required")
    if count < 1 or not 0 < boundary_fraction < 1:
        raise ValueError("positive count and interior boundary fraction required")
    diagnostic_points = np.asarray(diagnostic_points, dtype=float).reshape(-1, 3)
    diagnostic_errors = np.asarray(diagnostic_errors, dtype=float)
    if (
        diagnostic_errors.shape != (len(diagnostic_points),)
        or not np.isfinite(diagnostic_errors).all()
        or np.any(diagnostic_errors < 0)
    ):
        raise ValueError("matching finite nonnegative diagnostic errors required")
    selected, reasons = [], []
    known = set(map(tuple, np.round((mesh.points - mesh.bounds[0]) / mesh.width, 12)))

    def add(point: np.ndarray, reason: str) -> None:
        point = np.clip(point, *mesh.bounds)
        key = tuple(np.round((point - mesh.bounds[0]) / mesh.width, 12))
        if key not in known and len(selected) < count:
            known.add(key)
            selected.append(point)
            reasons.append(reason)

    order = np.argsort(-diagnostic_errors, kind="stable")
    for index in order[: count // 4]:
        if diagnostic_errors[index] > 0.125:
            add(diagnostic_points[index], "measured_image_error")
    triangulation = mesh.triangulation
    parent, opposite = np.where(triangulation.neighbors == -1)
    valid_parent = np.isfinite(triangulation.transform[parent]).all(axis=(1, 2))
    parent, opposite = parent[valid_parent], opposite[valid_parent]
    face_indices = np.array(
        [
            np.delete(triangulation.simplices[cell], vertex)
            for cell, vertex in zip(parent, opposite, strict=True)
        ]
    )
    faces = mesh.points[face_indices]
    centers = faces.mean(axis=1)
    for dimension in range(3):
        for edge in mesh.bounds[:, dimension]:
            centers[np.all(faces[:, :, dimension] == edge, axis=1), dimension] = edge
    face_order = np.argsort(-scores[parent], kind="stable")
    isco = np.all(faces[:, :, 2] == 0, axis=1)
    face_quota = int(count * boundary_fraction)
    for group, quota, reason in (
        (face_order[isco[face_order]], face_quota // 2, "isco_face_mass"),
        (face_order[~isco[face_order]], face_quota - face_quota // 2, "other_boundary_mass"),
    ):
        before = len(selected)
        for index in group:
            add(centers[index], reason)
            if len(selected) - before >= quota:
                break
    priority = scores.copy()
    valid = np.isfinite(mesh.triangulation.transform).all(axis=(1, 2))
    priority[~valid] = -np.inf
    if len(diagnostic_points):
        diagnostic_simplex, _ = mesh.weights(diagnostic_points)
        for simplex, error in zip(diagnostic_simplex, diagnostic_errors, strict=True):
            priority[simplex] *= 1 + (error / 0.25) ** 2
    for index in np.argsort(-priority, kind="stable"):
        if valid[index]:
            add(mesh.points[mesh.triangulation.simplices[index]].mean(axis=0), "mass_times_error")
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("not enough distinct refinement candidates")
    return np.array(selected), reasons
