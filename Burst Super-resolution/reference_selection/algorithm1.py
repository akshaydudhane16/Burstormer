import numpy as np

PATCH_SIZE = 8
COHERENCE_THRESHOLD = 0.234

# Take the horizontal and vertical gradients of an 8 × 8 patch, apply SVD, and calculate the noise-aware quality score
# Patches with insufficient directional coherence receive a score of zero
def compute_patch_score(gx: np.ndarray, gy: np.ndarray) -> float:
    """Measure the strength and directional coherence of gradients in a patch."""

    G = np.column_stack((gx.ravel(), gy.ravel()))

    s1, s2 = np.linalg.svd(G, compute_uv=False)

    if s1 + s2 == 0:
        return 0.0

    R = (s1 - s2) / (s1 + s2)

    if R < COHERENCE_THRESHOLD:
        return 0.0

    return float(s1 * R)


def compute_quality_score(image: np.ndarray) -> float:
    """Calculate the average patch quality score of a single image channel."""

    h, w = image.shape

    gy, gx = np.gradient(image)

    scores = []

    for i in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for j in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            gx_patch = gx[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            gy_patch = gy[i:i + PATCH_SIZE, j:j + PATCH_SIZE]

            score = compute_patch_score(gx_patch, gy_patch)
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def compute_frame_score(frame: np.ndarray) -> float:
    """Calculate the overall quality score by averaging the four RAW channel scores."""

    scores = [compute_quality_score(channel) for channel in frame]

    return float(np.mean(scores))


# Calculate the quality score of every frame and return the index of the highest-scoring frame
def select_reference_frame(burst) -> int:
    """Select the highest-quality reference frame from an input burst of shape (N, 4, H, W)."""

    if hasattr(burst, "detach"):
        burst = burst.detach().cpu().numpy()

    burst = np.asarray(burst, dtype=np.float64)

    if burst.ndim != 4 or burst.shape[0] == 0 or burst.shape[1] != 4:
        raise ValueError("Expected burst shape (N, 4, H, W).")

    if min(burst.shape[2:]) < PATCH_SIZE:
        raise ValueError("Image dimensions must be at least 8x8.")

    if not np.isfinite(burst).all():
        raise ValueError("Burst contains non-finite values.")

    scores = [compute_frame_score(frame) for frame in burst]

    return int(np.argmax(scores))