import numpy as np
import pytest

from reference_selection.algorithm1 import select_reference_frame


def create_sharp_image():
    """Generate an image containing sharp vertical edges."""

    x = np.arange(80)

    return np.tile(
        np.where((x // 8) % 2 == 0, 0.25, 0.75),
        (80, 1)
    )


def create_blurred_image(image):
    """Blur an image using a horizontal averaging filter."""

    kernel = np.ones(7) / 7

    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=1,
        arr=image
    )


def create_burst(image):
    """Generate a 14-frame burst with four identical RAW channels."""

    return np.tile(image, (14, 4, 1, 1))


def test_identical_frames():
    burst = create_burst(create_sharp_image())

    assert select_reference_frame(burst) == 0


def test_blurred_reference():
    image = create_sharp_image()
    burst = create_burst(image)

    burst[0] = create_blurred_image(image)

    assert select_reference_frame(burst) == 1


def test_noisy_reference():
    burst = create_burst(create_sharp_image())
    rng = np.random.default_rng(42)

    burst[0] += rng.normal(0, 0.1, burst[0].shape)

    assert select_reference_frame(burst) == 1


def test_single_best_frame():
    image = create_sharp_image()
    burst = create_burst(create_blurred_image(image))

    burst[7] = image

    assert select_reference_frame(burst) == 7


def test_constant_images():
    burst = np.zeros((14, 4, 80, 80))

    assert select_reference_frame(burst) == 0


def test_negative_raw_values():
    burst = create_burst(create_sharp_image()) - 0.5

    assert select_reference_frame(burst) == 0


def test_invalid_dimensions():
    burst = np.zeros((14, 3, 80, 80))

    with pytest.raises(ValueError):
        select_reference_frame(burst)


def test_input_unchanged():
    burst = create_burst(create_sharp_image())
    original = burst.copy()

    select_reference_frame(burst)

    np.testing.assert_array_equal(burst, original)


def test_torch_input():
    torch = pytest.importorskip("torch")

    burst = torch.tensor(
        create_burst(create_sharp_image()),
        dtype=torch.float32
    )

    result = select_reference_frame(burst)

    assert isinstance(result, int)
    assert result == 0


def load_real_burst(burst_path):
    """Load the 14 RAW frames in their original capture order."""

    import torch
    from datasets.burstsr_dataset import SamsungRAWImage

    frames = []

    for i in range(14):
        frame = SamsungRAWImage.load(
            str(burst_path / f"samsung_{i:02d}")
        )

        image = frame.get_image_data(
            normalize=True,
            substract_black_level=True,
            white_balance=False
        )

        frames.append(image)

    return torch.stack(frames).float()


def test_real_bursts():
    """Evaluate reference selection on two real BurstSR bursts."""

    from pathlib import Path
    import matplotlib.pyplot as plt
    from reference_selection.algorithm1 import compute_frame_score

    dataset_path = Path(__file__).resolve().parents[2] / "train_00"

    if not dataset_path.exists():
        pytest.skip("BurstSR dataset not available.")

    burst_paths = sorted(
        path for path in dataset_path.iterdir()
        if path.is_dir() and (path / "samsung_00").is_dir()
    )

    assert len(burst_paths) >= 2

    output_dir = Path(__file__).resolve().parents[1] / "results" / "algorithm1"
    output_dir.mkdir(parents=True, exist_ok=True)

    for burst_path in burst_paths[:2]:
        burst = load_real_burst(burst_path)
        images = burst.numpy().astype(np.float64)

        scores = [compute_frame_score(frame) for frame in images]
        selected = select_reference_frame(burst)

        assert isinstance(selected, int)
        assert 0 <= selected < 14
        assert np.isfinite(scores).all()

        fig, axes = plt.subplots(2, 7, figsize=(16, 6))

        previews = images[:, [1, 2]].mean(axis=1)
        vmin, vmax = np.percentile(previews, [1, 99])

        for i, ax in enumerate(axes.flat):
            ax.imshow(previews[i], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"Frame {i}\nQ = {scores[i]:.4f}",
                color="green" if i == selected else "black"
            )
            ax.axis("off")

        fig.suptitle(
            f"{burst_path.name} — Selected reference: Frame {selected}"
        )

        plt.tight_layout()
        fig.savefig(output_dir / f"{burst_path.name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
