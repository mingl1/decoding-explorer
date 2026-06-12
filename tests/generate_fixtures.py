#!/usr/bin/env python3
"""
Script to generate sample TIFF files for testing.

This script creates synthetic microscopy-like TIFF files
that can be used for testing the Decoding Explorer application.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def generate_tiff(
    output_path: str,
    shape: tuple = (1, 512, 512),
    dtype=np.uint16,
    seed: int = None,
    add_noise: bool = True,
    add_pattern: bool = False,
):
    """
    Generate a synthetic TIFF image for testing.

    Args:
        output_path: Path to save the TIFF file
        shape: Shape of the image (channels, height, width)
        dtype: Data type of the image
        seed: Random seed for reproducibility
        add_noise: Whether to add random noise
        add_pattern: Whether to add a synthetic pattern
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base image
    if add_pattern:
        # Create a pattern that looks somewhat like microscopy data
        h, w = shape[-2], shape[-1]
        x = np.linspace(-2 * np.pi, 2 * np.pi, w)
        y = np.linspace(-2 * np.pi, 2 * np.pi, h)
        xx, yy = np.meshgrid(x, y)

        # Create a circular pattern
        r = np.sqrt(xx**2 + yy**2)
        pattern = np.sin(r) * 1000 + 2000

        # Create multi-channel image
        arr = np.zeros(shape, dtype=dtype)
        for c in range(shape[0]):
            channel_pattern = pattern * (1 + 0.3 * np.sin(c * np.pi / shape[0]))
            noise = (
                np.random.randint(-100, 100, (h, w), dtype=dtype) if add_noise else 0
            )
            arr[c] = np.clip(channel_pattern + noise, 0, 65535)
    else:
        # Simple random image
        arr = np.random.randint(100, 60000, shape, dtype=dtype)

    # Save the TIFF with metadata
    metadata = {
        "axes": "CYX",
        "PhysicalSizeX": 0.1,
        "PhysicalSizeY": 0.1,
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeYUnit": "um",
    }

    tifffile.imwrite(output_path, arr, metadata=metadata)
    print(f"Generated: {output_path} (shape: {shape}, dtype: {dtype})")


def generate_test_suite(
    output_dir: str,
    num_files: int = 5,
    seed: int = 42,
):
    """
    Generate a complete test suite of TIFF files.

    Args:
        output_dir: Directory to save files
        num_files: Number of files to generate
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_files} test TIFF files in {output_dir}")

    for i in range(num_files):
        # Vary the parameters slightly
        shape = (1, 512, 512)
        file_seed = seed + i
        add_pattern = i % 2 == 0  # Alternate between patterns

        filename = f"test_sample_{i:02d}.tif"
        filepath = output_path / filename

        generate_tiff(
            str(filepath),
            shape=shape,
            dtype=np.uint16,
            seed=file_seed,
            add_noise=True,
            add_pattern=add_pattern,
        )

    # Generate a reference file
    generate_tiff(
        str(output_path / "reference.tif"),
        shape=(3, 512, 512),  # Multi-channel
        dtype=np.uint16,
        seed=seed + 100,
        add_noise=True,
        add_pattern=True,
    )

    # Generate aligned version (slightly shifted)
    np.random.seed(seed + 200)
    h, w = 512, 512
    arr = np.random.randint(100, 60000, (1, h, w), dtype=np.uint16)
    # Add a slight shift pattern
    for y in range(h):
        for x in range(w):
            if 200 < x < 300 and 200 < y < 300:
                arr[0, y, x] = min(65535, arr[0, y, x] + 5000)

    tifffile.imwrite(
        str(output_path / "aligned_sample_00.tif"),
        arr,
        metadata={"axes": "CYX"},
    )

    print(f"\nTest suite generated in: {output_dir}")
    print(f"Total files: {num_files + 2}")


def generate_bead_test_data(
    output_dir: str,
    num_beads: int = 100,
    image_size: int = 512,
):
    """
    Generate test data simulating bead detection results.

    Args:
        output_dir: Directory to save data
        num_beads: Number of bead positions to generate
        image_size: Size of the image
    """
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate random bead positions
    np.random.seed(42)
    x = np.random.randint(50, image_size - 50, num_beads)
    y = np.random.randint(50, image_size - 50, num_beads)

    # Generate intensity values for each cycle
    cycles = ["cy0", "cy1", "cy2", "cy3"]
    data = {
        "x": x,
        "y": y,
    }
    for cy in cycles:
        data[cy] = np.random.randint(100, 65535, num_beads)

    # Add some invalid beads (all 255)
    for i in range(10):
        for cy in cycles:
            data[cy][i] = 255

    df = pd.DataFrame(data)
    df.to_csv(output_path / "test_beads.csv", index=False)
    print(f"Generated bead test data: {output_path / 'test_beads.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample TIFF files for testing"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tests/fixtures/tiffs",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--num-files", "-n", type=int, default=5, help="Number of files to generate"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--beads", action="store_true", help="Also generate bead test data"
    )

    args = parser.parse_args()

    # Generate test suite
    generate_test_suite(
        output_dir=args.output,
        num_files=args.num_files,
        seed=args.seed,
    )

    # Optionally generate bead data
    if args.beads:
        generate_bead_test_data(
            output_dir=args.output,
            num_beads=100,
        )


if __name__ == "__main__":
    main()
