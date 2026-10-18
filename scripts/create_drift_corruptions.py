"""Create reproducible image corruptions for drift detector evaluation."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


SCENARIOS = {
    "brightness_low": ("brightness", 1.15),
    "brightness_medium": ("brightness", 1.40),
    "brightness_high": ("brightness", 1.80),
    "contrast_low": ("contrast", 1.15),
    "contrast_medium": ("contrast", 1.40),
    "contrast_high": ("contrast", 1.80),
    "blur_low": ("blur", 0.7),
    "blur_medium": ("blur", 1.5),
    "blur_high": ("blur", 3.0),
}


def corrupt(image, method, severity):
    if method == "brightness":
        return ImageEnhance.Brightness(image).enhance(severity)
    if method == "contrast":
        return ImageEnhance.Contrast(image).enhance(severity)
    if method == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    raise ValueError(f"Unknown corruption: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/scoring/current_manifest.csv"),
    )
    parser.add_argument("--size", type=int, default=500)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Keep the same first 500 images across all scenarios.
    selected = rows[:args.size]
    if len(selected) != args.size:
        raise ValueError("Not enough images in manifest")

    output_root = Path("data/scoring/corruptions")
    summary = {}

    for scenario, (method, severity) in SCENARIOS.items():
        scenario_dir = output_root / scenario
        image_dir = scenario_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        output_rows = []

        for index, row in enumerate(selected):
            source = Path(row["image_path"])
            target = image_dir / f"{index:04d}.jpg"

            with Image.open(source) as image:
                rgb = image.convert("RGB")
                modified = corrupt(rgb, method, severity)
                modified.save(target, quality=95)

            output_row = dict(row)
            output_row["image_path"] = str(target)
            output_rows.append(output_row)

        manifest_path = scenario_dir / "manifest.csv"

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)

        summary[scenario] = {
            "method": method,
            "severity": severity,
            "images": len(output_rows),
            "manifest": str(manifest_path),
        }

        print(f"{scenario}: {len(output_rows)} images created", flush=True)

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPASS: Corruption datasets created")


if __name__ == "__main__":
    main()
