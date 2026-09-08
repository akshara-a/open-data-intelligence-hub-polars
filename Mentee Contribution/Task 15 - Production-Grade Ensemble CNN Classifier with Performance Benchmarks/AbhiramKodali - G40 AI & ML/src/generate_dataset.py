from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

IMAGE_SIZE = 224

SPLITS = {
    "train": 120,
    "val": 30,
    "test": 30,
}

CLASSES = ["ok_front", "def_front"]


def clear_png_files():
    """Remove previously generated PNG files."""
    if not DATA_DIR.exists():
        return

    for png_file in DATA_DIR.rglob("*.png"):
        png_file.unlink()


def create_base_image(rng):
    """Create a synthetic casting surface."""
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (190, 190, 190),
    )

    draw = ImageDraw.Draw(image)

    # Rounded casting body
    draw.rounded_rectangle(
        (20, 20, 204, 204),
        radius=22,
        fill=(175, 175, 175),
        outline=(115, 115, 115),
        width=4,
    )

    # Subtle surface texture
    for _ in range(180):
        x = rng.randint(28, 196)
        y = rng.randint(28, 196)
        shade = rng.randint(145, 205)
        radius = rng.randint(1, 2)

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(shade, shade, shade),
        )

    return image


def create_ok_image(seed):
    """Create an acceptable casting image."""
    rng = random.Random(seed)
    image = create_base_image(rng)

    # Add small harmless surface variation
    draw = ImageDraw.Draw(image)

    for _ in range(4):
        x1 = rng.randint(50, 170)
        y1 = rng.randint(50, 170)
        x2 = x1 + rng.randint(3, 10)
        y2 = y1 + rng.randint(3, 10)

        draw.ellipse(
            (x1, y1, x2, y2),
            fill=(165, 165, 165),
        )

    return image


def create_defect_image(seed):
    """Create a defective casting image with obvious defect patterns."""
    rng = random.Random(seed)
    image = create_base_image(rng)
    draw = ImageDraw.Draw(image)

    defect_type = seed % 3

    if defect_type == 0:
        # Crack
        points = [
            (rng.randint(50, 80), rng.randint(50, 80)),
            (rng.randint(80, 120), rng.randint(80, 110)),
            (rng.randint(110, 145), rng.randint(100, 140)),
            (rng.randint(140, 180), rng.randint(130, 180)),
        ]

        draw.line(
            points,
            fill=(45, 45, 45),
            width=6,
        )

    elif defect_type == 1:
        # Large dark defect spot
        x = rng.randint(75, 145)
        y = rng.randint(75, 145)
        radius = rng.randint(18, 28)

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(50, 50, 50),
        )

    else:
        # Long scratch
        x1 = rng.randint(45, 70)
        y1 = rng.randint(150, 180)
        x2 = rng.randint(150, 180)
        y2 = rng.randint(45, 75)

        draw.line(
            (x1, y1, x2, y2),
            fill=(35, 35, 35),
            width=7,
        )

    return image


def save_image(image, path):
    """Apply light image variation and save."""
    image = image.filter(ImageFilter.GaussianBlur(radius=0.25))
    image.save(path, format="PNG")


def generate_split(split_name, count_per_class, seed_offset):
    for class_name in CLASSES:
        class_dir = DATA_DIR / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count_per_class):
            seed = seed_offset + index

            if class_name == "ok_front":
                image = create_ok_image(seed)
            else:
                image = create_defect_image(seed)

            filename = f"{class_name}_{index + 1:03d}.png"
            save_image(image, class_dir / filename)


def generate_unseen():
    unseen_dir = DATA_DIR / "unseen"
    unseen_dir.mkdir(parents=True, exist_ok=True)

    for index in range(5):
        # Alternate acceptable and defective examples
        seed = 10000 + index

        if index % 2 == 0:
            image = create_ok_image(seed)
        else:
            image = create_defect_image(seed)

        save_image(
            image,
            unseen_dir / f"unseen_{index + 1}.png",
        )


def main():
    clear_png_files()

    for split_number, (split_name, count) in enumerate(SPLITS.items()):
        generate_split(
            split_name,
            count,
            seed_offset=1000 * (split_number + 1),
        )

    generate_unseen()

    print(f"Dataset generated at: {DATA_DIR}")
    print("Classes:", ", ".join(CLASSES))

    for split_name, count in SPLITS.items():
        print(
            f"{split_name}: "
            f"{count} images per class "
            f"({count * len(CLASSES)} total)"
        )

    print("unseen: 5 images")


if __name__ == "__main__":
    main()