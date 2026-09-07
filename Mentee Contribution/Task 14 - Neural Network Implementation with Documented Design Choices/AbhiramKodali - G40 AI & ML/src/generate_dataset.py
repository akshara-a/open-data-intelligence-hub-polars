from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SIZE = (224, 224)
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def create_ok_image(rng):
    """Create a synthetic defect-free casting image."""
    image = Image.new("RGB", IMAGE_SIZE, (205, 205, 205))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (25, 35, 199, 189),
        radius=18,
        fill=(145, 145, 145),
        outline=(105, 105, 105),
        width=3,
    )

    # Subtle, harmless surface texture.
    for _ in range(30):
        x = int(rng.integers(40, 185))
        y = int(rng.integers(50, 175))
        radius = int(rng.integers(1, 3))
        shade = int(rng.integers(140, 160))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(shade, shade, shade),
        )

    return image


def create_defect_image(rng):
    """Create a casting image with a clear synthetic surface defect."""
    image = create_ok_image(rng)
    draw = ImageDraw.Draw(image)

    defect_type = rng.choice(["crack", "large_spot", "scratch"])

    if defect_type == "crack":
        # Prominent branching crack.
        start_x = int(rng.integers(65, 125))
        start_y = int(rng.integers(65, 105))

        points = [(start_x, start_y)]

        for _ in range(6):
            start_x += int(rng.integers(-20, 21))
            start_y += int(rng.integers(10, 18))
            points.append((start_x, start_y))

        draw.line(points, fill=(35, 35, 35), width=6)

        # Small branch.
        branch_start = points[3]
        draw.line(
            [
                branch_start,
                (
                    branch_start[0] + 30,
                    branch_start[1] - 25,
                ),
            ],
            fill=(35, 35, 35),
            width=5,
        )

    elif defect_type == "large_spot":
        # Large dark casting void.
        x = int(rng.integers(75, 150))
        y = int(rng.integers(75, 145))
        radius = int(rng.integers(16, 25))

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(40, 40, 40),
        )

        # Smaller surrounding pores.
        for _ in range(4):
            offset_x = int(rng.integers(-35, 36))
            offset_y = int(rng.integers(-35, 36))
            small_radius = int(rng.integers(4, 9))

            draw.ellipse(
                (
                    x + offset_x - small_radius,
                    y + offset_y - small_radius,
                    x + offset_x + small_radius,
                    y + offset_y + small_radius,
                ),
                fill=(70, 70, 70),
            )

    else:
        # Long, prominent surface scratch.
        x1 = int(rng.integers(45, 80))
        y1 = int(rng.integers(70, 150))
        x2 = int(rng.integers(145, 180))
        y2 = y1 + int(rng.integers(-15, 16))

        draw.line(
            (x1, y1, x2, y2),
            fill=(30, 30, 30),
            width=7,
        )

    return image


def clear_png_files():
    """Remove previously generated PNG images before regeneration."""
    if not DATA_DIR.exists():
        return

    for image_path in DATA_DIR.rglob("*.png"):
        image_path.unlink()


def generate_split(split, count_per_class, rng):
    """Generate one dataset split."""
    for class_name in ("ok_front", "def_front"):
        output_dir = DATA_DIR / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count_per_class):
            if class_name == "ok_front":
                image = create_ok_image(rng)
            else:
                image = create_defect_image(rng)

            image.save(output_dir / f"{class_name}_{index:04d}.png")


def main():
    rng = np.random.default_rng(SEED)

    clear_png_files()

    generate_split("train", 80, rng)
    generate_split("val", 20, rng)
    generate_split("test", 20, rng)

    unseen_dir = DATA_DIR / "unseen"
    unseen_dir.mkdir(parents=True, exist_ok=True)

    for index in range(5):
        if index % 2 == 0:
            image = create_ok_image(rng)
        else:
            image = create_defect_image(rng)

        image.save(unseen_dir / f"unseen_{index + 1}.png")

    print(f"Dataset generated at: {DATA_DIR}")
    print("Classes: ok_front, def_front")
    print("Training images: 160")
    print("Validation images: 40")
    print("Test images: 40")
    print("Unseen images: 5")


if __name__ == "__main__":
    main()