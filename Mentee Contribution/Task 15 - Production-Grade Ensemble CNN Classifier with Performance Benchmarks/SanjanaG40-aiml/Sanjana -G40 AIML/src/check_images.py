from pathlib import Path
import sys

import numpy as np

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_metadata,
    extract_image_bytes,
)


def is_valid_image(image_bytes):
    """
    Check common image signatures.
    """

    if not image_bytes:
        return False

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return True       # JPEG

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return True       # PNG

    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return True       # GIF

    if image_bytes.startswith(b"BM"):
        return True       # BMP

    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return True       # WEBP

    return False


def main():

    print("=" * 60)
    print("WEATHERNET-05 IMAGE VALIDATION")
    print("=" * 60)

    dataframe = load_metadata()

    invalid_indices = []

    for index, image_object in enumerate(
        dataframe["image"]
    ):

        try:

            image_bytes = extract_image_bytes(
                image_object
            )

            if not is_valid_image(image_bytes):
                invalid_indices.append(index)

        except Exception:
            invalid_indices.append(index)

        if (index + 1) % 1000 == 0:
            print(
                f"Checked {index + 1}/{len(dataframe)}"
            )

    print("\n" + "-" * 60)

    print(
        f"Total images   : {len(dataframe)}"
    )

    print(
        f"Valid images   : "
        f"{len(dataframe) - len(invalid_indices)}"
    )

    print(
        f"Invalid images : "
        f"{len(invalid_indices)}"
    )

    if invalid_indices:

        print("\nFirst invalid indices:")

        print(
            invalid_indices[:30]
        )

        print("\nCorresponding labels:")

        print(
            dataframe.iloc[
                invalid_indices[:30]
            ]["label"].to_list()
        )

    else:

        print("\nAll images passed signature validation! ✅")


if __name__ == "__main__":
    main()