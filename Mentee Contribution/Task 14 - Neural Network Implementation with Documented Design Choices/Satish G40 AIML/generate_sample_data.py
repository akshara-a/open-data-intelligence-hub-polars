"""Create a small synthetic dataset for testing the notebook pipeline."""
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parent / "dataset"
IMAGE_SIZE = (224, 224)
SAMPLES_PER_CLASS = 20

def write_png(path, background, foreground, defective):
    width = height = 32
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            inside = 7 <= x < 25 and 7 <= y < 25
            if defective:
                color = foreground if inside else background
                if inside and (x == y or x + y == 31):
                    color = (0, 0, 0)
            else:
                color = foreground if (x - 16) ** 2 + (y - 16) ** 2 < 64 else background
            row.extend(color)
        rows.append(row)
    raw = b"".join(rows)

    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


for class_name in ("ok_front", "def_front"):
    class_dir = ROOT / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(SAMPLES_PER_CLASS):
        write_png(
            class_dir / f"dummy_{index:02d}.png",
            (176, 196, 222) if class_name == "ok_front" else (255, 228, 225),
            (34, 139, 34) if class_name == "ok_front" else (220, 20, 60),
            class_name == "def_front",
        )

print(f"Created {SAMPLES_PER_CLASS} dummy images in each class under {ROOT}.")
print("These images are for pipeline testing only, not model-quality evaluation.")
