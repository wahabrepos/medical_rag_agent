"""Encode frames from record-demo.mjs as the animated WebP in the README.

    uv run --with pillow python docs/media/encode_demo.py frames/ docs/media/ui-demo.webp

Identical consecutive frames are merged; each frame is shown for its recorded playback time.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

WIDTH = 800


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    frames = json.loads((src / "frames.json").read_text())
    images: list[Image.Image] = []
    durations: list[int] = []
    for i, frame in enumerate(frames):
        end = frames[i + 1]["at"] if i + 1 < len(frames) else frame["at"] + 3000
        image = Image.open(frame["file"]).convert("RGB")
        image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
        if images and ImageChops.difference(image, images[-1]).getbbox() is None:
            durations[-1] += end - frame["at"]
            continue
        images.append(image)
        durations.append(max(40, end - frame["at"]))
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        quality=75,
        method=4,
    )
    print(f"{len(images)} frames, {sum(durations) / 1000:.1f} s -> {out}")


if __name__ == "__main__":
    main()
