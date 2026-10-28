import base64
import io

import numpy as np
from PIL import Image

from piper_studio.preview_image import PreviewImage


def test_live_preview_encodes_in_memory_without_accumulating_temp_files():
    preview = PreviewImage(interactive=False)
    for i in range(3):
        value = preview.postprocess(np.full((480, 640, 3), i * 20, np.uint8))
        assert value.url.startswith("data:image/jpeg;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(value.url.split(",", 1)[1])))
        assert image.size == (640, 480)
    assert len(preview.temp_files) == 0
    assert preview.get_block_name() == "image"
