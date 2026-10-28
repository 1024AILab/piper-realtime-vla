"""Use Gradio's base64 image payload to avoid a temporary file per preview frame."""
import base64
import io

import gradio as gr
from gradio.data_classes import Base64ImageData
from PIL import Image

from gradio.events import Dependency

class PreviewImage(gr.Image):
    def get_block_name(self):
        return "image"

    def postprocess(self, value):
        if value is None:
            return None
        image = value if isinstance(value, Image.Image) else Image.fromarray(value)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return Base64ImageData(url="data:image/jpeg;base64," + data)

    def api_info_as_output(self):
        return Base64ImageData.model_json_schema()
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component