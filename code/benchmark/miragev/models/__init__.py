from __future__ import annotations

# Import all model adapters so their @register decorators execute
from . import gemini_video  # noqa: F401
from . import hf  # noqa: F401
from . import openai_api  # noqa: F401
from . import gemini_frames  # noqa: F401
