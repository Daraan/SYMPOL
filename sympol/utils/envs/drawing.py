from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


def draw_env(env: gym.Env | gym.vector.VectorEnv, step_counter: int, running_reward: float) -> np.ndarray | None:
    render = env.render()
    if render is None:
        try:
            # in case spec is None
            logger.warning("Env.render of: %s did not output an image", env.spec.id)  # type: ignore[attr-defined]
        except AttributeError:
            logger.warning("Env.render of %s not output an image", env)
        return None
    frame = np.array(render).squeeze()  # might be 4D if from VectorEnv
    try:
        image = Image.fromarray(frame)
    except Exception:
        logger.exception("Error converting frame to image: %s")
        return None
    draw = ImageDraw.Draw(image)
    text_step = f"Step: {step_counter}"
    font_size = frame.shape[0] // 20
    draw.text(
        (font_size, font_size * 0.5),
        text_step,
        (200, 200, 200),
        font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size),
    )
    text_reward = f"Reward: {running_reward}"
    draw.text(
        (font_size, font_size * 2.0),
        text_reward,
        (200, 200, 200),
        font=ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size),
    )
    return np.array(image)
