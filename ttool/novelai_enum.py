import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import randint, random, sample
from typing import Optional

from anyutils.logger import get_console_logger
from novelai_api import NovelAIAPI
from novelai_api.ImagePreset import ImageModel, ImagePreset, ImageSampler, UCPreset
from pydantic import BaseModel, Field
from rich import print
from rich.traceback import install
from typer import Typer

install(show_locals=True)


@dataclass
class Config:
    token: str


opt_size = ((1216, 832), (832, 1216), (1024, 1024))

logger = get_console_logger("novelai_enum", True)


async def gen(prompt: str, scale: float = 10, rescale: float = 0.2):

    path = Path(os.getcwd()) / "novelai.secret.json"
    with open(path, "r") as f:
        config = Config(**json.loads(f.read()))

    client = NovelAIAPI().high_level
    await client.login_with_token(config.token)

    setting = ImagePreset.from_v3_config()
    setting.uc = """
                bad anatomy, bad hands, @_@, mismatched pupils, glowing eyes, female pubic hair, futanari, censored, long body, bad feet, condom, sketch, greyscale
                """
    setting.uc_preset = UCPreset.Preset_Heavy
    setting.resolution = opt_size[0]
    setting.steps = 28
    setting.scale = scale
    setting.uncond_scale = 1
    setting.cfg_rescale = rescale
    setting.smea = True
    setting.smea_dyn = True
    setting.sampler = ImageSampler.k_euler_ancestral
    setting.seed = randint(1, 999999999)

    img = list()
    async for i in client.generate_image(
        prompt,
        ImageModel.Anime_v3,
        setting,
    ):
        img.append(i)

    return img[0][1]


def read_as_list(path: Path):
    with open(path, "r") as f:
        return list(
            filter(
                lambda x: x,
                [(i.strip()) for i in f],
            )
        )


def format_as_set(src: Path, tar: Path | None = None, sep: str = ","):
    s: set[str] = set()
    if tar is None:
        tar = src

    with open(src, "r") as f:
        for i in f:
            l = set(filter(lambda x: x, map(lambda x: x.strip(), i.split(sep))))
            s |= l

    with open(tar, "w") as f:
        for i in sorted(s):
            f.write(i + "\n")


class Action(BaseModel):
    id: int
    nsfw: bool
    src: str
    description: str


class NaiThing(BaseModel):
    action: list[Action]


def get_schema(path: Path = Path("schema.json")):
    import json

    with open(path, "w") as f:
        f.write(json.dumps(NaiThing.model_json_schema()))


class Info(BaseModel):
    base: Path
    path: Path


class InfoCollection(BaseModel):
    info: dict[str, Info]


class FunctionType(str, Enum):
    choice = "choice"
    combine = "combine"
    upordown = "up|down"


class Function(BaseModel):
    name: str
    type: FunctionType


class Choice(BaseModel):
    from_info: str
    min_max: tuple[int, int]
    lines: tuple[int, int]


class Combine(Function):
    type: FunctionType = FunctionType.combine
    input: list[str] | None = Field(None)


class UpOrDown(Function):
    type: FunctionType = FunctionType.upordown
    adj_prob: float = Field(0, ge=0, le=1)
    up_prob: float = Field(0, ge=0, le=1)
    down_prob: float = Field(0, ge=0, le=1)


class Process(BaseModel):
    process: list[Choice | Combine | UpOrDown]
    info_path: Path | None = Field(None)


def choice(c: Choice, info: dict[str, Info]):
    a = info[c.from_info]


app = Typer()


def upordown(t: str, p: float = 0.3, q: float = 0.6):
    c = 0
    while random() < p:
        c = c + 1
    if random() <= q:
        t = "".join(["{" * c, t, "}" * c])
    else:
        t = "".join(["[" * c, t, "]" * c])
    return t


save_path = Path(r"C:\Users\TickT\Downloads\Misc")

base_path = Path(r"E:\AI\NAI Source")
artist_path = base_path / "artist.txt"
scene_path = base_path / "scene.txt"
style_path = base_path / "style.txt"
action_path = base_path / "action.txt"


class LoopConfig(BaseModel):
    random_stype: bool = True
    random_scene: bool = True
    random_artist: bool = True

    action_range: tuple[int, int] = (0, -1)
    style_range: tuple[int, int] = (0, -1)
    scene_range: tuple[int, int] = (0, -1)
    artist_range: tuple[int, int] = (0, -1)


@app.command()
def loop(style_range: tuple[int, int] = (0, -1)):

    action_list = read_as_list(action_path)
    style_list = read_as_list(style_path)

    for style_idx, style in enumerate(style_list[style_range[0] : style_range[1]]):
        style_idx += 1
        for action_idx, action in enumerate(action_list):
            action_idx += 1
            for scale, rescale in [(6, 0), (8, 0.1), (10, 0.2)]:

                try:
                    logger.info(
                        f"--- style({style_idx}) - action({action_idx}) - scale({scale}) ---"
                    )

                    img = asyncio.run(gen(",".join([style, action]), scale, rescale))
                    path = (
                        save_path
                        / f"{int(time.time())}_style({style_idx})_action({action_idx})_scale({scale}).png"
                    )

                    with open(path, "wb") as f:
                        f.write(img)

                    logger.info(f"{path.name}")

                except Exception as e:
                    logger.error(e)
                    time.sleep(5)


@app.command()
def art(repeat: int = 10000, action_target: int = 0, style_target: int = 0):

    def get_style():
        mn = 1
        mx = 15
        cnt = sample(range(mn, mx + 1), 1, counts=reversed(range(mn, mx + 1)))
        res = sample(artist_list, cnt[0])
        res = [upordown(i) for i in sorted(res)]
        logger.info(f"Artist count: {cnt[0]}")

        prompt = ", ".join(res)

        if random() < 0.6:
            res = sample(scene_list, randint(0, 5))  # 风格个数
            res = [upordown(i) for i in sorted(res)]

            prompt = ", ".join([prompt, ", ".join(res)])
        return prompt

    scene_list = read_as_list(scene_path)
    artist_list = read_as_list(artist_path)
    action_list = read_as_list(action_path)
    style_list = read_as_list(style_path)
    errCnt = 0

    for i in range(repeat):
        time.sleep(1)
        logger.info(f"------- Running {i+1} loop -------")

        # action
        try:
            action_list_idx = randint(1, len(action_list))
            if action_target == 0:
                action_part = action_list[action_list_idx - 1]
            else:
                action_list_idx = action_target
                action_part = action_list[action_target - 1]
        except Exception as e:
            logger.warning("没有符合条件的 action 项, 随机选择")
            action_part = action_list[action_list_idx - 1]

        # style
        try:
            style_list_idx = randint(1, len(style_list))
            if style_target == 0:
                style_part = get_style()
                style_list_idx = 0
            else:
                style_list_idx = style_target
                style_part = style_list[style_target - 1]

        except Exception as e:
            logger.warning("没有符合条件的 style 项, 随机选择")
            style_part = style_list[style_list_idx - 1]

        # main
        try:
            prompt = ""

            scale = round(random() * 4 + 6, 1)
            rescale = round(0.2 * max(0, (scale - 7)) / 3, 2)

            logger.info(f"Scale & rescale: {(scale, rescale)}")

            logger.info(f"Style: '{style_part}'")
            logger.info(f"Action: '{action_part}'")

            prompt = ", ".join(
                [style_part, action_part] if action_part else [style_part]
            )

            img = asyncio.run(gen(prompt, scale, rescale))

            path = (
                save_path
                / f"{int(time.time())}_style({style_list_idx})_action({action_list_idx}).png"
            )
            with open(path, "wb") as f:
                f.write(img)

            logger.info(f"Image: '{path.name}'")
            errCnt = 0

        except Exception as e:
            logger.error(e)
            time.sleep(5)
            errCnt += 1

        finally:
            logger.debug(f"Prompt: '{prompt}'")
            if errCnt == 10:
                logger.error("Too many tries...")
                break


if __name__ == "__main__":
    """"""
