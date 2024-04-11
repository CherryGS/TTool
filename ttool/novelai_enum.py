import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from random import randint, random, sample
from typing import Annotated, Literal

from anyutils.logger import get_console_logger
from novelai_api import NovelAIAPI, NovelAIError
from novelai_api.ImagePreset import ImageModel, ImagePreset, ImageSampler, UCPreset
from pydantic import BaseModel, Field
from rich.traceback import install
from typer import Argument, Typer

install(show_locals=True)


@dataclass
class NaiConfig:
    token: str


opt_size = ((1216, 832), (832, 1216), (1024, 1024))

logger = get_console_logger("novelai_enum", True)


async def gen(prompt: str, scale: float = 10, rescale: float = 0.2):

    path = Path(os.getcwd()) / "novelai.secret.json"
    with open(path, "r") as f:
        config = NaiConfig(**json.loads(f.read()))

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


app = Typer()


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


class Prob(BaseModel):
    """目前概率上优先级是 part > data > behavior"""

    change_prob: float = Field(-1, description="发生权重改变的概率限度")
    up_prob: float = Field(0, description="权重改变是升级的概率")
    max_change: int = Field(3)


class Part(Prob):
    src: str
    name: str
    description: str | None = None


class Parts(BaseModel):
    src: list[Part]


class PathCollection(BaseModel):
    base: Path = Field(description="可以是绝对路径，也可以是以该文件为中心的相对路径.")
    src: dict[str, Path]


class Behavior(Prob):
    type: Literal["loop", "choice"]
    min: int = Field(description="1 indexed")
    max: int = Field(description="neg value used like python slice")
    cnt: int = Field(1, description="选择的次数")


class SelectSettings(Prob):
    name: str


class Function(BaseModel):
    data_select: list[SelectSettings]
    behavior: dict[str, Behavior]
    repeat: int = Field(1)


class Config(BaseModel):
    data: dict[str, PathCollection]
    function: dict[str, Function]


def parser(path: Path, name: str):
    logger.info("正在解析配置文件...")
    now = os.getcwd()
    os.chdir(path.parent)
    res: dict[str, list[Part]] = dict()

    has_custom_prop = False

    config = Config.model_validate_json(path.read_text())
    for i in config.function[name].data_select:
        paths = config.data[i.name]
        for j in paths.src:
            if not paths.src[j].is_absolute():
                file = paths.base.absolute() / paths.src[j]
            else:
                file = paths.src[j]
            parts = Parts.model_validate_json(file.read_text())
            for k in parts.src:
                if k.change_prob == -1:
                    continue
                has_custom_prop = True
                k.change_prob = i.change_prob
                k.up_prob = i.up_prob
            if j not in res:
                res[j] = list()
            res[j].extend(parts.src)
    if has_custom_prop:
        logger.warning("有自定义概率的节点")
    os.chdir(now)
    logger.info("配置文件解析完成...")
    return (config, res)


def upordown(t: Part):
    c = 0
    while random() < t.change_prob:
        c = c + 1
    c = min(c, t.max_change)
    if random() <= t.up_prob:
        res = "".join(["{" * c, t.src, "}" * c])
    else:
        res = "".join(["[" * c, t.src, "]" * c])
    logger.debug(t)
    logger.debug(c)
    return res


# def get_style(artist_list: list[str], scene_list: list[str]):
#     mn = 2
#     mx = 15
#     cnt = sample(range(mn, mx + 1), 1, counts=reversed(range(mn, mx + 1)))
#     res = sample(artist_list, cnt[0])
#     res = [upordown(i) for i in sorted(res)]
#     logger.info(f"Artist count: {cnt[0]}")

#     prompt = ", ".join(res)

#     if random() < 0.6:
#         res = sample(scene_list, randint(0, 5))  # 风格个数
#         res = [upordown(i) for i in sorted(res)]

#         prompt = ", ".join([prompt, ", ".join(res)])
#     return prompt


@app.command("tojson", help="将列表中的每一行放到 src 中，排序去重去空行")
def parse_to_json(
    input: Annotated[Path, Argument()] = Path("data.txt"),
    output: Annotated[Path, Argument()] = Path("data.json"),
):
    parts = Parts(src=list())
    for j, i in enumerate(sorted(set(read_as_list(input)))):
        parts.src.append(
            Part(src=i, name=f"{j}", change_prob=-1, up_prob=0, max_change=5)
        )
    output.write_text(parts.model_dump_json())


@app.command("combine", help="合并若干个 json ，不去重不排序")
def combine_json(files: list[Path], output: Path = Path("combine.json")):
    s: list[Part] = list()
    for file in files:
        p = Parts.model_validate_json(file.read_text())
        s.extend(p.src)
    output.write_text(Parts(src=s).model_dump_json())


@app.command("schema")
def get_schema(path: Path = Path(".")):
    with open(path / "config.schema.json", "w") as f:
        f.write(json.dumps(Config.model_json_schema()))
    with open(path / "data.schema.json", "w") as f:
        f.write(json.dumps(Parts.model_json_schema()))


@app.command()
def loop(config_path: Path, name: str):

    # def generate(data: Iterable, p: Callable):
    #     idx = 0
    #     for i in data:
    #         idx += 1
    #         yield (idx, i)
    #     while True:
    #         yield (-1, p())

    # def closure():
    #     return get_style(artist_list, scene_list)

    def generate_image(dep: int = 0, prompt: str = "", name: str = ""):
        try:
            if dep == len(order):
                for scale, rescale in ((6, 0), (8, 0.1), (10, 0.2)):
                    path = (
                        save_path
                        / f"{name}_scale({scale})_time({int(time.time())}).png"
                    )

                    time.sleep(2)
                    logger.info(f"正在生成图片到 '{path}'")
                    logger.debug(f"Prompt: '{prompt}'")

                    img = asyncio.run(gen(prompt, scale, rescale))
                    with open(path, "wb") as f:
                        f.write(img)

                    logger.info("生成完成")

            else:
                # 获取当前需要处理的项
                a = order[dep]
                d = data[a[0]]

                # 处理概率
                for i in d:
                    if i.change_prob != -1:
                        continue
                    i.change_prob = a[1].change_prob
                    i.up_prob = a[1].up_prob
                    i.max_change = a[1].max_change

                # 如果是 随机选择
                if a[1].type == "choice":

                    for _ in range(a[1].cnt):
                        res = sample(
                            d,
                            randint(a[1].min, min(a[1].max, len(d))),
                        )
                        if len(res) == 0:
                            continue
                        res_name = res[0].name
                        generate_image(
                            dep + 1,
                            ",".join(
                                filter(
                                    lambda x: x, [prompt, *[upordown(i) for i in res]]
                                )
                            ),
                            f"{name}_{a[0]}({res_name})",
                        )

                # 如果是 循环
                elif a[1].type == "loop":
                    for i in d[a[1].min - 1 : min(a[1].max, len(d))]:
                        generate_image(
                            dep + 1,
                            ",".join(filter(lambda x: x, [prompt, upordown(i)])),
                            f"{name}_{a[0]}({i.name})",
                        )

        except NovelAIError as e:
            logger.error(e)
            time.sleep(60)

        except Exception as e:
            logger.error(e, stack_info=True, exc_info=True)
            time.sleep(60)

    config_path = config_path.absolute()
    a, data = parser(config_path, name)

    # 所有项的遍历顺序及相关行为 (所需项的名字, 行为)
    order = list(a.function[name].behavior.items())

    logger.info(f"本次共有遍历项 {len(order)} 个")
    logger.info(f"本次循环次数为 {a.function[name].repeat} 次")
    logger.info("开始循环")

    for i in range(a.function[name].repeat):
        logger.info(f"第 {i+1} 次循环")
        generate_image()

    logger.info("结束循环")

    exit(0)
    scene_list = read_as_list(scene_path)
    artist_list = read_as_list(artist_path)
    style_list = read_as_list(style_path)

    data = NaiThing.model_validate_json(action_path.read_text())
    action_list = data.action
    style_list = data.style

    errCnt = 0

    for style_idx, style in generate(
        style_list[style_range[0] - 1 : style_range[1]], closure
    ):
        for action in action_list:
            for scale, rescale in [(6, 0), (8, 0.1), (10, 0.2)]:

                try:
                    logger.info(
                        f"--- style({style_idx}) - action({action.name}) - scale({scale}) ---"
                    )

                    prompt = ",".join([style, action.src])
                    img = asyncio.run(gen(prompt, scale, rescale))

                    config_path = (
                        save_path
                        / f"style({style_idx})_action({action.name})_scale({scale})_time({int(time.time())}).png"
                    )

                    with open(config_path, "wb") as f:
                        f.write(img)

                    logger.info(f"{config_path.name}")

                except Exception as e:
                    errCnt += 1
                    logger.error(e)
                    time.sleep(5)

                finally:
                    logger.debug(f"Prompt: '{prompt}'")
                    if errCnt == 5:
                        logger.error("Too many tries...")
                        time.sleep(randint(600, 1200))
                        errCnt = 0


# @app.command()
# def art(repeat: int = 10000, action_target: int = 0, style_target: int = 0):

#     def get_style():
#         mn = 1
#         mx = 15
#         cnt = sample(range(mn, mx + 1), 1, counts=reversed(range(mn, mx + 1)))
#         res = sample(artist_list, cnt[0])
#         res = [upordown(i) for i in sorted(res)]
#         logger.info(f"Artist count: {cnt[0]}")

#         prompt = ", ".join(res)

#         if random() < 0.6:
#             res = sample(scene_list, randint(0, 5))  # 风格个数
#             res = [upordown(i) for i in sorted(res)]

#             prompt = ", ".join([prompt, ", ".join(res)])
#         return prompt

#     scene_list = read_as_list(scene_path)
#     artist_list = read_as_list(artist_path)
#     action_list = read_as_list(action_path)
#     style_list = read_as_list(style_path)
#     errCnt = 0

#     for i in range(repeat):
#         time.sleep(1)
#         logger.info(f"------- Running {i+1} loop -------")

#         # action
#         try:
#             action_list_idx = randint(1, len(action_list))
#             if action_target == 0:
#                 action_part = action_list[action_list_idx - 1]
#             else:
#                 action_list_idx = action_target
#                 action_part = action_list[action_target - 1]
#         except Exception as e:
#             logger.warning("没有符合条件的 action 项, 随机选择")
#             action_part = action_list[action_list_idx - 1]

#         # style
#         try:
#             style_list_idx = randint(1, len(style_list))
#             if style_target == 0:
#                 style_part = get_style()
#                 style_list_idx = 0
#             else:
#                 style_list_idx = style_target
#                 style_part = style_list[style_target - 1]

#         except Exception as e:
#             logger.warning("没有符合条件的 style 项, 随机选择")
#             style_part = style_list[style_list_idx - 1]

#         # main
#         try:
#             prompt = ""

#             scale = round(random() * 4 + 6, 1)
#             rescale = round(0.2 * max(0, (scale - 7)) / 3, 2)

#             logger.info(f"Scale & rescale: {(scale, rescale)}")

#             logger.info(f"Style: '{style_part}'")
#             logger.info(f"Action: '{action_part}'")

#             prompt = ", ".join(
#                 [style_part, action_part] if action_part else [style_part]
#             )

#             img = asyncio.run(gen(prompt, scale, rescale))

#             path = (
#                 save_path
#                 / f"{int(time.time())}_style({style_list_idx})_action({action_list_idx}).png"
#             )
#             with open(path, "wb") as f:
#                 f.write(img)

#             logger.info(f"Image: '{path.name}'")
#             errCnt = 0

#         except Exception as e:
#             logger.error(e)
#             time.sleep(5)
#             errCnt += 1

#         finally:
#             logger.debug(f"Prompt: '{prompt}'")
#             if errCnt == 10:
#                 logger.error("Too many tries...")
#                 break


if __name__ == "__main__":
    """"""
