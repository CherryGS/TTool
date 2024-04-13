import asyncio
from genericpath import isfile
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import randint, random, sample
from typing import Annotated, Any, Self

from aiohttp import ClientConnectorError
from novelai_api import NovelAIAPI, NovelAIError
from novelai_api.ImagePreset import ImageModel, ImagePreset, ImageSampler, UCPreset
from pydantic import BaseModel, ConfigDict, Field
from rich import print
from rich.traceback import install
from typer import Argument, Typer

try:
    from anyutils.logger import get_console_logger

    logger = get_console_logger("novelai_enum", True)
except Exception as e:
    from logging import getLogger

    logger = getLogger("novelai_enum")

install(show_locals=True)


@dataclass
class NaiConfig:
    token: str


opt_size = ((1216, 832), (832, 1216), (1024, 1024))
global_token: str = ""


async def gen(prompt: str, options: dict[str, Any] = {}):
    global global_token
    client = NovelAIAPI().high_level
    await client.login_with_token(global_token)
    setting = ImagePreset.from_v3_config()
    setting.uc = ""
    setting.uc_preset = UCPreset.Preset_Heavy
    setting.resolution = opt_size[0]
    setting.steps = 28
    setting.scale = 6
    setting.uncond_scale = 1
    setting.cfg_rescale = 0
    setting.smea = True
    setting.smea_dyn = True
    setting.decrisper = True
    setting.sampler = ImageSampler.k_euler_ancestral
    setting.seed = randint(1, 999999999)
    setting.update(options)

    img = list()
    async for i in client.generate_image(
        prompt,
        ImageModel.Anime_v3,
        setting,
    ):
        img.append(i)

    return img[0][1]


def read_as_list(path: Path):
    with open(path, "r", encoding="utf8") as f:
        return list(
            filter(
                lambda x: x,
                [(i.strip()) for i in f],
            )
        )


app = Typer()


class BaseWithConfig(BaseModel):
    """特殊的模型, 用来附加全局的设置"""

    model_config = ConfigDict(extra="ignore")
    """忽略额外项"""

    def newer(self, b: BaseModel, c: BaseModel):
        d = b.model_dump(include=self.model_fields_set)
        d.update(c.model_dump(exclude_none=True))
        return self.model_validate(d)


class Prob(BaseWithConfig):
    """
    权重升降级的概率集合

    运算规律是 `c+=1` until `random() >= change_prob` 然后 `c=min(c,max_change)`

    最后如果 `random() <= up_prob` 那么 `c` 就是权重升级次数, 反之
    """

    change_prob: Annotated[
        float | None, Field(description="发生权重改变的概率限度")
    ] = None
    up_prob: Annotated[float | None, Field(description="权重改变是升级的概率")] = None
    max_change: Annotated[int | None, Field(description="权重改变的次数上限")] = None


class UpToDown(Prob):
    """
    特殊的模型
    - 该模型的继承链是连续的
    - 所有附加在该模型上的属性上层覆盖下层
    - 默认 `None` 来表示没有被设置
    """


global_setting = UpToDown(change_prob=0, up_prob=0.5, max_change=3)


class Unit(UpToDown):
    """数据的最小表示单位"""

    name: str | None = Field(None, description="数据的名称, 可以省略, 此时名称为内容")
    src: str = Field(description="数据的内容")
    description: str | None = Field(None, description="数据的描述")
    resolution: tuple[int, int] | None = Field(
        None,
        init=False,
        description="该数据希望对应的图片的大小, 合并时越靠后的数据的值会越先考虑",
    )

    def __hash__(self) -> int:
        return hash(self.src)

    def __lt__(self, o: Self):
        return (
            self.name < o.name
            if self.name is not None and o.name is not None
            else (
                self.name < o.src
                if self.name is not None
                else self.src < o.name if o.name is not None else self.src < o.src
            )
        )


class DataList(UpToDown):
    """数据列"""

    data_list: list[Unit] = Field(description="数据列")
    description: str | None = Field(None, init=False, description="数据列的描述")


class Loop(UpToDown):
    """循环一个数据表中的指定的项"""

    select: str = Field(description="所要操作的数据表的名称")
    loop_list: list[str] | None = Field(
        None,
        description="需要循环的 unit 的名称,如果为空代表所有都要循环,注意此处是无序的",
    )


class Choice(UpToDown):
    """从一个数据表中等概率抽取一些项"""

    select: str = Field(description="所要操作的数据表的名称")
    min: int = Field(1, description="选择数量的下界", init=False)
    max: int = Field(1, description="选择数量的上界,会和序列长度取 min", init=False)
    cnt: int = Field(1, description="该次选择的重复轮数", init=False)


class Select(UpToDown):
    """选择的数据集集"""

    name: str


class Space(UpToDown):
    """项目空间"""

    select: list[Select]

    queue: list[str] = Field(description="执行队列")
    loop: dict[str, Loop] = Field(description="循环执行")
    choice: dict[str, Choice] = Field(description="随机选择")

    repeat: int = Field(1, description="该部分重复次数")
    img_path: Path = Field(description="图片保存路径")
    default_resolution: tuple[int, int] = Field(
        opt_size[0], init=False, description="默认图片大小"
    )
    uc: str = Field(
        "bad anatomy, bad hands, @_@, mismatched pupils, glowing eyes, female pubic hair, futanari, censored, long body, bad feet, condom",
        init=False,
        description="负面提示词",
    )


class PathCollection(BaseWithConfig):
    """指向数据表的路径合集, 构建数据集所需, 所有 `unit` 按照 `src` 去重"""

    base: Path = Field(description="可以是绝对路径，也可以是以该文件为中心的相对路径.")
    data_table: dict[str, list[Path]] = Field(
        description="数据表的类型名和相关数据表的路径"
    )


class Config(BaseWithConfig):
    """总配置"""

    data_set: dict[str, PathCollection] = Field(description="数据集的名称和所需内容")
    space: dict[str, Space]


def parser(path: Path, name: str):
    """
    解析配置并提取数据
    """

    logger.info("正在解析配置文件...")
    now = os.getcwd()
    os.chdir(path.parent)

    # 数据类型和对应数据
    _data_set: dict[str, set[Unit]] = dict()
    config = Config.model_validate_json(path.read_text(encoding="utf8"))
    space = config.space[name]

    s1 = sum([len(i) for i in [space.loop, space.choice]])
    s2 = len(set([*space.loop, *space.choice]))
    if s1 != s2:
        raise ValueError("loop 和 choice 中有重复键值")

    cfg = global_setting.newer(global_setting, space)

    for i in space.select:
        paths = config.data_set[i.name]
        cfgg = cfg.newer(i, cfg)

        for j in paths.data_table:
            for path in paths.data_table[j]:
                if not path.is_absolute():
                    file = paths.base.absolute() / path
                else:
                    file = path

                data = DataList.model_validate_json(file.read_text(encoding="utf8"))
                cfggg = cfgg.newer(data, cfgg)

                if j not in _data_set:
                    _data_set[j] = set()

                for unit in data.data_list:
                    u = unit.newer(unit, cfggg)
                    if u.name is None:
                        u.name = u.src
                    _data_set[j].add(u)

    data_set: dict[str, list[Unit]] = {}
    for i in _data_set:
        data_set[i] = sorted(list(_data_set[i]))

    os.chdir(now)
    logger.info("配置文件解析完成...")
    return (space, data_set)


def upordown(t: Unit, ex: Prob):
    p = t.newer(t, ex)
    assert not p.change_prob is None
    assert not p.max_change is None
    assert not p.up_prob is None

    c = 0
    while random() < p.change_prob:
        c = c + 1
    c = min(c, p.max_change)
    if random() <= p.up_prob:
        res = "".join(["{" * c, t.src, "}" * c])
    else:
        res = "".join(["[" * c, t.src, "]" * c])

    logger.debug(p)
    logger.debug(res)

    return res


@app.command("tojson", help="将列表中的每一行放到 src 中，排序去重去空行")
def parse_to_json(
    input: Annotated[Path, Argument()] = Path("data.txt"),
    output: Annotated[Path, Argument()] = Path("data.json"),
):
    parts = DataList(data_list=list())
    for j, i in enumerate(sorted(set(read_as_list(input)))):
        parts.data_list.append(
            Unit(
                src=i,
                name=f"{j}",
                description=f"{datetime.now()}",
            )
        )
    output.write_text(parts.model_dump_json(exclude_none=False))


@app.command("combine", help="合并若干个 json ，不去重不排序")
def combine_json(files: list[Path], output: Path = Path("combine.json")):
    s: list[Unit] = list()
    for file in files:
        p = DataList.model_validate_json(file.read_text(encoding="utf8"))
        s.extend(p.data_list)
    output.write_text(DataList(data_list=s).model_dump_json(exclude_none=False))


@app.command("schema")
def get_schema(path: Path = Path(".")):
    with open(path / "config.schema.json", "w") as f:
        f.write(json.dumps(Config.model_json_schema()))
    with open(path / "data.schema.json", "w") as f:
        f.write(json.dumps(DataList.model_json_schema()))


def receive_image(path: Path, prompt: str, options: dict, errCnt: int = 0):
    try:
        time.sleep(2)
        logger.info(f"正在生成图片到 '{path}'")
        logger.debug(f"Prompt: '{prompt}'")
        logger.debug(f"options: {options}")

        img = asyncio.run(gen(prompt, options))
        with open(path, "wb") as f:
            f.write(img)
        logger.info("生成完成")

    except (NovelAIError, ClientConnectorError) as e:
        logger.error(e)
        time.sleep(60 * (1 + errCnt))
        logger.warning(f"尝试重新生成第 {errCnt+1} 次")
        receive_image(path, prompt, options, errCnt + 1)


@app.command()
def loop(config_path: Path, name: str, debug: bool = False, token: str = ""):

    def generate_image(dep: int, prompt: str, name: str, resolution: tuple[int, int]):
        try:
            if dep == len(order):
                for scale, rescale in ((6, 0), (8, 0.1), (10, 0.2)):
                    path = (
                        save_path
                        / f"{name}_scale({scale})_time({int(time.time())}).png".replace(
                            ":", "-"
                        )
                    )

                    options = {
                        "scale": scale,
                        "cfg_rescale": rescale,
                        "uc": space.uc,
                        "resolution": resolution,
                    }

                    receive_image(path, prompt, options)
            else:
                # 获取当前需要处理的项
                a = order[dep]
                d = data_set[a[0]]

                # 如果是 随机选择
                match a[1]:
                    case Choice():
                        for _ in range(a[1].cnt):
                            res = sample(
                                d,
                                randint(a[1].min, min(a[1].max, len(d))),
                            )
                            if len(res) == 0:
                                continue
                            res_name = res[0].name
                            for i in res:
                                if i.resolution:
                                    resolution = i.resolution

                            generate_image(
                                dep + 1,
                                ", ".join(
                                    filter(
                                        lambda x: x,
                                        [prompt, *[upordown(i, a[1]) for i in res]],
                                    )
                                ),
                                f"{name}_{a[0]}({res_name})",
                                resolution,
                            )
                    # 如果是 循环
                    case Loop():
                        for i in d:
                            if (
                                a[1].loop_list is not None
                                and i.name not in a[1].loop_list
                            ):
                                continue
                            if i.resolution:
                                resolution = i.resolution
                            generate_image(
                                dep + 1,
                                ", ".join(
                                    filter(lambda x: x, [prompt, upordown(i, a[1])])
                                ),
                                f"{name}_{a[0]}({i.name})",
                                resolution,
                            )
                    case _:
                        raise TypeError("未知的操作类型, 可能是逻辑出现错误")

        except Exception as e:
            logger.error(e, stack_info=True, exc_info=True)
            time.sleep(60)

    global logger, global_token
    if not debug:
        logger = get_console_logger("novelai_enum_no_debug")

    path = Path(os.getcwd()) / "novelai.secret.json"
    if path.is_file():
        with open(path, "r") as f:
            config = NaiConfig(**json.loads(f.read()))
            global_token = config.token
    if token:
        global_token = token

    if not global_token:
        raise ValueError("Novelai api token 未设置")

    config_path = config_path.absolute()
    space, data_set = parser(config_path, name)

    p: dict[str, Loop | Choice] = {}
    p.update(space.loop)
    p.update(space.choice)
    # 所有项的遍历顺序及相关行为 (所需项的名字, 行为)
    order = [(p[i].select, p[i]) for i in space.queue]

    # 图片保存路径
    save_path = space.img_path

    logger.debug(f"Debug 环境")
    time.sleep(2)

    logger.info(f"本次共有遍历项 {len(order)} 个")
    for i in order:
        logger.info(f"其中 '{i[0]}' 项中有 {len(data_set[i[0]])} 个元素")

    logger.info(f"本次循环次数为 {space.repeat} 次")
    logger.info(f"图片保存文件夹为 '{save_path}'")
    logger.debug(data_set)
    time.sleep(3)

    logger.info("开始循环")

    for i in range(space.repeat):
        logger.info(f"第 {i+1} 次循环")
        generate_image(0, "", "", space.default_resolution)

    logger.info("结束循环")


if __name__ == "__main__":
    """"""

    print(Choice(select="1").__pydantic_fields_set__)
