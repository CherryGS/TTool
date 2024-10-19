"""
读取图片中 metadata 的元数据并将其中的 Prompt 解析并写入 tag list
目前来说
* 支持 NAI
* 使用 digikam 的读取形式
* 会将权重去掉，但原始 prompt 还在metadata里
"""

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from itertools import filterfalse
import os
from pathlib import Path
from typing import Annotated

import pyexiv2
from PIL import Image
from rich import print
from rich.progress import track
from typer import Abort, Option, Typer, confirm

app = Typer()


@dataclass
class NAIPrompt:
    prompt: str
    steps: int
    scale: float
    uncond_scale: float
    cfg_rescale: float
    seed: int
    noise_schedule: str
    sampler: str

    @classmethod
    def from_dict(cls, **kw):
        return cls(
            **{
                k: v
                for k, v in kw.items()
                if k in list(map(lambda x: x.name, dataclasses.fields(cls)))
            }
        )


def extract_data(f: Path):
    with Image.open(f, "r") as i:
        d = NAIPrompt.from_dict(**json.loads(i.info["Comment"]))
    return d


@app.command(help="提取图片信息并保存成 json")
def extract(f: Path):
    with open(f"{f.stem}.json", "w") as e:
        e.write(json.dumps(asdict(extract_data(f))))


@app.command(help="转变 NAI/SD 生成图片中的元数据以便于 digikam 识别")
def transfer(path: Path, debug: Annotated[int, Option()] = 4):
    pyexiv2.set_log_level(debug)
    if path.is_file():
        ff = [path]
    else:
        if not confirm("输入路径可能为文件夹，是否确认?", abort=False):
            raise Abort()
        ff = list(filterfalse(lambda x: False if x.is_file() else True, path.iterdir()))
    for f in track(ff):
        if f.suffix != ".png":
            continue
        try:
            sta = f.stat()
            d = extract_data(f)
            with open(f, "rb+") as ff:
                with pyexiv2.ImageData(ff.read()) as i:
                    if "Xmp.digiKam.TagsList" in i.read_xmp():
                        continue
                    p = list(
                        map(
                            lambda x: f"NAI/{x.replace('{', '').replace('}', '').replace('[', '').replace(']', '').strip()}",
                            filterfalse(
                                lambda x: False if x else True,
                                d.prompt.split(","),
                            ),
                        )
                    )
                    lis = [
                        *p,
                        f"AI/sampler:{d.sampler}",
                        f"AI/step:{d.steps}",
                        f"AI/scale:{d.scale}",
                        f"AI/noise_schedule:{d.noise_schedule}",
                    ]
                    i.modify_xmp({"Xmp.digiKam.TagsList": lis})
                    ff.seek(0)
                    ff.truncate()
                    ff.write(i.get_bytes())
            os.utime(f, (sta.st_atime, sta.st_mtime))
        except Exception as e:
            raise e
