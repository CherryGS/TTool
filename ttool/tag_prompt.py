"""
读取图片中 metadata 的元数据并将其中的 Prompt 解析并写入 tag list
目前来说
* 支持 NAI
* 使用 digikam 的读取形式
* 会将权重去掉，但原始 prompt 还在metadata里
"""

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
    height: int
    width: int
    scale: float
    uncond_scale: float
    cfg_rescale: float
    seed: int
    n_samples: int
    hide_debug_overlay: bool
    noise_schedule: str
    sampler: str
    controlnet_strength: float
    controlnet_model: None
    dynamic_thresholding: bool
    dynamic_thresholding_percentile: float
    dynamic_thresholding_mimic_scale: float
    sm: bool
    sm_dyn: bool
    skip_cfg_below_sigma: float
    lora_unet_weights: None
    lora_clip_weights: None
    uc: str
    request_type: str
    reference_information_extracted_multiple: list[float] = field(default_factory=list)
    reference_strength_multiple: list[float] = field(default_factory=list)
    reference_information_extracted: float = 0
    reference_strength: float = 0
    add_original_image: bool = False
    signed_hash: str = ""
    legacy_v3_extend: bool = False
    strength: int = 0
    noise: int = 0
    extra_noise_seed: int = 0
    legacy: int = 0


def extract_data(f: Path):
    with Image.open(f, "r") as i:
        d = NAIPrompt(**json.loads(i.info["Comment"]))
    return d


@app.command(help="将图片的信息提取并保存成 json")
def extract(f: Path):
    with open(f"{f.stem}.json", "w") as e:
        e.write(json.dumps(asdict(extract_data(f))))


@app.command(
    help="将 NAI/SD 生成图片中的 Prompt 和一些额外信息转成 tag list 存储在图片中便于 digikam 识别"
)
def run(path: Path, debug: Annotated[int, Option()] = 4):
    pyexiv2.set_log_level(debug)
    pyexiv2.enableBMFF()
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
