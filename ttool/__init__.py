from typer import Typer
from .tag_prompt import app as tag_prompt_app
from .novelai_enum import app as novelai_app

app = Typer()
app.add_typer(tag_prompt_app, name="prompt2tag")
app.add_typer(novelai_app, name="nai")
