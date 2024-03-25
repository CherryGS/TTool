from typer import Typer
from .tag_prompt import app as tag_prompt_app

app = Typer()
app.add_typer(tag_prompt_app, name="prompt2tag")
