from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.loader import (BaseModelLoader, BitsAndBytesModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    model = loader.load_model(vllm_config=vllm_config)

    import rich
    import rich.pretty
    import rich.terminal_theme
    import rich.color
    p = rich.pretty.Pretty(loader.weight_layout)
    consolex = rich.console.Console(record=True, emoji=True, no_color=False)
    if isinstance(loader, BitsAndBytesModelLoader):
        consolex.print("[pink1 on grey0] START WEIGHT LAYOUT")
        consolex.print(p)
        consolex.print("[grey0 on pink1] END WEIGHT LAYOUT")
    consolex.print("[pink1 on grey0] START MODEL")
    p = rich.pretty.Pretty(model)
    consolex.print(p)
    consolex.print("[grey0 on pink1] END MODEL")

    black_monokai = rich.terminal_theme.MONOKAI
    black_monokai.background_color = rich.color.ColorTriplet(0,0,0)
    consolex.save_svg("/home/jeff/weight_layout.svg", theme=black_monokai, clear=False)
    consolex.save_html("/home/jeff/weight_layout.html", theme=black_monokai, clear=True)

    return model

__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
