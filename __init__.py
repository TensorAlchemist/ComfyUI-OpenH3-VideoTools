"""Open-H3 workflow helpers. See THIRD_PARTY_NOTICES.md."""
from comfy_api.latest import ComfyExtension
from .tiled_diffusion import TAOpenH3TiledDiffusion
from .prompt_encode_release import TAOpenH3PromptEncodeRelease
from .refine_release import TAOpenH3RefineRelease
from .vae_decode_low_memory import TAOpenH3VAEDecodeLowMemory
from .auto_input import TAOpenH3AutoInput

class TAOpenH3TiledDiffusionExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [TAOpenH3TiledDiffusion, TAOpenH3PromptEncodeRelease, TAOpenH3RefineRelease, TAOpenH3VAEDecodeLowMemory, TAOpenH3AutoInput]

async def comfy_entrypoint() -> TAOpenH3TiledDiffusionExtension:
    return TAOpenH3TiledDiffusionExtension()
