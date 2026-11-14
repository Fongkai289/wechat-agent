"""Skill adapter: package permissions feed the shared platform API."""
from .platform_api import PlatformAPI, COLUMNS  # noqa: F401  (COLUMNS is re-exported)
from .code_skill_packages import manifest


class CodeAPI(PlatformAPI):
    def __init__(self, package, context, *args, **kwargs):
        super().__init__(manifest(package), context, *args, **kwargs)
