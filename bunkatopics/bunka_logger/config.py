from pydantic import BaseSettings, Field

_FORMAT_WITH_TIME = (
    "<level>{level:<10}</level>|"
    "<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow>|"
    "<magenta>{extra}</magenta>|"
    "<blue>{function}</blue>|"
    "<level>{message}</level>"
)
_FORMAT_WITHOUT_TIME = (
    "<level>{level:<10}</level>|"
    "<magenta>{extra}</magenta>|"
    "<blue>{function}</blue>|"
    "<level>{message}</level>"
)


class LoggerConfig(BaseSettings):
    no_color: bool = Field(False, env="LOGGER_NO_COLOR")
    fmt_without_time: bool = Field(False, env="LOGGER_NO_TIME")

    @property
    def fmt(self) -> str:
        if self.fmt_without_time:
            return _FORMAT_WITHOUT_TIME
        return _FORMAT_WITH_TIME
