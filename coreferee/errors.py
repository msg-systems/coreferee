class CorefereeError(Exception):
    def __init__(self, text: str = ""):
        super().__init__()
        self.text = text

    def __str__(self) -> str:
        return self.text


class LanguageNotSupportedError(CorefereeError):
    pass


class ModelNotSupportedError(CorefereeError):
    pass


class VectorsModelNotInstalledError(CorefereeError):
    pass


class VectorsModelHasWrongVersionError(CorefereeError):
    pass


class OutdatedCorefereeModelError(CorefereeError):
    pass
