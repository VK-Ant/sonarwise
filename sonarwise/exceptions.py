"""Custom exceptions for sonarwise."""


class SonarWiseError(Exception):
    """Base exception for sonarwise."""
    pass


class AudioFormatError(SonarWiseError):
    """Unsupported or corrupted audio format."""
    pass


class ModelLoadError(SonarWiseError):
    """Failed to load a model."""
    pass


class DeviceError(SonarWiseError):
    """GPU/device not available."""
    pass


class IndexError(SonarWiseError):
    """File already indexed or not found in index."""
    pass


class SpeakerNotFoundError(SonarWiseError):
    """Speaker not registered."""
    pass


class StreamError(SonarWiseError):
    """Live stream connection failed."""
    pass


class ExportError(SonarWiseError):
    """Export operation failed."""
    pass
