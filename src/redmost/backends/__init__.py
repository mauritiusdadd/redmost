
import redmost.backends.rrock as rrock
try:
    import redmost.backends.xcorr as xcorr
except (ModuleNotFoundError, ImportError):
    HAS_XCORR = False
else:
    HAS_XCORR = True

HAS_REDROCK = rrock.HAS_REDROCK
