 
try:
    import redrock
except (ImportError, ModuleNotFoundError):
    HAS_REDROCK = False
else:
    HAS_REDROCK = True