# /src/data/collate/__init__.py

registered_collates = {}
def register_collate(name):
    def decorator(func):
        registered_collates[name] = func
        return func
    return decorator