##
##
##

import sys
from functools import reduce
from typing import Optional, Any


def get_class_by_name(class_path: str) -> Optional[Any]:
    cls = None
    try:
        components = class_path.split('.')
        if len(components) < 2:
            raise NameError()
        cls = reduce(getattr, components[1:], sys.modules[components[0]])
    except AttributeError:
        pass

    return cls


def create_instance(class_path: str, parent: Optional[Any], *args, **kwargs) -> Any:
    cls = get_class_by_name(class_path)
    if cls is None:
        raise NameError(f'No class with path \'{class_path}\' was found.')
    if parent is not None and not issubclass(cls, parent):
        raise TypeError(f'{cls} is not a subclass of {parent}.')

    return cls(*args, **kwargs)
