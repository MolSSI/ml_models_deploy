import basis_set_exchange as bse
from typing import Tuple
from .. import preprocessors


def list_basis_sets() -> Tuple[str, str]:
    return sorted((k, v['display_name']) for k, v in bse.get_metadata().items())


def list_methods() -> Tuple[str, str]:
    return sorted((k, k) for k in list(preprocessors.method_names))