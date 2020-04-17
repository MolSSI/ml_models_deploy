import basis_set_exchange as bse
from typing import Tuple


def list_basis_sets() -> Tuple[str, str]:
    return sorted((v['display_name'], v['display_name']) for k, v in bse.get_metadata().items())

