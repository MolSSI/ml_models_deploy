from typing import Optional
from pydantic import BaseModel, validator, root_validator, conint, confloat
from qcelemental.models import DriverEnum
from qc_time_estimator.processing.input_utils import get_nelec, get_nmo, get_element_counts
import pandas as pd
from datetime import datetime



class ExecutationTimeInput(BaseModel):

    # ---------- User Input
    nthreads: conint(ge=1) = 1
    cpu_clock_speed: confloat(ge=500, le=75000)
    cpu_launch_year: conint(ge=1990, le=datetime.now().year)

    driver: DriverEnum
    method: str             # TODO: get accepted list from qcelemental
    restricted: bool = False

    # --------

    molecule: Optional[str]       # excluded later, used to calc nmo and nelec
    basis_set: Optional[str]      # excluded later, used to calc nmo and nelec

    # ----------- Generated from User Input
    nelec: Optional[int]
    nmo: Optional[int]

    @root_validator
    def check_passwords_match(cls, values):

        # check accepted input pairs
        if values.get('molecule', None) and values.get('basis_set', None):
            return values

        # or (can't be none or <=0
        if values.get('nelec', None) and values.get('nmo', None):
            return values

        raise ValueError("('molecule, 'basis_set') or ('nelec', 'nmo') should be provided.")


    @validator('molecule')
    def validate_molecule(cls, v):
        return get_element_counts(v)

    @validator('nelec', always=True)
    def validate_nelec(cls, v, values):
        if v:
            return v
        else:
            return get_nelec(values.get('molecule'))

    @validator('nmo', always=True)
    def validate_nmo(cls, v, values):
        if v:
            return v
        else:
            return get_nmo(values.get('molecule'), values.get('basis_set'))


def validate_inputs(input_data: dict) -> dict:

    validated_data = []

    # Common use case is for only one row
    for i in input_data.index:
        validated_data.append(ExecutationTimeInput(**input_data.iloc[i]).dict(exclude={'molecule'}))

    return pd.DataFrame(validated_data)
