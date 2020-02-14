from qc_time_estimator.processing.validation import validate_inputs, DriverEnum
import pytest
import pandas as pd


@pytest.mark.parametrize(
    "molecule,basis,nmo,nelec,driver,cpu_launch_year,cpu_clock_speed",
    [
        ("InChI=1S/CH4O/c1-2/h2H,1H3", "cc-pvdz", 48, 18, DriverEnum.hessian, 2005, 1100),
        ("H2O", "6-31g*", 19, 10, DriverEnum.energy, 2020, 500),
        ("CC1=CC=CC=C1", "def2-tzvpd", 331, 50, DriverEnum.gradient, 1999, 3400),
    ],
)
def test_validation_success(molecule, basis, nmo, nelec, driver,
                            cpu_launch_year, cpu_clock_speed):
    data = dict(
        nthreads=1,
        cpu_clock_speed=cpu_clock_speed,
        cpu_launch_year=cpu_launch_year,
        driver=driver,
        method='b3lyp',
        restricted=False,
        molecule=molecule,
        basis_set=basis
    )

    val_data = validate_inputs(pd.DataFrame([data]))

    assert val_data.loc[0, 'nmo'] == nmo
    assert val_data.loc[0, 'nelec'] == nelec
    assert 'molecule' not in val_data


@pytest.mark.parametrize(
    "molecule,basis,cpu_launch_year,cpu_clock_speed,driver",
    [
        ("InChI=invalid", "cc-pvdz", 2015, 1400, DriverEnum.hessian),
        ("H2O", "6-31g*", 2005, 2, 'Invalid driver'),
        ("CC1=CC=CC=C1", "def2-tzvpd", 1900, 3100, DriverEnum.energy),
        ("CC1=CC=CC=C1", "def2-tzvpd", 2019, -2, DriverEnum.gradient),
    ],
)
def test_validation_error(molecule, basis, cpu_launch_year, cpu_clock_speed, driver):
    data = dict(
        nthreads=1,
        cpu_clock_speed=cpu_clock_speed,
        cpu_launch_year=cpu_launch_year,
        driver=driver,
        method='b3lyp',
        restricted=False,
        molecule=molecule,
        basis_set=basis
    )

    with pytest.raises(Exception):
        validate_inputs(pd.DataFrame([data]))

