import pytest
from qc_time_estimator.processing.input_utils import list_basis_sets, \
     get_nelec, get_nmo, get_element_counts
from qc_time_estimator.processing.input_utils.molecule_basis_input import \
     get_element_nbasis_map, smiles_to_formula, inchi_to_formula, xyz_to_formula, \
     formula_to_elements, input_to_elements
from qcelemental.testing import compare_recursive


def test_list_basis_sets():
    assert ("nasa ames cc-pcv5z", "NASA Ames cc-pCV5Z") in list_basis_sets()


def test_get_element_nbasis_map():
    assert get_element_nbasis_map("sto-3g")["N"] == 5
    assert get_element_nbasis_map("6-31g*")["C"] == 15
    assert "In" not in get_element_nbasis_map("6-311+g*")


def test_smiles_to_formula():
    assert smiles_to_formula("N1C=Cc2ccccc12") == "C8H7N"
    with pytest.raises(ValueError):
        smiles_to_formula("N1C=Cc2ccccc1")


def test_inchi_to_formula():
    assert inchi_to_formula("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3") == "C2H6O"
    with pytest.raises(ValueError):
        inchi_to_formula("InChi=1S/C2H6O/c1-2-3/h3H,2H2,1H")
    with pytest.raises(ValueError):
        inchi_to_formula("1S/C2H6O/c1-2-3/h3H,2H2,1H3")


def test_xyz_to_formula():
    assert xyz_to_formula("3\n\nH 1 2 3\nO 4 5 6\nO 4 1 3\n") == "HO2"
    with pytest.raises(ValueError):
        xyz_to_formula("InChi=1S/C2H6O/c1-2-3/h3H,2H2,1H")


def test_formula_to_elements():
    assert compare_recursive(
        formula_to_elements("C2H6O"), {"C": 2, "H": 6, "O": 1}
    )
    assert compare_recursive(
        formula_to_elements("C2H6OOO"), {"C": 2, "H": 6, "O": 3}
    )


@pytest.mark.parametrize(
    "string,result",
    [
        ("N1C=Cc2ccccc12", {"C": 8, "H": 7, "N": 1}),
        ("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", {"C": 2, "H": 6, "O": 1}),
        ("3\n\nH 1 2 3\nO 4 5 6\nO 4 1 3\n", {"H": 1, "O": 2}),
        ("C2H6OOO", {"C": 2, "H": 6, "O": 3}),
    ],
)
def test_input_to_elements(string, result):
    assert compare_recursive(input_to_elements(string), result)


@pytest.mark.parametrize(
    "molecule,basis,nmo,nelec",
    [
        ("InChI=1S/CH4O/c1-2/h2H,1H3", "cc-pvdz", 48, 18),
        ("H2O", "6-31g*", 19, 10),
        ("CC1=CC=CC=C1", "def2-tzvpd", 331, 50),
    ],
)
def test_get_molecule_basis_features(molecule, basis, nmo, nelec):

    element_count = get_element_counts(molecule)
    assert get_nmo(element_count, basis) == nmo
    assert get_nelec(element_count) == nelec
