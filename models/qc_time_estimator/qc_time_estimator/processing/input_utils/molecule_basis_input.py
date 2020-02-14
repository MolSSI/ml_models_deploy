import rdkit
import rdkit.Chem.rdMolDescriptors
import qcelemental
import re
from collections import defaultdict
from typing import Dict
import basis_set_exchange as bse


def smiles_to_formula(string: str) -> str:
    """Converts a SMILES string to a chemical formula"""
    mol = rdkit.Chem.MolFromSmiles(string)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {string}")
    return rdkit.Chem.rdMolDescriptors.CalcMolFormula(rdkit.Chem.AddHs(mol))


def inchi_to_formula(string: str) -> str:
    """Converts an InChI string to a chemical formula"""
    try:
        mol = rdkit.Chem.MolFromInchi(string, removeHs=False, treatWarningAsError=True)
    except rdkit.Chem.InchiReadWriteError:
        raise ValueError(f"Invalid InChI: {string}")
    return rdkit.Chem.rdMolDescriptors.CalcMolFormula(rdkit.Chem.AddHs(mol))


def xyz_to_formula(string: str) -> str:
    """Converts a string representing an xyz file to a chemical formula"""
    try:
        return qcelemental.models.Molecule.from_data(
            string, dtype="xyz"
        ).get_molecular_formula()
    except qcelemental.exceptions.MoleculeFormatError:
        raise ValueError(f"Invalid xyz file: {string}")


def formula_to_elements(string: str) -> Dict[str, int]:
    """Converts a chemical formula to a list of elements and counts"""
    matches = re.findall("[A-Z][^A-Z]*", string)
    ret = defaultdict(int)
    for match in matches:
        match_n = re.match("(\D+)(\d*)", match)
        assert match_n
        if match_n.group(2) == "":
            n = 1
        else:
            n = int(match_n.group(2))

        ret[match_n.group(1)] += n
    return ret


def input_to_elements(string: str) -> Dict[str, int]:
    """Try to guess whether input is a formula, inchi, smiles, or xyz"""
    if string.startswith("InChI="):
        formula = inchi_to_formula(string)
    else:
        try:
            formula = smiles_to_formula(string)
        except ValueError:
            try:
                formula = xyz_to_formula(string)
            except ValueError:
                formula = string
    return formula_to_elements(formula)


def get_element_nbasis_map(basis_set_name: str) -> Dict[str, int]:
    """Returns a map of element to number of basis functions for a given basis set"""

    basis = bse.get_basis(basis_set_name, uncontract_general=1, header=False)

    element_nbasis_map = {}
    for atomic_nr_str in basis["elements"]:
        atomic_nr = int(atomic_nr_str)
        element = qcelemental.periodictable.to_E(atomic_nr)
        shells = []
        for shell in basis["elements"][atomic_nr_str]["electron_shells"]:
            shell.pop("region")
            ft = shell.pop("function_type")
            if ft in ["gto", "gto_spherical"]:
                shell["harmonic_type"] = "spherical"
            elif ft == "gto_cartesian":
                shell["harmonic_type"] = "cartesian"
            else:
                raise ValueError(
                    f"Can't determine harmonic type from function type: {ft}"
                )
            shells.append(qcelemental.models.basis.ElectronShell(**shell))
        element_nbasis_map[element] = sum([shell.nfunctions() for shell in shells])
    return element_nbasis_map


# ------------------ Public -- --------------------


def get_element_counts(molecule_input):
    element_counts = input_to_elements(molecule_input)

    return element_counts


def get_nmo(element_counts, basis_set_name) -> int:
    """Returns the features nmo from user molecule and basis set inputs"""

    element_nbasis_map = get_element_nbasis_map(basis_set_name)
    nmo = sum(
        [
            count * element_nbasis_map[element]
            for element, count in element_counts.items()
        ]
    )

    return nmo


def get_nelec(element_counts) -> int:

    nelec = sum(
        qcelemental.periodictable.to_Z(element) * count
        for element, count in element_counts.items()
    )
    return nelec
