import numpy as np
from rdkit import Chem
from smiles_encoder import SmilesEncoder
import matminer.featurizers.composition as mm_composition
import matminer.featurizers.structure as mm_structure
from pymatgen.core import Composition
from ase import Atoms
from dscribe.descriptors import SOAP
from rdkit.Chem import AllChem

# # 示例：一组 SMILES
# smiles_list = ["C1=CC=CC=C1"]
# encoder = SmilesEncoder(smiles_list)
# encoded_smiles = encoder.encode_many(smiles_list)
# decoded_smiles = encoder.decode_many(encoded_smiles)
# print(len(encoded_smiles[0]), decoded_smiles)

def get_SOAP_descriptor(mol, vdw_max):
    SOAP_mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(SOAP_mol)
    AllChem.UFFOptimizeMolecule(SOAP_mol)
    conf = SOAP_mol.GetConformer()
    positions = []
    symbols = []
    for atom in SOAP_mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())
    positions = np.array(positions)

    ase_mol = Atoms(symbols=symbols, positions=positions)

    species = list(set(symbols))
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=2*vdw_max,
        n_max=8,
        l_max=6,
    )
    soap_descriptor = soap.create(ase_mol)

    return soap_descriptor

def get_reproted_descriptor(formula, mol, vdw_max):
    comp = Composition(formula)
    print(comp)
    md_featurizer = mm_composition.Meredig()
    MD_descriptor = md_featurizer.featurize(comp)

    # os_featurizer = mm_composition.OxidationStates()
    # OS_descriptor = os_featurizer.featurize(comp)

    sc_featurizer = mm_structure.StructuralComplexity()

    vo_featurizer = mm_composition.ValenceOrbital()
    VO_descriptor = vo_featurizer.featurize(comp)

    yss_featurizer = mm_composition.YangSolidSolution()
    YSS_descriptor = yss_featurizer.featurize(comp)

    SOAP_descriptor = get_SOAP_descriptor(mol, vdw_max)

    return MD_descriptor, VO_descriptor, YSS_descriptor, SOAP_descriptor

comp = Composition('C9H13BO3')
mol = Chem.MolFromSmiles('B(O)(O)OC1=CC=CC(=C1)C(C)C')
vdw_max = 8.0
MD_descriptor, VO_descriptor, YSS_descriptor, SOAP_descriptor = get_reproted_descriptor(comp, mol, vdw_max)
print(len(MD_descriptor), len(VO_descriptor), len(YSS_descriptor), SOAP_descriptor.shape)