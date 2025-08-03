import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles
from rdkit.Chem import Descriptors
from torch_geometric.utils import from_smiles as tg_from_smiles
import requests

def calculate_descriptors(mol):
    if mol is None:
        return torch.zeros((1, 25))

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.Chi0(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.MolMR(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.PEOE_VSA1(mol),
        Descriptors.PEOE_VSA2(mol),
        Descriptors.SMR_VSA1(mol),
        Descriptors.SMR_VSA2(mol),
        Descriptors.EState_VSA1(mol),
        Descriptors.EState_VSA2(mol),
        Descriptors.VSA_EState1(mol)
    ]

    return torch.tensor(descriptors, dtype=torch.float).view(1, -1)

def from_smiles(smile):
    return tg_from_smiles(smile)

def calculate_descriptors_solv(mol):
    """
    Calculate molecular descriptors for a given molecule.

    Args:
        mol (RDKit Mol object): Molecule for which to calculate descriptors.

    Returns:
        torch.Tensor: Descriptors as a tensor.
    """
    if mol is None:
        return torch.zeros(1, 5, dtype=torch.float)

    descriptors = [
        Descriptors.MolWt(mol),       # Molecular weight
        Descriptors.TPSA(mol),        # Topological polar surface area
        Descriptors.MolLogP(mol),     # LogP (octanol-water partition coefficient)
        Descriptors.NumHDonors(mol),  # Number of hydrogen bond donors
        Descriptors.NumHAcceptors(mol) # Number of hydrogen bond acceptors
    ]
    
    return torch.tensor(descriptors, dtype=torch.float).view(1, -1)

def get_edge_features(mol):
    """
    Extract edge features for a given molecule in SMILES form.
    
    Args:
        mol (RDKit Mol object): Molecule for which to extract edge features.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Edge indices and edge attributes.
    """
    bond_features = []
    edge_index = []

    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    num_bond_types = 4  # One-hot length for bond types

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # One-hot encode the bond type
        bond_type = bond_type_to_idx.get(bond.GetBondType(), -1)
        bond_type_one_hot = [0] * num_bond_types
        if 0 <= bond_type < num_bond_types:
            bond_type_one_hot[bond_type] = 1
        else:
            continue  # Skip unrecognized bond types

        # Additional boolean edge features
        extra_features = [
            float(bond.GetIsConjugated()),  # Is conjugated (1 or 0)
            float(bond.IsInRing())          # Is part of a ring (1 or 0)
        ]

        features = bond_type_one_hot + extra_features

        # Add bidirectional edges (i, j) and (j, i)
        edge_index += [[i, j], [j, i]]
        bond_features += [features, features]

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_types + 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # Shape [2, num_edges]
        edge_attr = torch.tensor(bond_features, dtype=torch.float)  # Shape [num_edges, feature_dim]

    return edge_index, edge_attr

def process_smiles_pair(smile1, smile2):
    """
    Process a pair of SMILES strings and return graph objects for both molecules.
    
    Args:
        smile1 (str): SMILES string for the first molecule (polymer).
        smile2 (str): SMILES string for the second molecule (solvent).
        
    Returns:
        Tuple[Data, Data]: Processed graph objects for both molecules.
    """
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    # Create graph objects from SMILES
    g1 = from_smiles(smile1)
    g2 = from_smiles(smile2)

    # Ensure node features are floats
    g1.x = g1.x.float()
    g2.x = g2.x.float()

    # Compute edge features
    g1.edge_index, g1.edge_attr = get_edge_features(mol1)
    g2.edge_index, g2.edge_attr = get_edge_features(mol2)

    # Compute molecular descriptors
    descriptor_tensor1 = calculate_descriptors_solv(mol1)
    descriptor_tensor2 = calculate_descriptors_solv(mol2)

    # Add descriptors to graph objects
    g1.descriptors = descriptor_tensor1
    g2.descriptors = descriptor_tensor2

    # Return the graph pair
    return g1, g2

def classify_solvent_type(solvent_smiles):
    """Classify the solvent as Protic, Aprotic, or Intermediate based on functional groups."""
    
    # Convert SMILES string to RDKit molecule object
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    
    if solvent_mol is None:
        return "Invalid SMILES"
    
    # Define functional groups
    protic_groups = [
        Chem.MolFromSmiles('O'),  # Hydroxyl group (-OH)
        Chem.MolFromSmiles('N'),  # Amine group (-NH, -NH2)
        Chem.MolFromSmiles('S'),  # Thiol group (-SH)
        Chem.MolFromSmiles('C(=O)O'),  # Carboxyl group (-COOH)
    ]
    
    aprotic_groups = [
        Chem.MolFromSmiles('C(O)C'),  # Ether group (-O-)
        Chem.MolFromSmiles('C=O'),    # Carbonyl group
        Chem.MolFromSmiles('C(C)C'),  # Alkyl groups (-CH3, -CH2-)
        Chem.MolFromSmiles('C#N'),    # Nitrile group (-CN)
        Chem.MolFromSmiles('CCl'),    # Chlorine attached to carbon
    ]
    
    intermediate_groups = [
        Chem.MolFromSmiles('C=O'),  # Carbonyl group (both protic and aprotic solvents can have this)
        Chem.MolFromSmiles('O'),    # Esters (weaker H-bonding than alcohols)
        Chem.MolFromSmiles('N=O'),  # Nitro groups
        Chem.MolFromSmiles('C-O'),  # Ethers (-O-) in intermediates
    ]
    
    # Check for protic groups (strong hydrogen bond donors)
    for group in protic_groups:
        if solvent_mol.HasSubstructMatch(group):
            return "Protic"
    
    # Check for aprotic groups (no hydrogen bond donors)
    for group in aprotic_groups:
        if solvent_mol.HasSubstructMatch(group):
            return "Aprotic"
    
    # Check for intermediate (ambiguous) groups
    for group in intermediate_groups:
        if solvent_mol.HasSubstructMatch(group):
            return "Intermediate"
    
    # If no match, return unknown
    return "Unknown"

def estimate_solvent_polarity(solvent_smiles):
    """Estimate polarity using RDKit's MolLogP and partial charge distribution."""
    
    # Convert SMILES to molecule object
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    
    if solvent_mol is None:
        return "Invalid SMILES", None, None
    
    # Calculate LogP (higher LogP means more non-polar)
    logP = Descriptors.MolLogP(solvent_mol)
        
    # Heuristic to classify polarity
    if logP < 0:
        polarity = "High polarity"
    elif logP < 2:
        polarity = "Moderate polarity"
    else:
        polarity = "Low polarity"
    
    return polarity

def get_trivial_name(smiles):
    # Preprocess the SMILES string to replace wildcards with known groups
    smiles = smiles.replace("[*]", "C").replace("R", "C")
    
    # Try to get the trivial name from PubChem (if available)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/StandardName/TXT"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.text.strip()
    
    # If the trivial name is not available, identify functional group and generate a name
    else:
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return "Invalid SMILES"

        # Define substructure patterns for functional groups
        functional_groups = {
            "Vinyl": "[C]=[C]",
            "Acrylic": "[C]=O",
            "Carboxylic Acid": "C(=O)O",
            "Hydroxy": "O",
            "Acrylonitrile": "C#N",
            "Ester": "C(=O)O",
            "Aldehyde": "C=O",
            "Ketone": "C(=O)C",
            "Amine": "N",
            "Thiol": "S",
            "Isocyanate": "N=C=O",
            "Epoxide": "C1O1C1"
        }
        
        # Check for functional group substructure match
        for group, pattern in functional_groups.items():
            # Convert pattern to RDKit molecule
            substructure = Chem.MolFromSmarts(pattern)
            
            # Check if substructure is valid
            if substructure is None:
                print(f"Invalid SMARTS pattern for {group}: {pattern}")
                continue  # Skip to the next functional group pattern
            
            if mol.HasSubstructMatch(substructure):
                return f"{group} Monomer"
        
        # If no known functional group is found
        return "Unknown Monomer"