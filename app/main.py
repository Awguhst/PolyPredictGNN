import streamlit as st
import torch
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from torch_geometric.data import Data
from rdkit.Chem import Descriptors
from torch_geometric.utils import from_smiles
from torch_geometric.nn import global_max_pool, global_mean_pool
from models_gnn import SolubilityGNN, HybridGNN
from utils import process_smiles_pair, classify_solvent_type, estimate_solvent_polarity, calculate_descriptors, get_trivial_name

# Custom CSS
st.markdown("""
    <style>
        /* Base background & text */
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #f1f1f1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Headings */
        h1, h2, h3 {
            color: #61dafb;
            font-weight: 700;
            letter-spacing: 1px;
        }

        /* Buttons */
        .stButton>button {
            color: white;
            background: linear-gradient(135deg, #1e5a84, #3186e1);  /* Darker initial color */
            padding: 0.6rem 1.5rem;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: background 0.3s ease, transform 0.2s ease;
        }

        /* Hover state */
        .stButton>button:hover {
            background: linear-gradient(135deg, #3186e1, #1e5a84);  /* Slightly darker hover color */
            cursor: pointer;
            color: white;  /* Prevent red text on hover */
            transform: scale(1.05);  /* Add slight scale effect */
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            background-color: #1c1c1c;
            color: #f1f1f1;
            border: 1.5px solid #3a86ff;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            transition: border-color 0.3s ease;
        }

        .stTextInput>div>div>input:focus {
            border-color: #61dafb;
            outline: none;
            box-shadow: 0 0 8px #61dafb;
        }

        /* Form container */
        .stForm {
            background-color: #1a1a1a;
            padding: 25px 30px;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(97, 218, 251, 0.15);
        }

        /* Metric boxes */
        .stMetric {
            background-color: #222222;
            color: #f1f1f1;
            border-radius: 12px;
            padding: 16px 20px;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: inset 0 0 5px rgba(97, 218, 251, 0.2);
        }

        /* Center molecule image */
        .css-1d391kg img {
            border-radius: 16px;
            box-shadow: 0 0 20px rgba(97, 218, 251, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the HybridGNN model for Tg/Tm prediction
tg_tm_model = HybridGNN(hidden_channels=128, descriptor_size=12, dropout_rate=0.6)
tg_tm_model.load_state_dict(torch.load("../models/hybrid_gnn.pt", map_location=device))
tg_tm_model.eval().to(device)

# Load the SolubilityGNN model for solubility prediction
solubility_model = SolubilityGNN(hidden_channels=128, descriptor_size=5, dropout_rate=0.3)
solubility_model.load_state_dict(torch.load("../models/solubility_gnn.pt", map_location=device))
solubility_model.eval().to(device)

# Load Scalers
scaler = joblib.load("../models/scaler.pkl")
pca = joblib.load("../models/pca.pkl")

# Load training data for similarity search (Tg/Tm model)
@st.cache_data
def load_training_data():
    df = pd.read_csv("../data/polymer_tg_tm.csv")

    # Normalize column name just once
    df.rename(columns={"SMILES": "smiles"}, inplace=True)

    df = df[df["smiles"].notna()]
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df["fingerprint"] = df["mol"].apply(
        lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) if m else None
    )
    return df.dropna(subset=["fingerprint"])

training_df = load_training_data()

# Function to predict solubility
def predict_solubility(polymer_smiles_input, solvent_smiles_input):
    """ Solubility prediction logic """
    polymer_mol = Chem.MolFromSmiles(polymer_smiles_input)
    solvent_mol = Chem.MolFromSmiles(solvent_smiles_input)
    
    # Create graphs and process as needed 
    g1, g2 = process_smiles_pair(polymer_smiles_input, solvent_smiles_input) 
    
    g1.x = g1.x.float().to(device)
    g2.x = g2.x.float().to(device)
    descriptor_tensor1 = g1.descriptors.to(device)
    descriptor_tensor2 = g2.descriptors.to(device)

    g1.edge_index = g1.edge_index.to(device)
    g1.edge_attr = g1.edge_attr.to(device)
    g2.edge_index = g2.edge_index.to(device)
    g2.edge_attr = g2.edge_attr.to(device)

    batch_index1 = torch.zeros(g1.x.size(0), dtype=torch.long).to(device)
    batch_index2 = torch.zeros(g2.x.size(0), dtype=torch.long).to(device)
    
    # Run the solubility model
    with torch.no_grad():
        output, _ = solubility_model(g1.x, g1.edge_index, batch_index1, descriptor_tensor1, g1.edge_attr, g2.x, g2.edge_index, batch_index2, descriptor_tensor2, g2.edge_attr)
    
    solvent_characteristic = torch.sigmoid(output).cpu().numpy()[0][0]
    return solvent_characteristic

# Function to predict Tg/Tm
def predict_properties(smiles_input):
    """ Tg/Tm prediction logic """
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("❌ Invalid SMILES string.")
        return None, None
    descriptors = calculate_descriptors(mol)  
    desc_scaled = scaler.transform(descriptors)
    desc_pca = pca.transform(desc_scaled)
    desc_tensor = torch.tensor(desc_pca, dtype=torch.float).to(device)
    
    graph = from_smiles(smiles_input)  
    graph.x = graph.x.float().to(device)
    graph.edge_index = graph.edge_index.to(device)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
    
    # Run the Tg/Tm model
    with torch.no_grad():
        pred, _ = tg_tm_model(graph.x, graph.edge_index, batch_index, desc_tensor)
    
    tg, tm = pred[0].cpu().numpy()
    return tg, tm

# Function to find similar monomers
def find_similar_monomers(smiles_input):
    """ Find top 5 similar monomers based on fingerprint similarity """
    mol = Chem.MolFromSmiles(smiles_input)
    input_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    def tanimoto(fp1, fp2):
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    training_df["similarity"] = training_df["fingerprint"].apply(lambda fp: tanimoto(input_fp, fp))
    top_similar = training_df.sort_values(by="similarity", ascending=False).head(5)
    
    return top_similar

# Streamlit interface configuration
st.set_page_config(page_title="Polymer Property Prediction", layout="wide", page_icon="🧬")

# Main App UI
st.title("Polymer Property Prediction & Solubility Assessment")
st.markdown("Leverage machine learning models to predict key polymer properties, including **Glass Transition Temperature (Tg)**, **Melting Temperature (Tm)**, and **Solubility** in specific solvents. Additionally, explore **Top 5 Chemically Similar Monomers** and **Top 5 Solvent Recommendations** for your polymer.")

# Sidebar Section: Inputs for SMILES
st.sidebar.header("SMILES Input")
polymer_smiles_input = st.sidebar.text_input("Monomer SMILES:", "*SC(C)C(*)=O", help="Enter monomer structure in SMILES format")
solvent_smiles_input = st.sidebar.text_input("Solvent SMILES (Optional):", "ClC(Cl)Cl", help="Enter solvent structure in SMILES format")

# Prediction Button
if st.sidebar.button("Predict"):
    with st.spinner("🔬 Analyzing Molecules..."):
        # Check if SMILES are valid
        polymer_mol = Chem.MolFromSmiles(polymer_smiles_input)
        solvent_mol = Chem.MolFromSmiles(solvent_smiles_input) if solvent_smiles_input else None
        
        if not polymer_smiles_input:
            st.error("❌ Please enter a valid polymer SMILES string.")
        elif polymer_mol is None:
            st.error("❌ Invalid polymer SMILES string. Please enter a valid SMILES.")
        elif solvent_mol is None and solvent_smiles_input:
            st.error("❌ Invalid solvent SMILES string. Please enter a valid SMILES.")
        else:
            m_col1, m_col2, m_col3 = st.columns([1, 1, 1.2])
            with m_col1:
                st.subheader("Monomer 2D Structure")
                mol_img_polymer = Draw.MolToImage(polymer_mol, size=(700, 700))  # Adjust size to 300x300
                st.image(mol_img_polymer, use_container_width=True)

            # Second column: Solvent 2D Structure (if available)
            if solvent_smiles_input:
                with m_col2:
                    st.subheader("Solvent 2D Structure")
                    mol_img_solvent = Draw.MolToImage(solvent_mol, size=(700, 700))  # Adjust size to 300x300
                    st.image(mol_img_solvent, use_container_width=True)
                    
            # Predict Polymer Properties (Tg/Tm)
            tg, tm = predict_properties(polymer_smiles_input)
            st.subheader("📈 Predicted Properties")
            
            # Display Tg and Tm
            col1, col2 ,col3 = st.columns(3)
            col1.metric(label="Glass Transition Temperature (Tg)", value=f"{tg:.2f} °C")
            col2.metric(label="Melting Temperature (Tm)", value=f"{tm:.2f} °C")
            
            if solvent_smiles_input:
                # Predict Solubility (solvability) in the given solvent
                with torch.no_grad():
                    g1, g2 = process_smiles_pair(polymer_smiles_input, solvent_smiles_input)
                    g1.x = g1.x.float().to(device)
                    g2.x = g2.x.float().to(device)
                    descriptor_tensor1 = g1.descriptors.to(device)
                    descriptor_tensor2 = g2.descriptors.to(device)
                    g1.edge_index = g1.edge_index.to(device)
                    g1.edge_attr = g1.edge_attr.to(device)
                    g2.edge_index = g2.edge_index.to(device)
                    g2.edge_attr = g2.edge_attr.to(device)
                    batch_index1 = torch.zeros(g1.x.size(0), dtype=torch.long).to(device)
                    batch_index2 = torch.zeros(g2.x.size(0), dtype=torch.long).to(device)

                    # Run the solubility model
                    output, _ = solubility_model(g1.x, g1.edge_index, batch_index1, descriptor_tensor1, g1.edge_attr, 
                                                g2.x, g2.edge_index, batch_index2, descriptor_tensor2, g2.edge_attr)

                solvent_characteristic = torch.sigmoid(output).cpu().numpy()[0][0]
                
                # Display Solubility Prediction
                with col3:
                    st.metric(label="Probability of Polymer Being Solvable in This Solvent", value=f"{solvent_characteristic * 100:.2f}%")

            # Create a new set of columns for the **Top 5 Similar Monomers** and **Top 5 Solvents** side by side
            similar_monomers = find_similar_monomers(polymer_smiles_input)

            # Create two main columns for Similar Monomers and Solvents
            col1, col2 = st.columns(2)  # Two main columns: one for Similar Monomers and one for Solvents

            # Column 1: Similar Monomers
            with col1:
                st.subheader("🔍 Similar Monomers")
                for _, row in similar_monomers.iterrows():
                    img_col, text_col = st.columns([1, 1.9])  # One for image, one for text

                    # Monomer Image
                    with img_col:
                        st.image(Draw.MolToImage(row["mol"], size=(700, 700)), use_container_width=True)

                    # Get Trivial Name (using PubChem or RDKit)
                    trivial_name = get_trivial_name(row['smiles'])  # If using PubChem example

                    # Monomer Details
                    with text_col:
                        st.markdown(f"**Name**: `{trivial_name}`")  # Trivial Name
                        st.markdown(f"**SMILES**: `{row['smiles']}`")
                        st.markdown(f"**Similarity**: `{row['similarity']:.2f}`")
                        st.markdown(f"**Tg**: `{row['Tg']:.2f} °C`")
                        st.markdown(f"**Tm**: `{row['Tm']:.2f} °C`")

            # Column 2: Solvents
            with col2:
                st.subheader("💧 Alternative Solvents")
                solvent_data = pd.read_csv('../data/solvent_smiles.csv')  # Assuming solvent data
                top_solvents = []

                # Recompute g1 and solvent properties for each solvent
                for _, row in solvent_data.iterrows():
                    solvent_smiles = row['solvent_smiles']
                    solvent_name = row['solvent']  # Get the trivial solvent name
                    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                    
                    # Recompute g1 for the polymer (polymer_smiles_input) and g2 for the solvent (solvent_smiles)
                    g1, g2 = process_smiles_pair(polymer_smiles_input, solvent_smiles)
                    
                    # Process g1 (Polymer Graph)
                    g1.x = g1.x.float().to(device)
                    descriptor_tensor1 = g1.descriptors.to(device)
                    g1.edge_index = g1.edge_index.to(device)
                    g1.edge_attr = g1.edge_attr.to(device)
                    batch_index1 = torch.zeros(g1.x.size(0), dtype=torch.long).to(device)
                    
                    # Process g2 (Solvent Graph)
                    g2.x = g2.x.float().to(device)
                    descriptor_tensor2 = g2.descriptors.to(device)
                    g2.edge_index = g2.edge_index.to(device)
                    g2.edge_attr = g2.edge_attr.to(device)
                    batch_index2 = torch.zeros(g2.x.size(0), dtype=torch.long).to(device)

                    # Predict solubility
                    with torch.no_grad():
                        output, _ = solubility_model(g1.x, g1.edge_index, batch_index1, descriptor_tensor1, g1.edge_attr,
                                                    g2.x, g2.edge_index, batch_index2, descriptor_tensor2, g2.edge_attr)
                    solubility = torch.sigmoid(output).cpu().numpy()[0][0]
                    top_solvents.append((solvent_name, solvent_smiles, solubility))  # Store name, smiles, and solubility

                # Sort and Display Top 5 Solvents
                top_solvents = sorted(top_solvents, key=lambda x: x[2], reverse=True)[:5]

                # Display Solvent Images and Details side by side
                for i, (solvent_name, solvent_smiles, solubility) in enumerate(top_solvents):
                    # Create two sub-columns for image and text next to each other
                    img_col, text_col = st.columns([1, 1.9])  # One for image, one for text

                    # Solvent Image
                    with img_col:
                        st.image(Draw.MolToImage(Chem.MolFromSmiles(solvent_smiles), size=(700, 700)))

                    # Solvent Details
                    with text_col:
                        st.markdown(f"**Trivial Name**: `{solvent_name}`")
                        st.markdown(f"**SMILES**: `{solvent_smiles}`")
                        st.markdown(f"**Predicted Solubility**: `{solubility * 100:.2f}%`")
                        solvent_type = classify_solvent_type(solvent_smiles)
                        polarity = estimate_solvent_polarity(solvent_smiles)
                        st.markdown(f"**Solvent Type**: `{solvent_type}`")
                        st.markdown(f"**Solvent Polarity**: `{polarity}`")

            # Success message
            st.success("✅ Prediction complete!")


