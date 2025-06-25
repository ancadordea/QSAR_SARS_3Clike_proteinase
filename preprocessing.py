import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split
import pickle

def preprocess(): 
    selection = ['molecule_chembl_id', 'smiles', 'bioactivity_class', 'standard_value']

    df = pd.read_csv('raw_CHEMBL3927_SARS_3Clike_proteinase.csv', sep = ";")

    # Clean column names: lowercase + replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Use only IC50 nM to target a specific type of bioactivity interaction and for unit consistency
    df = df[df['standard_type'] == "IC50"]

    df = df[df['standard_value'].notna()].reset_index(drop=True)

    # Classify comppunds into inactive, active or intermediate, values are IC50 unit (nM).
    df['bioactivity_class'] = df['standard_value'].apply(lambda x: 'inactive' if x >= 10000 else 'active' if x <= 1000 else 'intermediate')

    # Extract key columns, add bioactivity class labels, and save the cleaned dataset for modeling
    df = df[selection]

    return df

def add_lipinski_descriptors(df):
    """Calculates Lipinski descriptors of drug-likeness of compounds(ADME) for each molecule"""
    df = df.copy()
    mols = df['smiles'].apply(Chem.MolFromSmiles)
    
    df['MW'] = mols.apply(Descriptors.MolWt) # Size of the molecule 
    df['LogP'] = mols.apply(Descriptors.MolLogP) # Solubility
    df['NumHDonors'] = mols.apply(Lipinski.NumHDonors) # Hydrogen bond donors
    df['NumHAcceptors'] = mols.apply(Lipinski.NumHAcceptors) # Hydrogen bond acceptors
    
    return df

# Check data distribution 
def hist_ic50(df):
    plt.hist(df['standard_value'].clip(upper=1e5), bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('IC50 (nM)'), plt.ylabel('Count'), plt.title('IC50 Histogram')
    plt.tight_layout(), plt.savefig('Results/IC50_histogram.png')
    plt.close()

# Convert IC50 to pIC50 (to allow IC50 data to be more uniformly distributed)
def convert_to_pIC50(df):
    """Convert standard_value (IC50 in nM) to pIC50"""
    df2 = df.copy()
    df2['standard_value'] = -np.log10(df['standard_value'].clip(upper=1e8) * 1e-9) # Cap values at 100,000,000 (1e8), convert to molar(M) and apply -log10
    df2.rename(columns={'standard_value': 'pIC50_value'}, inplace=True) # Replace standard_value column with new pic50 values
    df2.to_csv('bioactivity_data.csv', index=False)
    return df2

# Check data distribution after conversion
def hist_pIC50(df2):
    plt.figure()
    plt.hist(df2['pIC50_value'], bins=50, color='mediumseagreen', edgecolor='black')
    plt.xlabel('pIC50'), plt.ylabel('Count'), plt.title('pIC50 Histogram')
    plt.tight_layout(), plt.savefig('Results/pIC50_histogram.png')
    plt.close()

# Check bioactivity levels to decide whether to remove intermediate class or merge it with active/inactive
def plot_bioactivity(df2):
    """Violin plot to check bioactivity amongst classess"""
    order = ['active', 'intermediate', 'inactive']
 
    counts = df2['bioactivity_class'].value_counts()
    new_labels = [f"{cls.capitalize()} ({counts.get(cls, 0)})" for cls in order]
    
    plt.figure(figsize=(9,6))
    sns.violinplot(data=df2, x='bioactivity_class', y='pIC50_value', order=order, palette = {'active':'green', 'intermediate':'orange', 'inactive':'red'}, inner=None, alpha=0.4)
    sns.stripplot(data=df2, x='bioactivity_class', y='pIC50_value', order=order, size=4.5, jitter=0.25, alpha=0.6, color='k')

    meaning = r"$\bf{pIC50: meaning}$" + "\n >6: Potent\n 5-6: Moderate\n <5: Weak"
    plt.text(2.15, 6.4, meaning, fontsize=10, bbox=dict(facecolor='white', edgecolor='black')) 

    plt.ylim(2, 8), plt.yticks(np.arange(2, 8.5, 0.5)), plt.xticks(ticks=range(len(order)), labels=new_labels)
    plt.xlabel('Bioactivity Class', weight='bold'), plt.ylabel('pIC50 Value', weight='bold')
    plt.tight_layout(), plt.grid(axis='y', linestyle='--', alpha=0.5), plt.savefig('Results/bioactivity_class_distribution.png')
    plt.close()

# Kruskal Wallis + Dunn's test to check significance between groups
def kruskal_dunn(df2):
    groups = df2.groupby('bioactivity_class')['pIC50_value']  # Group data

    h, p = stats.kruskal(*[groups.get_group(g) for g in groups.groups])  # Kruskal–Wallis test
    dunn = sp.posthoc_dunn(df2, val_col='pIC50_value', group_col='bioactivity_class', p_adjust='bonferroni')  # Post hoc Dunn's test  # noqa: F841

# Check physicochemical properties of active vs. inactive 
def scatterplot_MW_LogP(df3):
    plt.figure(figsize=(5.5, 5.5))

    sns.scatterplot(data=df3, x='MW', y='LogP', hue='bioactivity_class', palette = {'active':'green', 'inactive':'red'}, size='pIC50_value', edgecolor='black', alpha=0.7)

    plt.xlabel('Molecular Weight', fontsize=14, fontweight='bold'), plt.ylabel('LogP', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0), plt.savefig('Results/plot_MW_vs_LogP.png', bbox_inches='tight')
    plt.close()

# Plot Lipinksi descriptors
def plot_and_test_lipinski(df3, descriptors, bioactivity_class='bioactivity_class'):
    """Generate box plots and run Mann-Whitney"""
    for desc in descriptors:
        plt.figure(figsize=(5.5, 5.5))
        ax = sns.boxplot(data=df3, x=bioactivity_class, y=desc,  palette = {'active':'green', 'inactive':'red'})
        
        medians = df3.groupby(bioactivity_class)[desc].median() # Add median annotation
        for tick, label in enumerate(ax.get_xticklabels()):
            ax.text(tick, medians[label.get_text()], f'{medians[label.get_text()]:.2f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='semibold', color='black')

        plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold'), plt.ylabel(desc, fontsize=14, fontweight='bold')
        plt.title(f'{desc} by Bioactivity Class', fontsize=12), plt.tight_layout(), plt.savefig(f'Results/{desc}_boxplot.png')
        plt.close()

        active = df3[df3[bioactivity_class] == 'active'][desc]
        inactive = df3[df3[bioactivity_class] == 'inactive'][desc]
        stat, p = mannwhitneyu(active, inactive)
        interpretation = "Different distribution (reject H₀)" if p < 0.05 else "Same distribution (fail to reject H₀)"
        print(f"{desc:<15} U={stat:.1f}  p={p:.5f}  =  {interpretation}")

# Calculate structural fingerprints
    # Morgan fingerprints (ECFP) are sensitive to the connectivity of atoms. If SMILES includes:
    # Salts (Cl⁻, Na⁺), solvents, molecules with ambiguous representations (like nitro groups)
def clean_molecule(mol):
    """Remove salts/ions and keep only the largest organic fragment. Return None if molecule is invalid."""
    if mol is None:
        return None
    try:
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol_clean = lfc.choose(mol)
        if mol_clean is None or mol_clean.GetNumAtoms() == 0:
            return None
        return mol_clean
    except Exception as e:
        print(f"[clean_molecule] Cleaning failed: {e}")
        return None

# Creates a Morgan fingerprint, also known as ECFP (Extended Connectivity Fingerprint)
    # = a binary vector (length = nBits) that encodes substructures in the molecule
def mol_to_fp(mol, radius=2, nBits=1024): #
    if mol is None:
        return np.zeros(nBits, dtype=int)  # If the molecule is invalid/missing, return a zero fingerprint vector
    bitInfo = {}
    # Generate a Morgan fingerprint, radius=2: atom + neighbors + their neighbors (2 bonds out) -- most common for ECFP4 (because radius 2 × 2 = diameter 4)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo) 
    arr = np.zeros((nBits,), dtype=int)  # Create an empty numpy array to hold the fingerprint bits
    DataStructs.ConvertToNumpyArray(fp, arr) # Extract the bits from the RDKit fingerprint object into a standard NumPy array
    return arr, bitInfo

def collect_bitinfo_mapping(df3, radius=2, nBits=1024):
    mols = df3['smiles'].apply(Chem.MolFromSmiles).apply(clean_molecule)
    bit_infos = []  # List of bitInfo dicts, one per molecule

    for mol in mols:
        _, bitInfo = mol_to_fp(mol, radius=radius, nBits=nBits) 
        bit_infos.append(bitInfo)

    return mols, bit_infos

def generate_morgan_fingerprints(df3, radius=2, nBits=1024):
    df3 = df3.copy()
    mols = df3['smiles'].apply(Chem.MolFromSmiles).apply(clean_molecule) # Clean molecule
    fps = mols.apply(lambda mol: mol_to_fp(mol, radius, nBits)[0])  # Call mol to fp and extract just fingerprint array (not bitInfo)
    fp_matrix = np.stack(fps.values)
    fp_df = pd.DataFrame(fp_matrix, columns=[f'FP_{i}' for i in range(nBits)])
    
    df4 = pd.concat([df3[['molecule_chembl_id', 'smiles']].reset_index(drop=True), fp_df], axis=1)
    df4.to_csv('fingerprints.csv', index=False)

    return df4

def split_data(X, y_class, y_value, test_size=0.2, random_state=42):
    """
    Splits features and targets into train and test sets, stratifying by class labels.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix
        y_class (pd.Series or np.ndarray): Categorical labels for stratification
        y_value (pd.Series or np.ndarray): Continuous target values
        test_size (float): Proportion of data to reserve for testing
        random_state (int): Random seed for reproducibility

    Returns:
        X_train, X_test, y_class_train, y_class_test, y_train, y_test
    """
    return train_test_split(
        X, y_class, y_value,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class # To stratify split based on class balance
    )

if __name__ == "__main__":
    os.makedirs('Results', exist_ok=True) # Make dir for graphs
    # Preprocessing and EDA/CSA
    df =  preprocess()
    df = add_lipinski_descriptors(df)
    hist_ic50(df)
    df2 = convert_to_pIC50(df)
    plot_bioactivity(df2)
    hist_pIC50(df2)
    kruskal_dunn(df2)
    df3 = df2[df2['bioactivity_class'] != 'intermediate'] # Drop intermediates as there is statistical significance between the groups
    scatterplot_MW_LogP(df3)
    descriptors = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'] # List of descriptors to for Lipinski func
    plot_and_test_lipinski(df3, descriptors, bioactivity_class='bioactivity_class')
    mols, bit_infos = collect_bitinfo_mapping(df3, radius=2, nBits=1024)
    df4 = generate_morgan_fingerprints(df3)

    # Data split for ML
    X = df4.drop(columns=['molecule_chembl_id', 'smiles'])  # Keep only numeric columns
    y_value = df3['pIC50_value']         # Continuous target for regression
    y_class = df3['bioactivity_class']   # Categorical target for stratifying
    X_train, X_test, y_train, y_test = train_test_split(X, y_value, test_size=0.2, random_state=42, stratify=y_class) # Data split 80/20
   
    # Export for analysis
    np.save('fingerprints.npy', X)  # Save fingerprints as .npy
    metadata = df3.copy() 
    metadata.to_pickle('metadata.pkl')  # Save metadata as .pkl 
    with open("mols_and_bitinfo.pkl", "wb") as f: # Save bit info
        pickle.dump((mols, bit_infos), f)