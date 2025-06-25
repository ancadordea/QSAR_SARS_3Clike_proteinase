import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
from rdkit.Chem.Draw import Chem
from sklearn.feature_selection import VarianceThreshold

def load_data(fingerprint_path='fingerprints.npy', metadata_path='metadata.pkl'):
    X = np.load(fingerprint_path)
    metadata = pd.read_pickle(metadata_path)
    y = metadata['pIC50_value']
    y_class = metadata['bioactivity_class']

    path= "mols_and_bitinfo.pkl"
    with open(path, "rb") as f:
        mols, bit_infos = pickle.load(f)

    return X, y, y_class, mols, bit_infos

#Random Forest Regressor
def split_data(X, y, y_class, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_class)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"R2: {r2:.3f}, RMSE: {rmse:.3f}")
    return y_pred, r2, rmse

def apply_variance_threshold(X_train, X_test, threshold=0.1):
    """Remove low-variance features"""
    selector = VarianceThreshold(threshold=threshold)
    X_train_processed = selector.fit_transform(X_train)
    X_test_processed = selector.transform(X_test)
    print(f"Removed {X_train.shape[1] - X_train_processed.shape[1]} low-variance features.")
    return X_train_processed, X_test_processed

def apply_rf_feature_selection(X_train, X_test, y_train, n_components=50):
    """Select top features by Random Forest importance WITHOUT SCALING"""
    # Remove scaling (not needed for Random Forest)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,  # Use GridSearch's best depth
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-n_components:][::-1]
    
    return X_train[:, top_indices], X_test[:, top_indices], top_indices

def plot_predictions(y_test, y_pred, save_path='Results/regression.png'):
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Observed pIC50'), plt.ylabel('Predicted pIC50')
    plt.title('Random Forest Regression Performance')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout(), plt.savefig(save_path)

def tune_random_forest(X_train, y_train, param_grid=None, cv=5): # ML improvement
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    Returns the best model and parameters.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_

# Chem analysis -- add Lipinski into analysis 
def top_fp_feature_importance_df(model, top_n=50):
    """Get top important bits"""
    importances = model.feature_importances_
    top_indices = importances.argsort()[-top_n:][::-1]
    # print(f"Top {top_n} fingerprint bits by importance:")
    # for i in top_indices:
    #     print(f"Bit {i}: importance {importances[i]:.4f}")

    # Create DataFrame
    df = pd.DataFrame({
        'bit': top_indices,
        'importance': importances[top_indices]
    })
    # print(f"Top {top_n} fingerprint bits by importance:")
    # print(df)
    return df

def get_substructure_from_bit(mol, bit_info, bit_id):
    """
    Extract the substructure molecule that triggered a fingerprint bit.
    bit_info[bit_id] is a list of (atom_id, radius) tuples.
    We take the first tuple to get the subgraph around atom_id with radius.
    """
    if bit_id not in bit_info:
        return None

    # Take first atom_id and radius tuple that triggered this bit
    atom_id, radius = bit_info[bit_id][0]

    # Extract subgraph atoms within radius bonds from atom_id
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
    atoms_in_env = set()
    for bond_id in env:
        bond = mol.GetBondWithIdx(bond_id)
        atoms_in_env.add(bond.GetBeginAtomIdx())
        atoms_in_env.add(bond.GetEndAtomIdx())
    
    # Also include the central atom
    atoms_in_env.add(atom_id)
    
    # Create submol
    submol = Chem.PathToSubmol(mol, env)
    return submol

# Define some common functional groups SMARTS patterns
functional_groups = {
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "phenyl": "c1ccccc1",
    "carboxylic acid": "C(=O)[OH]",
    "hydroxyl": "[OX2H]",
    "ketone": "C(=O)[#6]",
    "ether": "[$([OX2]([#6])[#6])]",
    "alkene": "C=C",
    "alkyne": "C#C",
    "amide": "C(=O)N",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ester": "C(=O)O[#6]",
    "nitro": "[NX3](=O)=O",
    "halide": "[F,Cl,Br,I]",
    "thiol": "[#16X2H]",
    "thioether": "[#16X2]([#6])[#6]",
    "phenol": "c1ccccc1O",
    "carbamate": "N[C](=O)O",
    "sulfone": "S(=O)(=O)[#6]",
    "sulfoxide": "S(=O)[#6]",
    "imine": "C=N",
    "cyanide": "[C-]#[N+]",
    "phosphate": "P(=O)(O)(O)",
    "fluoroalkane": "[CX4][F]",
    "chloralkane": "[CX4][Cl]",
    "bromoalkane": "[CX4][Br]",
    "iodoalkane": "[CX4][I]",
}

def describe_functional_groups(submol):
    descriptions = []
    for name, smarts in functional_groups.items():
        patt = Chem.MolFromSmarts(smarts)
        if submol.HasSubstructMatch(patt):
            descriptions.append(name)
    return descriptions

def describe_bit_substructure(mol, bit_info, bit_id):
    submol = get_substructure_from_bit(mol, bit_info, bit_id)
    if submol is None:
        return f"Bit {bit_id}: no substructure found"

    groups = describe_functional_groups(submol)
    if not groups:
        return f"Bit {bit_id}: substructure detected but no common functional groups matched"
    else:
        return f"Bit {bit_id}: contains {' and '.join(groups)}"
    
def get_bit_functional_group_descriptions(mols, bit_infos, bit_ids):
    """
    For each bit_id in bit_ids and each molecule,
    describe the functional groups triggered by that bit.
    
    Returns a DataFrame with columns: ['mol_index', 'bit_id', 'description']
    """
    records = []
    for bit_id in bit_ids:
        for mol_idx, mol in enumerate(mols):
            desc = describe_bit_substructure(mol, bit_infos[mol_idx], bit_id)
            records.append({
                'mol_index': mol_idx,
                'bit_id': bit_id,
                'description': desc
            })
    return pd.DataFrame(records)

def summarize_functional_groups(bit_desc_df):
    """
    Given the DataFrame from get_bit_functional_group_descriptions,
    summarise counts of each functional group phrase per bit.
    
    Returns a summary DataFrame with 'bit_id', 'functional_group', 'count'.
    """
    from collections import Counter
    summaries = []

    for bit_id, group_df in bit_desc_df.groupby('bit_id'):
        # Extract all descriptions for this bit
        all_descriptions = group_df['description'].tolist()

        # Parse functional groups from description text, ignore 'no substructure' or similar
        all_groups = []
        for desc in all_descriptions:
            if 'no substructure found' in desc or 'no common functional groups matched' in desc:
                continue
            # The functional groups part is after the colon ": contains ..."
            if ':' in desc:
                parts = desc.split(':')
                groups_part = parts[1].strip() if len(parts) > 1 else ''
                # Functional groups separated by "and"
                groups = [g.strip() for g in groups_part.replace('contains ', '').split(' and ')]
                all_groups.extend(groups)

        # Count frequency of each functional group for this bit
        counter = Counter(all_groups)
        for group, count in counter.items():
            summaries.append({'bit_id': bit_id, 'functional_group': group, 'count': count})

    summary_df = pd.DataFrame(summaries)
    return summary_df

if __name__ == "__main__":
    X, y, y_class, mols, bit_infos = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, y_class)
    X_train_var, X_test_var = apply_variance_threshold(X_train, X_test, threshold=0.1)
    X_train_processed, X_test_processed, top_indices = apply_rf_feature_selection(
        X_train_var, X_test_var, y_train, n_components=50
    )

    # Train model on processed features
    rf_model = tune_random_forest(X_train_processed, y_train)
    y_pred, r2, rmse = evaluate_model(rf_model, X_test_processed, y_test)
    plot_predictions(y_test, y_pred)

    # For bit analysis, map back to original bit indices
    # Get the important bits after all feature selection steps
    top_bits_df = top_fp_feature_importance_df(rf_model, top_n=50)

    # Get descriptions for top bits across molecules
    bit_desc_df = get_bit_functional_group_descriptions(mols, bit_infos, top_bits_df['bit'])
    bit_desc_df.to_csv('bit_functional_group_descriptions.csv', index=False)

    # Summarize functional groups per bit
    summary_df = summarize_functional_groups(bit_desc_df)
    summary_df.to_csv('functional_group_summary_per_bit.csv', index=False)