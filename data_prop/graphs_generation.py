
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
covalent_radii = {
    1: 0.31, 6: 0.76, 8: 0.66, 7: 0.71,
    16: 1.05, 9: 0.57, 17: 1.02, 35: 1.20
}

atomic_number_to_symbol = {
    1: "H", 6: "C", 8: "O", 7: "N", 16: "S", 9: "F", 17: "Cl", 35: "Br"
}
# One-hot encoding map 
atom_list_all = ["H", "C", "O", "N", "S", "F", "Cl", "Br"]
atom_map_onehot = {
    a: [1 if i == j else 0 for j in range(len(atom_list_all))]
    for i, a in enumerate(atom_list_all)
}


def extract_coordinates_from_geometry(geometry_array):
    atomic_numbers = geometry_array[:, 0].astype(int)
    coordinates = geometry_array[:, 2:5].astype(float)
    return atomic_numbers, coordinates

# Single Graph Builder (geometry + forces)
def build_single_graph(geometry_array, forces_array, label, rxn_name, scale=1.2, global_feats=None):
    atomic_numbers = geometry_array[:, 0].astype(int)
    coordinates = geometry_array[:, 2:5].astype(float)
    n_atoms = len(atomic_numbers)

    x_features = []
    for i, atomic_num in enumerate(atomic_numbers):
        symbol = atomic_number_to_symbol.get(atomic_num, "H")
        onehot = atom_map_onehot.get(symbol, [0] * len(atom_map_onehot))
        fx, fy, fz = forces_array[i]
        x_features.append(onehot + [fx, fy, fz])

    x = torch.tensor(x_features, dtype=torch.float)

    edge_index = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = np.linalg.norm(coordinates[i] - coordinates[j])
            r1 = covalent_radii.get(atomic_numbers[i], 0.7)
            r2 = covalent_radii.get(atomic_numbers[j], 0.7)
            if d < scale * (r1 + r2):
                edge_index.append([i, j])
                edge_index.append([j, i])
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.float)
    pos = torch.tensor(coordinates, dtype=torch.float)

    # global features (reaction-level)
    u = torch.tensor(global_feats, dtype=torch.float) if global_feats is not None else None

    return Data(x=x, edge_index=edge_index, y=y, pos=pos, rxn_name=rxn_name, u=u)

#Building three graphs: Reactant, TS, Product
def build_reaction_graphs(entry, scale=1.2):
    rxn_name = entry['reaction']
    label = entry['early_or_late']  
    
    reactant_geom = entry['reactant_geom']
    ts_geom = entry['ts_geom']
    product_geom = entry['product_geom']
    
    reactant_forc = entry['reactant_forces']
    ts_forc = entry['ts_forces']
    product_forc = entry['product_forces']

    global_feats = torch.tensor([
        entry["dist_R_TS"],
        entry["dist_P_TS"],
        entry["reaction_fraction"],  
        entry["disp_RTS_max"],
        entry["disp_PTS_max"],
        entry["force_rmsd_R_TS"],
        entry["force_rmsd_P_TS"],
        entry["adv_coord_max_disp_R_TS"],
        entry["adv_coord_max_disp_P_TS"]
    ], dtype=torch.float)

    g_reactant = build_single_graph(reactant_geom, reactant_forc, label, rxn_name, scale, global_feats)
    g_ts       = build_single_graph(ts_geom, ts_forc, label, rxn_name, scale, global_feats)
    g_product  = build_single_graph(product_geom, product_forc, label, rxn_name, scale, global_feats)

    return g_reactant, g_ts, g_product

class ReactionGraphDataset(Dataset):
    def __init__(self, df, scale=1.2):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scale = scale
        self.graphs = []
        self._build_all_graphs()

    def _build_all_graphs(self):
        for idx, row in self.df.iterrows():
            try:
                rxn_name = row['reaction']
                label = row['early_or_late']  # or 'label'
                
                # Geometries and forces
                reactant_geom, ts_geom, product_geom = row['reactant_geom'], row['ts_geom'], row['product_geom']
                reactant_forc, ts_forc, product_forc = row['reactant_forces'], row['ts_forces'], row['product_forces']

                # Global features
                global_feats = torch.tensor([
                    #row["dist_R_TS"],
                    #row["dist_P_TS"],
                    #row["reaction_fraction"],  # skip to avoid leakage
                    #row["disp_RTS_max"],
                    #row["disp_PTS_max"],
                    #row["force_rmsd_R_TS"],
                    #row["force_rmsd_P_TS"],
                    #row["adv_coord_max_disp_R_TS"],
                    #row["adv_coord_max_disp_P_TS"]
                ], dtype=torch.float)

                # Build graphs for reactant, TS, product
                for geom, forc in zip([reactant_geom, ts_geom, product_geom],
                                      [reactant_forc, ts_forc, product_forc]):
                    g = build_single_graph(geom, forc, label, rxn_name, self.scale, global_feats)
                    self.graphs.append(g)
            except Exception as e:
                print(f"Error building graphs for {row['reaction']}: {e}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]





