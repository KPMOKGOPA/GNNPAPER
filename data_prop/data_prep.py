import h5py
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

def geom_distance_matrix(coords):
    """Compute Euclidean distance matrix from coordinates"""
    return cdist(coords, coords)

def matrix_difference(a, b):
    """RMSD between distance matrices"""
    return np.sqrt(np.mean((a - b)**2))

results = []

with h5py.File('all_logs.h5', 'r') as f:
    reactions = list(f.keys())
    for rxn in reactions:
        try:
            irc_energies = f[rxn]['irc_energies'][:]
            irc_rxcoords = f[rxn]['irc_rxcoords'][:]
            
            max_idx = int(np.argmax(irc_energies))
            max_ene = irc_energies[max_idx]
            
            geometries_group = f[rxn]['geometries']
            point_names = [name.decode() for name in f[rxn]['point_names'][:]]
            
            forces_group = f[rxn]['forces']
   
            # extract geometries
            reactant_geom = geometries_group[point_names[0]][:]
            product_geom  = geometries_group[point_names[-1]][:]
            ts_geom       = geometries_group[point_names[max_idx]][:]

            # extract corresponding forces
            reactant_forc = forces_group[point_names[0]][:]
            product_forc  = forces_group[point_names[-1]][:]
            ts_forc       = forces_group[point_names[max_idx]][:]

            # compute Euclidean distance matrices
            D_R  = geom_distance_matrix(reactant_geom)
            D_P  = geom_distance_matrix(product_geom)
            D_TS = geom_distance_matrix(ts_geom)
            
            # calculate distances between distance matrices
            dist_R_TS = matrix_difference(D_R, D_TS)
            dist_P_TS = matrix_difference(D_P, D_TS)
            
            # determine early/late TS
            reaction_fraction = dist_R_TS / (dist_R_TS + dist_P_TS + 1e-10)
            
            # Dummy variable: 0 for early, 1 for late
            dummy = 0 if reaction_fraction < 0.5 else 1
            
            # Store results including forces
            results.append([
                rxn,                    # reaction name
                dist_R_TS,              # distance from reactant to TS
                dist_P_TS,              # distance from product to TS  
                reaction_fraction,      # reaction coordinate fraction
                dummy,                  # 0=early, 1=late
                reactant_geom,          # reactant geometry
                ts_geom,                # transition state geometry
                product_geom,           # product geometry
                reactant_forc,          # reactant forces
                ts_forc,                # TS forces
                product_forc            # product forces
            ])
            
        except Exception as e:
            print(f"Error processing {rxn}: {e}")
            continue
results_array = np.array(results, dtype=object)

df_results = pd.DataFrame({
    'reaction': [r[0] for r in results],
    'dist_R_TS': [r[1] for r in results],
    'dist_P_TS': [r[2] for r in results],
    'reaction_fraction': [r[3] for r in results],
    'ts_type': [r[4] for r in results],  # 0=early, 1=late
    'reactant_geom': [r[5] for r in results],
    'ts_geom': [r[6] for r in results],
    'product_geom': [r[7] for r in results],
    'reactant_forc': [r[8] for r in results],
    'ts_forc': [r[9] for r in results],
    'product_forc': [r[10] for r in results]
})

print(f"Processed {len(results)} reactions")
print(f"Early TS (dummy=0): {len([r for r in results if r[4] == 0])}")
print(f"Late TS (dummy=1): {len([r for r in results if r[4] == 1])}")
