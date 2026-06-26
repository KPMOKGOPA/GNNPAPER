import os
import numpy as np
import h5py


#extracting ino from gaussian IRC log files
def parse_log_file(file_path):
    start_marker1 = "******** Start new reaction path calculation ********"
    stop_marker1 = "Calculation of FORWARD path complete"
    
    start_marker2 = "Beginning calculation of the REVERSE path"
    stop_marker2 = "Reaction path calculation complete."
    
    ts_start_marker = "GENERAL PARAMETERS:"
    ts_stop_marker = "******** Start new reaction path calculation ********"
    
    
    forward_data = {}
    reverse_data = {}
    ts_data = {}
    irc_energies = []
    irc_rxcoord = []

    def collect_path_data(lines, start_idx, stop_marker, is_reverse=False):
        """Collect data for one path (forward or reverse)"""
        path_data = {
            'geometries': {},
            'forces': {},
            'distance_matrices': {},
            'scf_energies': {},
            'delta_rc': {},
            'net_rc': {},
            'step_numbers': [],
            'irc_energy': {},  # Store matched IRC energies
            'irc_rxcoord': {}  # Store matched IRC reaction coordinates
        }
        
        i = start_idx
        step_counter = 0
        collecting = True
        
        while i < len(lines) and collecting:
            line = lines[i]
            line_strip = line.strip()
            
            # Check for stop marker
            if stop_marker in line:
                collecting = False
                break
            
            # Check for new point marker
            if 'Calculating another point on the path.' in line:
                step_counter += 1
                path_data['step_numbers'].append(step_counter)
                
                # Collect all data for this step
                j = i + 1
                while j < len(lines) and 'Calculating another point on the path.' not in lines[j] and stop_marker not in lines[j]:
                    current_line = lines[j]
                    
                    # Geometry
                    if 'Input orientation:' in current_line:
                        geom_lines = []
                        k = j + 5  # Skip 4 lines after "Input orientation:"
                        while k < len(lines) and '-----' not in lines[k]:
                            if lines[k].strip():
                                geom_lines.append(lines[k].strip())
                            k += 1
                        if geom_lines:
                            path_data['geometries'][step_counter] = geom_lines
                    
                    # Distance matrix
                    elif 'Distance matrix (angstroms)' in current_line:
                        matrix_lines = []
                        k = j + 1
                        while k < len(lines) and 'Symmetry turned off' not in lines[k]:
                            if lines[k].strip():
                                matrix_lines.append(lines[k].strip())
                            k += 1
                        if matrix_lines:
                            path_data['distance_matrices'][step_counter] = matrix_lines
                    
                    # Forces
                    elif "NMatT=" in current_line:
                        force_lines = []
                        k = j + 5  # Skip 4 lines after "NMatT="
                        while k < len(lines) and '-----' not in lines[k]:
                            if lines[k].strip():
                                force_lines.append(lines[k].strip())
                            k += 1
                        if force_lines:
                            path_data['forces'][step_counter] = force_lines
                    
                    # SCF Energy
                    elif 'SCF Done:' in current_line:
                        parts = current_line.split()
                        try:
                            idx = parts.index('=')
                            energy = float(parts[idx + 1])
                            path_data['scf_energies'][step_counter] = energy
                        except (ValueError, IndexError):
                            pass
                    
                    # Delta RC
                    elif 'CHANGE IN THE REACTION COORDINATE ' in current_line:
                        try:
                            delta = float(current_line.split('=')[-1])
                            path_data['delta_rc'][step_counter] = delta
                        except ValueError:
                            pass
                    
                    # Net RC
                    elif 'NET REACTION COORDINATE UP TO THIS POINT' in current_line:
                        try:
                            net = float(current_line.split('=')[-1])
                            if is_reverse:
                                net = -net  # Apply negative for reverse path
                            path_data['net_rc'][step_counter] = net
                        except ValueError:
                            pass
                    
                    j += 1
                
                i = j - 1  # Continue from where we left off
            
            i += 1
        
        return path_data, i

    def collect_ts_data(lines):
        """Collect transition state data"""
        ts_data = {
            'geometry': [],
            'forces': [],
            'distance_matrix': [],
            'scf_energy': None,
            'irc_energy': 0.0,  # Zero for TS
            'irc_rxcoord': 0.0  # Zero for TS
        }
        
        collecting = False
        
        for i, line in enumerate(lines):
            if ts_start_marker in line:
                collecting = True
                continue
                
            if ts_stop_marker in line and collecting:
                collecting = False
                break
                
            if collecting:
                # Geometry
                if 'Input orientation:' in line:
                    geom_lines = []
                    k = i + 5  # Skip 4 lines after "Input orientation:"
                    while k < len(lines) and '-----' not in lines[k]:
                        if lines[k].strip():
                            geom_lines.append(lines[k].strip())
                        k += 1
                    if geom_lines:
                        ts_data['geometry'] = geom_lines
                
                # Distance matrix
                elif 'Distance matrix (angstroms)' in line:
                    matrix_lines = []
                    k = i + 1
                    while k < len(lines) and 'Symmetry turned off' not in lines[k]:
                        if lines[k].strip():
                            matrix_lines.append(lines[k].strip())
                        k += 1
                    if matrix_lines:
                        ts_data['distance_matrix'] = matrix_lines
                
                # Forces
                elif "NMatT=" in line:
                    force_lines = []
                    k = i + 5  # Skip 4 lines after "NMatT="
                    while k < len(lines) and '-----' not in lines[k]:
                        if lines[k].strip():
                            force_lines.append(lines[k].strip())
                        k += 1
                    if force_lines:
                        ts_data['forces'] = force_lines
                
                # SCF Energy
                elif 'SCF Done:' in line:
                    parts = line.split()
                    try:
                        idx = parts.index('=')
                        energy = float(parts[idx + 1])
                        ts_data['scf_energy'] = energy
                    except (ValueError, IndexError):
                        pass
        
        return ts_data

    def collect_irc_data(lines):
        """Collect IRC energy and reaction coordinate data"""
        energies = []
        rxcoords = []
        
        readingirc = 0
        for line in lines:
            if readingirc == 3:
                continue
                
            if readingirc == 2:
                if "---" in line:
                    readingirc = 3
                    continue
                words = line.split()
                if len(words) >= 3:
                    try:
                        en = float(words[1])
                        rc = float(words[2])
                        energies.append(en)
                        rxcoords.append(rc)
                    except ValueError:
                        readingirc = 3

            if readingirc == 1:
                if "Energy" in line:
                    readingirc = 2

            if "Summary of reaction path following" in line:
                readingirc = 1
        
        return energies, rxcoords

    def match_irc_to_paths(forward_data, reverse_data, irc_energies, irc_rxcoord):
        """Match IRC data to forward and reverse paths based on net_rc values"""
        tolerance = 1e-6  # Tolerance for floating point comparison
        
        # Match IRC data to forward path
        for step in forward_data['step_numbers']:
            net_rc = forward_data['net_rc'].get(step)
            if net_rc is not None:
                for i, irc_rc in enumerate(irc_rxcoord):
                    if abs(net_rc - irc_rc) < tolerance:
                        forward_data['irc_energy'][step] = irc_energies[i]
                        forward_data['irc_rxcoord'][step] = irc_rxcoord[i]
                        break
        
        # Match IRC data to reverse path  
        for step in reverse_data['step_numbers']:
            net_rc = reverse_data['net_rc'].get(step)
            if net_rc is not None:
                for i, irc_rc in enumerate(irc_rxcoord):
                    if abs(net_rc - irc_rc) < tolerance:
                        reverse_data['irc_energy'][step] = irc_energies[i]
                        reverse_data['irc_rxcoord'][step] = irc_rxcoord[i]
                        break
        
        return forward_data, reverse_data

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Collect IRC data first
    irc_energies, irc_rxcoord = collect_irc_data(lines)

    # Process forward path (marker1)
    forward_data = {}
    for i, line in enumerate(lines):
        if start_marker1 in line:
            forward_data, end_idx = collect_path_data(lines, i + 1, stop_marker1, is_reverse=False)
            break

    # Process reverse path (marker2)
    reverse_data = {}
    for i, line in enumerate(lines):
        if start_marker2 in line:
            reverse_data, end_idx = collect_path_data(lines, i + 1, stop_marker2, is_reverse=True)
            break

    ts_data = collect_ts_data(lines)

    if forward_data and reverse_data:
        forward_data, reverse_data = match_irc_to_paths(forward_data, reverse_data, irc_energies, irc_rxcoord)

    combined_data = {
        'forward': forward_data,
        'reverse': reverse_data,
        'ts': ts_data,
        'irc_energies': irc_energies,
        'irc_rxcoord': irc_rxcoord
    }

    return combined_data

def organize_final_data(combined_data):
    final_data = {}
    data_points = []
    
    if combined_data['ts']['geometry']:
        data_points.append(('ts', 0.0, combined_data['ts']))
    for path_type in ['forward', 'reverse']:
        if path_type in combined_data and combined_data[path_type]:
            path_data = combined_data[path_type]
            
            for step in path_data.get('step_numbers', []):
                step_key = f"{path_type}_{step}"
                irc_rxcoord = path_data['irc_rxcoord'].get(step)
                if irc_rxcoord is not None:
                    data_point = {
                        'geometry': path_data['geometries'].get(step, []),
                        'forces': path_data['forces'].get(step, []),
                        'distance_matrix': path_data['distance_matrices'].get(step, []),
                        'scf_energy': path_data['scf_energies'].get(step, None),
                        'delta_rc': path_data['delta_rc'].get(step, 0.0),
                        'net_rc': path_data['net_rc'].get(step, 0.0),
                        'irc_energy': path_data['irc_energy'].get(step, None),
                        'irc_rxcoord': irc_rxcoord
                    }
                    data_points.append((step_key, irc_rxcoord, data_point))
    
    
    data_points.sort(key=lambda x: x[1])
    
    for key, rxcoord, data in data_points:
        final_data[key] = data
    
    final_data['remaining_irc'] = {
        'energies': combined_data['irc_energies'],
        'reaction_coordinates': combined_data['irc_rxcoord']
    }
    final_data['_ordered_keys'] = [key for key, _, _ in data_points]
    
    return final_data



def print_ordered_data(final_data):
    """Print the ordered data for verification"""
    if '_ordered_keys' in final_data:
        for key in final_data['_ordered_keys']:
            data = final_data[key]
            rxcoord = data.get('irc_rxcoord', 'N/A')
            energy = data.get('irc_energy', 'N/A')
            print(f"{key:15} | RC: {rxcoord:12.6f} | Energy: {energy:15.8f}")
    
    # print remaining unmatched IRC data
    if 'remaining_irc' in final_data and final_data['remaining_irc']['energies']:
        print("\nUnmatched IRC data points:")
        for i, (energy, rc) in enumerate(zip(final_data['remaining_irc']['energies'], 
                                           final_data['remaining_irc']['reaction_coordinates'])):
            print(f"IRC_{i:03d}         | RC: {rc:12.6f} | Energy: {energy:15.8f}")



def save_logs_to_h5_structured(log_dir, output_h5):
    """Store geometries, forces, and distance matrices as proper arrays in HDF5."""
    with h5py.File(output_h5, 'w') as h5f:
        for file_name in os.listdir(log_dir):
            if not file_name.endswith('.log'):
                continue

            file_path = os.path.join(log_dir, file_name)
            basename = os.path.splitext(file_name)[0]
            print(f"Processing {file_name}")

            combined_data = parse_log_file(file_path)
            final_data = organize_final_data(combined_data)

            grp = h5f.create_group(basename)

            if '_ordered_keys' not in final_data:
                continue

            ordered_keys = final_data['_ordered_keys']

            point_names, scf_energies, delta_rcs, net_rcs, irc_energies, irc_rxcoords = ([] for _ in range(6))
            geom_grp = grp.create_group('geometries')
            force_grp = grp.create_group('forces')
            dm_grp = grp.create_group('distance_matrices')

            for key in ordered_keys:
                if key not in final_data or key in ['remaining_irc', '_ordered_keys']:
                    continue

                point_data = final_data[key]
                point_names.append(key)

                #Geometry 
                geom_matrix = []
                for line in point_data.get('geometry', []):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            atomic_num = int(parts[1])
                            coords = [atomic_num] + [float(x) for x in parts[3:6]]
                            geom_matrix.append(coords)
                        except ValueError:
                            continue
                if geom_matrix:
                    geom_array = np.array(geom_matrix, dtype=np.float64)
                    geom_grp.create_dataset(key, data=geom_array)

                # Forces
                force_matrix = []
                for line in point_data.get('forces', []):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            fx, fy, fz = [float(x) for x in parts[2:5]]
                            force_matrix.append([fx, fy, fz])
                        except ValueError:
                            continue
                if force_matrix:
                    force_array = np.array(force_matrix, dtype=np.float64)
                    force_grp.create_dataset(key, data=force_array)

                # Distance Matrix
                dm_matrix = []
                for line in point_data.get('distance_matrix', []):
                    parts = line.split()
                    try:
                        row = [float(x) for x in parts if x.replace('.', '', 1).isdigit()]
                        if row:
                            dm_matrix.append(row)
                    except ValueError:
                        continue
                if dm_matrix:
                    n = len(dm_matrix)
                    dm_array = np.zeros((n, n), dtype=np.float64)
                    for i, row in enumerate(dm_matrix):
                        for j, val in enumerate(row):
                            dm_array[i, j] = val
                            dm_array[j, i] = val
                    dm_grp.create_dataset(key, data=dm_array)

                # Scalar values 
                scf_energies.append(point_data.get('scf_energy', np.nan))
                delta_rcs.append(point_data.get('delta_rc', np.nan))
                net_rcs.append(point_data.get('net_rc', np.nan))
                irc_energies.append(point_data.get('irc_energy', np.nan))
                irc_rxcoords.append(point_data.get('irc_rxcoord', np.nan))

            # Save scalar arrays
            grp.create_dataset('point_names', data=np.array(point_names, dtype='S'))
            grp.create_dataset('scf_energies', data=np.array(scf_energies, dtype=np.float64))
            grp.create_dataset('delta_rc', data=np.array(delta_rcs, dtype=np.float64))
            grp.create_dataset('net_rc', data=np.array(net_rcs, dtype=np.float64))
            grp.create_dataset('irc_energies', data=np.array(irc_energies, dtype=np.float64))
            grp.create_dataset('irc_rxcoords', data=np.array(irc_rxcoords, dtype=np.float64))

            # Unmatched IRC data
            if 'remaining_irc' in final_data:
                irc_data = final_data['remaining_irc']
                if irc_data.get('energies'):
                    grp.create_dataset('remaining_irc_energies',
                                       data=np.array(irc_data['energies'], dtype=np.float64))
                if irc_data.get('reaction_coordinates'):
                    grp.create_dataset('remaining_irc_rxcoords',
                                       data=np.array(irc_data['reaction_coordinates'], dtype=np.float64))

log_directory = '/home/kabelo/IRC/complete/'
output_file = '/home/kabelo/IRC/all_logs_structured.h5'
save_logs_to_h5_structured(log_directory, output_file)

