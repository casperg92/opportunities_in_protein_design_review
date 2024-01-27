import os
import pydssp
import numpy as np
import random
random.seed(52)

from Bio.PDB import PDBIO, Select, Polypeptide
from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch
from Bio.PDB import is_aa

# PyRosetta dependencies
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, hbonds
from pyrosetta.rosetta.protocols.scoring import Interface
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector
from pyrosetta.rosetta.core.chemical import aa_ala, aa_cys, aa_phe, aa_gly, aa_ile, aa_leu, aa_met, aa_pro, aa_val, aa_trp


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return 1
        else:
            return 0
        
    def accept_residue(self, residue):
        return Polypeptide.is_aa(residue)

def write_chain_to_pdb(structure, chain_id, output_file):
    """
    Write a specific chain from a BioPython Structure object to a PDB file.

    :param structure: BioPython Structure object
    :param chain_id: str, the target chain ID
    :param output_file: str, the output file name
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, ChainSelect(chain_id))

def pdb_to_string(pdb_file, chains=None, models=[1]):
  '''read pdb file and return as string'''

  MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
            'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
            'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
            'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
            'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
            '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
            'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
            'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
            'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
            'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
            'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
            'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
            'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
            'CSD':'CYS','SEC':'CYS'}
  restype_1to3 = {'A': 'ALA','R': 'ARG','N': 'ASN',
                  'D': 'ASP','C': 'CYS','Q': 'GLN',
                  'E': 'GLU','G': 'GLY','H': 'HIS',
                  'I': 'ILE','L': 'LEU','K': 'LYS',
                  'M': 'MET','F': 'PHE','P': 'PRO',
                  'S': 'SER','T': 'THR','W': 'TRP',
                  'Y': 'TYR','V': 'VAL'}

  restype_3to1 = {v: k for k, v in restype_1to3.items()}

  if chains is not None:
    if "," in chains: chains = chains.split(",")
    if not isinstance(chains,list): chains = [chains]
  if models is not None:
    if not isinstance(models,list): models = [models]

  modres = {**MODRES}
  lines = []
  seen = []
  model = 1
  for line in open(pdb_file,"rb"):
    line = line.decode("utf-8","ignore").rstrip()
    if line[:5] == "MODEL":
      model = int(line[5:])
    if models is None or model in models:
      if line[:6] == "MODRES":
        k = line[12:15]
        v = line[24:27]
        if k not in modres and v in restype_3to1:
          modres[k] = v
      if line[:6] == "HETATM":
        k = line[17:20]
        if k in modres:
          line = "ATOM  "+line[6:17]+modres[k]+line[20:]
      if line[:4] == "ATOM":
        chain = line[21:22]
        if chains is None or chain in chains:
          atom = line[12:12+4].strip()
          resi = line[17:17+3]
          resn = line[22:22+5].strip()
          if resn[-1].isalpha(): # alternative atom
            resn = resn[:-1]
            line = line[:26]+" "+line[27:]
          key = f"{model}_{chain}_{resn}_{atom}"
          if key not in seen: # skip alternative placements
            lines.append(line)
            seen.append(key)
      if line[:5] == "MODEL" or line[:3] == "TER" or line[:6] == "ENDMDL":
        lines.append(line)
  return "\n".join(lines)

def secondary_structure_ranges(array, offset=0):
    ranges = {}
    start = None
    prev_elem = None
    
    for i, elem in enumerate(array):
        if elem != '-' and elem != prev_elem:
            if start is not None and prev_elem is not None and prev_elem != '-':
                if prev_elem in ranges:
                    ranges[prev_elem].append([start + offset, i - 1 + offset])
                else:
                    ranges[prev_elem] = [[start + offset, i - 1 + offset]]
            start = i
        elif elem == '-' and prev_elem != '-':
            if prev_elem is not None and prev_elem in ranges:
                ranges[prev_elem].append([start + offset, i - 1 + offset])
            elif prev_elem is not None:
                ranges[prev_elem] = [[start + offset, i - 1 + offset]]
            start = None
        
        prev_elem = elem
        
    if start is not None and prev_elem is not None and prev_elem != '-':
        if prev_elem in ranges:
            ranges[prev_elem].append([start + offset, len(array) - 1 + offset])
        else:
            ranges[prev_elem] = [[start + offset, len(array) - 1 + offset]]
    
    return ranges

def get_chain_length(chain):
     return sum(is_aa(residue) for residue in chain)

def get_longest_shortest_chain(structure):
    chain_lengths = {}
    chains = {}
    for chain in structure.get_chains():
        chain_length = get_chain_length(chain)
        if chain_length < 5:
            continue
        chain_lengths[chain.id] = chain_length
        chains[chain.id] = chain
    shortest_chain_id = min(chain_lengths, key=chain_lengths.get)
    longest_chain_id = max(chain_lengths, key=chain_lengths.get)

    # if longest chain is 10 residues longer than shortest, print it
    if chain_lengths[longest_chain_id] - chain_lengths[shortest_chain_id] > 10:
        print(f"Longest chain {longest_chain_id} is {chain_lengths[longest_chain_id]}, shortest chain {shortest_chain_id} is {chain_lengths[shortest_chain_id]}")

    return chains[shortest_chain_id], chains[longest_chain_id]

def calculate_rc_threshold(range1, range2, model):
    try:
        atoms1 = [atom for res in model.get_residues() if res.id[1] in range(range1[0], range1[1] + 1) for atom in res.get_atoms()]
        atoms2 = [atom for res in model.get_residues() if res.id[1] in range(range2[0], range2[1] + 1) for atom in res.get_atoms()]

        ns = NeighborSearch(atoms1)
        min_distance = float('inf')
        
        for atom in atoms2:
            nearby_atoms = ns.search(atom.coord, 8.0)
            if nearby_atoms:
                for atom1 in nearby_atoms:
                    distance = atom1 - atom
                    if distance < min_distance:
                        min_distance = distance
                        
        return min_distance
    except IndexError:
        return float('inf')

def get_sse_ranges_from_pdb(pdb_file, target_chain_id):
    sse_ranges = {'H': [], 'E': []}
    sse_count = {'H': 0, 'E': 0}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("HELIX") and line[19] == target_chain_id:
                start = int(line[21:25].strip())
                end = int(line[33:37].strip())
                sse_ranges['H'].append([start, end])
                sse_count['H'] += 1
            elif line.startswith("SHEET") and line[21] == target_chain_id:
                start = int(line[22:26].strip())
                end = int(line[33:37].strip())
                sse_ranges['E'].append([start, end])
                sse_count['E'] += 1
            
    #if list is empty, remove the key
    for key in list(sse_ranges):
        if not sse_ranges[key]:
            del sse_ranges[key]
    return sse_ranges, sse_count

pdb_string = ''

def get_sse_from_structure(model, target_chain, pdb_file):
    pdb_file = os.path.basename(pdb_file)
    new_pdb_file = f"denovo_pdbs/tmp/{pdb_file}"
    write_chain_to_pdb(model, target_chain.id, new_pdb_file)
    return get_sse_from_pdb(model, target_chain, new_pdb_file)

def get_sse_from_pdb(model, target_chain, pdb_file):
    min_res_id = min([residue.id[1] for residue in target_chain.get_residues() if is_aa(residue)])
    max_res_id = max([residue.id[1] for residue in target_chain.get_residues() if is_aa(residue)])
    global pdb_string
    pdb_string = pdb_to_string(pdb_file)
    coord = pydssp.read_pdbtext(pdb_string)
    ss = pydssp.assign(coord)
    ss_ranges = secondary_structure_ranges(ss, offset=min_res_id)
    ss_ranges = {key: [value for value in ss_ranges[key] if value[0] >= min_res_id and value[1] <= max_res_id] for key in ss_ranges}

    #dlog(ss, ss_ranges, min_res_id, v=1)

    # filter out ranges which are too short
    ss_ranges = {key: [value for value in ss_ranges[key] if value[1] - value[0] >= 2] for key in ss_ranges}
    # remove empty keys
    ss_ranges = {key: ss_ranges[key] for key in ss_ranges if ss_ranges[key]}
    return ss_ranges, ss

def calculate_relative_contact_order(pdb_file, distance_cutoff=8.0, min_seq_separation=4, verbose=False, sse_from_pdb=False, shortest=True):
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    # Get the longest chain
    shortest_chain, longest_chain = get_longest_shortest_chain(model)
    target_chain = shortest_chain if shortest else longest_chain
    if sse_from_pdb:
        # Extract secondary structure elements (SSEs) from the PDB file
        sse_ranges, ss = get_sse_ranges_from_pdb(pdb_file, target_chain.id)
    else:
        try:
            sse_ranges, ss = get_sse_from_structure(model, target_chain, pdb_file)
        except AssertionError:
            sse_ranges, ss = get_sse_ranges_from_pdb(pdb_file, target_chain.id)
            print(sse_ranges)

    # Calculate pairwise distances and sequence separations
    contact_count = 0
    seq_separation_sum = 0
    for ss1, ranges1 in sse_ranges.items():
        for ss2, ranges2 in sse_ranges.items():
            for r1 in ranges1:
                for r2 in ranges2:
                    # skip equal SSEs
                    if ss1 == ss2 and r1 == r2:
                        continue
                    # skip if r1 is after r2
                    if r1[0] > r2[0]:
                        continue
                    if verbose:
                        print(f"Checking distance separation for {ss1} {r1} and {ss2} {r2}")
                    seq_separation = abs(r2[0] - r1[1])
                    if verbose:
                        print("Sequence separation: ", seq_separation)
                    if seq_separation >= min_seq_separation:
                        if verbose:
                            print(f"Passed, checking {ss1} {r1} and {ss2} {r2}")
                        rc_threshold = calculate_rc_threshold(r1, r2, target_chain)
                        if verbose:
                            print("Distance: ", rc_threshold)
                        if rc_threshold <= distance_cutoff:
                            if verbose:
                                print(f"Passed, adding contact {seq_separation}")
                            contact_count += 1
                            seq_separation_sum += seq_separation

    # Compute the relative contact order
    rco = seq_separation_sum / contact_count if contact_count > 0 else 0
    if verbose:
        print(f"Seq separation sum: {seq_separation_sum}, contact count: {contact_count}, RCO: {rco}")
    target_chain_length = get_chain_length(target_chain)
    protein_class = ''.join(sorted(sse_ranges.keys()) )
    return rco, contact_count, target_chain_length, protein_class

def obtain_sse_content(pdb_file, chain = None, shortest=True):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    # Get the longest chain
    if chain == None:
        shortest_chain, longest_chain = get_longest_shortest_chain(model)
        target_chain = shortest_chain if shortest else longest_chain
    else:
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]

        for x in structure.get_chains():
            if x.id == chain:
                target_chain = x
        
    pdb_file = os.path.basename(pdb_file)
    new_pdb_file = f"denovo_pdbs/tmp/{pdb_file}"
    write_chain_to_pdb(model, target_chain.id, new_pdb_file)

    pdb_string = pdb_to_string(new_pdb_file)
    coord = pydssp.read_pdbtext(pdb_string)
    ss = pydssp.assign(coord)

    helix_count = 0
    sheet_count = 0
    loop_count = 0

    for element in ss:
        if element == 'H':
            helix_count += 1
        elif element == 'E':
            sheet_count += 1
        else:
            loop_count += 1

    total_count = helix_count + sheet_count + loop_count

    return total_count, helix_count, sheet_count, loop_count

def score_interface(pdb_file, binder_chain="B"):
    # load pose
    pose = pr.pose_from_pdb(pdb_file)

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    #iam.set_compute_interface_delta_hbond_unsat(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    
    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    

    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)
    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # shape complementarity
    interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds
    interface_dG = iam.get_interface_dG() # interface dG
    interface_dSASA = iam.get_interface_delta_sasa() # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat() # interface pack stat score
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100 # ratio of dG/dSASA (normalised energy for interface area size)
    #interface_delta_unsat_hbonds = interfacescore.delta_unsat_hbonds # number of unsaturated/unsatisfied hydrogen bonds at the interface
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100 # Hbonds per interface size percentage
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100 # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate surface hydrophobicity
    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(pose)

    exp_apol_count = 0
    total_count = 0 
    
    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count/total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
    'binder_score': binder_score,
    'surface_hydrophobicity': surface_hydrophobicity,
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    'interface_hbond_percentage': interface_hbond_percentage,
    'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # round to two decimal places
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    return interface_scores, interface_AA, interface_residues_pdb_ids_str

def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("seed", trajectory_pdb)

    # Get the specified chain
    chain = structure[0][binder_chain]

    # Find interacting residues from other chains based on atom distance cutoff
    interacting_residues = {}
    for residue in chain:
        for atom in residue:
            for other_chain in structure[0]:
                if other_chain.id != binder_chain:
                    for other_residue in other_chain:
                        for other_atom in other_residue:
                            if atom - other_atom < atom_distance_cutoff:
                                # Convert three-letter amino acid code to single-letter code
                                aa_single_letter = Polypeptide.three_to_one(residue.get_resname())
                                interacting_residues[residue.id[1]] = aa_single_letter

    return interacting_residues

def obtain_sse_content_interface(pdb_file, chain = None):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    res_coords = {}
    res_nr = {}

    for x in structure.get_chains():
        # first we want to obtain the coordinates for all of the chains
        current_chain = x.get_id()
        res_coords[current_chain] = []
        res_nr[current_chain] = []
        for res in x.get_residues():
            for atom in res.get_atoms():
                res_coords[current_chain].append(atom.get_coord())
                res_nr[current_chain].append(res.get_id()[1])
                # print(res.get_id()[1], atom.get_coord())
        if current_chain == chain:
            target_chain = x


    pdb_file = os.path.basename(pdb_file)
    new_pdb_file = f"denovo_pdbs/tmp/{pdb_file}"
    write_chain_to_pdb(model, target_chain.id, new_pdb_file)

    pdb_string = pdb_to_string(new_pdb_file)
    coord = pydssp.read_pdbtext(pdb_string)
    ss = pydssp.assign(coord)

    interacting_residues = []
    all_residues = []

    # compare the distance of the atoms between the binder and 
    key_B = chain
    for key_A in res_coords.keys():
        if key_A != key_B:
            for i in range(len(res_coords[key_A])):
                coord_A = res_coords[key_A][i]
                for j in range(len(res_coords[key_B])):
                    coord_B = res_coords[key_B][j]
                    distance = np.linalg.norm(coord_A-coord_B)
                    if str(res_nr[key_B][j]) not in all_residues:
                        all_residues.append(str(res_nr[key_B][j]))
                    if distance < 3.5:
                        # print('chain: ', key_A, 'res: ', res_nr[key_A][i], 'interacts with chain: ', key_B, 'res: ', res_nr[key_B][j])
                        if str(res_nr[key_B][j]) not in interacting_residues:
                            interacting_residues.append(str(res_nr[key_B][j]))

    # pop interacting residues from all residues
    res_interaction = []
    helix_count = 0
    sheet_count = 0
    loop_count = 0

    for res in all_residues:
        if res in interacting_residues:
            res_interaction.append(True)
        else:
            res_interaction.append(False)

    for i in range(len(res_interaction)):
        if res_interaction[i]:
            if ss[i] == 'H':
                helix_count += 1
            elif ss[i] == 'E':
                sheet_count += 1
            else:
                loop_count += 1

    total_count = helix_count + sheet_count + loop_count
    return total_count, helix_count, sheet_count, loop_count

def count_interface_bb_hbonds(pose, jump_number=1):
    """
    Count the number of backbone hydrogen bonds between two protein chains at an interface.
    
    Args:
        pose (Pose): A PyRosetta Pose object.
        jump_number (int): The jump number defining the interface. 
                           Typically 1 for a simple two-body complex.
    
    Returns:
        int: The number of inter-chain backbone hydrogen bonds at the interface.
    """
    
    # Create an Interface object for the given jump number
    interface = Interface(jump_number)
    interface.calculate(pose)
    
    # Compute all hydrogen bonds in the pose
    hbond_set = hbonds.HBondSet(pose, False)  # False for no "calc deriv"
    
    # Filter for hydrogen bonds at the interface and on the backbone
    count = 0
    for i in range(1, hbond_set.nhbonds() + 1):
        hbond = hbond_set.hbond(i)
        donor_res_num = hbond.don_res()
        acceptor_res_num = hbond.acc_res()
        
        # Check for inter-chain hydrogen bonds at the interface
        if interface.is_interface(donor_res_num) and interface.is_interface(acceptor_res_num) and pose.chain(donor_res_num) != pose.chain(acceptor_res_num):
            donor_atom = pose.residue(donor_res_num).atom_name(hbond.don_hatm()).strip()
            acceptor_atom = pose.residue(acceptor_res_num).atom_name(hbond.acc_atm()).strip()
            
            # Check if both atoms are backbone atoms
            if donor_atom in ['N', 'O', 'H', 'HA', 'HA2', 'HA3'] and acceptor_atom in ['N', 'O', 'H', 'HA', 'HA2', 'HA3']:
                count += 1
                
    return count

def get_interface_residues(pose, chain1, chain2, distance=8.0):
    # Select chains
    chain1_selector = ChainSelector(chain1)
    chain2_selector = ChainSelector(chain2)
    
    # Get neighborhood residues around chain2 for chain1 and vice versa
    chain1_interface = NeighborhoodResidueSelector(chain2_selector, distance, True)
    chain2_interface = NeighborhoodResidueSelector(chain1_selector, distance, True)
    
    return chain1_interface.apply(pose), chain2_interface.apply(pose)

def calculate_hydrophobicity(pose, interface_residues):
    hydrophobic_aas = [aa_ala, aa_cys, aa_phe, aa_gly, aa_ile, aa_leu, aa_met, aa_pro, aa_val, aa_trp]
    
    hydrophobic_count = sum(1 for i, residue in enumerate(interface_residues) if residue and pose.residue(i + 1).aa() in hydrophobic_aas)
    total_count = sum(1 for residue in interface_residues if residue)
    
    return hydrophobic_count / total_count if total_count else 0