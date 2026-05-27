import time
import streamlit as st
import subprocess
import os
import urllib.request
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from PIL import Image

# Attempt to import pyzbar for QR scanning
try:
    from pyzbar.pyzbar import decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False


# --- CLOUD CONTEXT ENGINE MANAGEMENT ---

def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")

ensure_linux_vina_exists()


# --- PUBCHEM AUTOMATED DATA CONVERTER ---

def fetch_ligand_data_from_pubchem(smiles_string):
    """Queries NCBI PubChem REST API to dynamically fetch validated small molecule attributes."""
    metadata = {"name": "Unknown Compound Name", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical Derivative")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception:
        pass 
    return metadata


# --- PDB METADATA & HETATM CO-CRYSTAL PARSER ---

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "name": "Unknown Protein",
        "title": "Uploaded Protein Structure Matrix", 
        "id": pdb_id.upper() if pdb_id and pdb_id != "Uploaded File" else "Unknown",
        "class": "Unknown Classification", "organism": "Unknown",
        "system": "Unknown Expression System", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path): return meta
    
    with open(file_path, "r") as f:
        title_parts = []
        for line in f:
            if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
            elif line.startswith("HEADER"): 
                meta["class"] = line[10:50].strip().title()
                if len(line) >= 66:
                    possible_id = line[62:66].strip()
                    if len(possible_id) == 4:
                        meta["id"] = possible_id.upper()
            elif line.startswith("COMPND"):
                if "MOLECULE:" in line:
                    mol_name = line.split("MOLECULE:")[1].split(";")[0].strip()
                    if meta["name"] == "Unknown Protein":
                        meta["name"] = mol_name.title()
            elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
            elif "EXPRESSION_SYSTEM" in line: meta["system"] = line.split(":")[-1].replace(";","").strip()
            elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
            elif "RESOLUTION." in line and "ANGSTROMS." in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match: meta["res"] = f"{match.group(1)} Å"
                
    if title_parts: meta["title"] = " ".join(title_parts).title()
    
    if meta["name"] == "Unknown Protein" and meta["title"] != "Uploaded Protein Structure Matrix":
        meta["name"] = meta["title"]
        
    return meta

def parse_bound_ligands(file_path):
    ligands = {}
    if not os.path.exists(file_path): return ligands
    
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip() if line[21].strip() else "A"
                try: res_seq = int(line[22:26].strip())
                except ValueError: continue
                if res_name in ["HOH", "WAT", "DOD"]: continue
                
                key = f"{res_name}-{chain_id}-{res_seq}"
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError: continue
                
                if key not in ligands:
                    ligands[key] = {"res": res_name, "chain": chain_id, "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))
                
    processed_ligands = []
    for key, info in ligands.items():
        pts = info["coords"]
        n_atoms = len(pts)
        if n_atoms < 4: continue
        
        cx, cy, cz = sum([p[0] for p in pts])/n_atoms, sum([p[1] for p in pts])/n_atoms, sum([p[2] for p in pts])/n_atoms
        bx = max([p[0] for p in pts]) - min([p[0] for p in pts]) + 10.0
        by = max([p[1] for p in pts]) - min([p[1] for p in pts]) + 10.0
        bz = max([p[2] for p in pts]) - min([p[2] for p in pts]) + 10.0
        
        processed_ligands.append({
            "ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms,
            "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2),
            "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)
        })
    return processed_ligands

def compute_protein_centroid(pdbqt_file):
    """Calculates the geometric center and dimensions for a full-protein grid box."""
    coords = []
    if not os.path.exists(pdbqt_file): 
        return 0.0, 0.0, 0.0, 20.0, 20.0, 20.0
    
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    
    if not coords: return 0.0, 0.0, 0.0, 20.0, 20.0, 20.0
    
    arr = np.array(coords)
    center = np.mean(arr, axis=0)
    dims = np.max(arr, axis=0) - np.min(arr, axis=0) + 10.0 
    return center[0], center[1], center[2], dims[0], dims[1], dims[2]

# --- ADVANCED BIOPHYSICAL INTERACTION PARSER ENGINE ---

def parse_pdbqt_coordinates(pdbqt_string):
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip().upper()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    
    with open(receptor_file, "r") as f:
         receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]:
                    b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C" and any(aro in r_at["res"] for aro in ["PHE", "TYR", "TRP"])):
                    b_type = "pi-Stacking / Hydrophobic"
                else:
                    b_type = "van der Waals Contact"
                    
                seen.add(res_id)
                interactions.append({
                    "Residue Contact": res_id,
                    "Interaction Type": b_type,
                    "Distance (Å)": round(dist, 2),
                    "r_coord": r_at["coord"].tolist(),
                    "l_coord": l_at["coord"].tolist()
                })
    return interactions


# --- BIOINFORMATICS STRUCTURAL CONVERTERS ---

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    autodock_type_map = {
        "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", 
        "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", 
        "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg"
    }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception: torsions = 4
    try:
        with open(input_pdb, "r") as pdb, open(output_pdbqt, "w") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip() if line[21].strip() else "A"
                    try: res_seq = int(line[22:26].strip())
                    except ValueError: res_seq = 1
                    try: x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    except ValueError: continue
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"
                    pdbqt.write(f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vina_type:<2}\n")
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        return True, output_pdbqt
    except Exception as e: return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return True, output_filename
    except Exception as e: return False, str(e)


# --- LOG FILE PARSERS ---

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def parse_vina_output_with_residues(stdout_text):
    data = []
    pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
    poses_dict = split_docking_poses("docking_poses.pdbqt")
    if not stdout_text: return pd.DataFrame(data)
    for line in stdout_text.split("\n"):
        match = pattern.match(line)
        if match:
            mode_idx = int(match.group(1))
            res_string, bond_types = "N/A", "N/A"
            if mode_idx in poses_dict:
                ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mode_idx])
                if ints:
                    res_string = ", ".join(list(set([i["Residue Contact"] for i in ints])))
                    bond_types = ", ".join(list(set([i["Interaction Type"] for i in ints])))
            data.append({"Binding Mode": mode_idx, "Affinity (kcal/mol)": float(match.group(2)), "RMSD l.b.": float(match.group(3)), "RMSD u.b.": float(match.group(4)), "Interacting Residues": res_string, "Contact Bond Types": bond_types})
    return pd.DataFrame(data)

def build_styled_html_table(df):
    """Custom function to inject red CSS styling for positive affinity scores in HTML Export"""
    html = '<table class="data-table"><thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            val = row[col]
            if col == 'Affinity (kcal/mol)':
                try:
                    if float(val) > 0:
                        html += f'<td style="color: #c62828; font-weight: bold;">{val}</td>'
                    else:
                        html += f'<td style="color: #1b5e20;">{val}</td>'
                except:
                    html += f'<td>{val}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

# --- HIGH PERFORMANCE VISUALIZATION CONSTRUCTS ---

def generate_2d_ligand_img(mol):
    if mol is None: return None
    try:
        mol_flat = Chem.Mol(mol)
        Chem.SanitizeMol(mol_flat)
        AllChem.Compute2DCoords(mol_flat)
        img = Draw.MolToImage(mol_flat, size=(340, 260))
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception: return None

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[]):
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc = interact["r_coord"]
        lc = interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer.addLabel("{interact['Residue Contact']} ({interact['Distance (Å)']}A)", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:11}});
        """

    html_content = f"""
    <div id="wrapper_div" style="position:relative; width:100%;">
        <button onclick="toggleFullScreen()" style="position:absolute; top:12px; right:12px; z-index:9999; padding:6px 12px; background:#007bff; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; font-family:sans-serif; box-shadow:0 2px 4px rgba(0,0,0,0.15);">🖥 Fullscreen View</button>
        <div id="container" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{
                viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
            }} else if ('{mode}' === 'spacefill') {{
                viewer.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}});
            }} else {{
                viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}});
            }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer.zoomTo(); viewer.render();
        function toggleFullScreen() {{
            let elem = document.getElementById("wrapper_div");
            if (!document.fullscreenElement) {{ elem.requestFullscreen(); document.getElementById("container").style.height = "90vh"; }}
            else {{ document.exitFullscreen(); document.getElementById("container").style.height = "480px"; }}
        }}
        document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) document.getElementById("container").style.height = "480px"; }});
    </script>
    """
    components.html(html_content, height=510)


# --- APPLICATION DASHBOARD WORKSPACE ---

st.set_page_config(page_title="In Silico Docking Hub", layout="wide")
st.title("🔬 InSilico BioSphere - Docking")
st.markdown("""
**InSilico BioSphere** | Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangresearch@gmail.com
""")

# Initialize states safely
if "protein_name" not in st.session_state: st.session_state.protein_name = "Unknown Protein"
if "pdb_id_display" not in st.session_state: st.session_state.pdb_id_display = "Custom"
if "cx" not in st.session_state: st.session_state.cx = 0.0
if "cy" not in st.session_state: st.session_state.cy = 0.0
if "cz" not in st.session_state: st.session_state.cz = 0.0
if "sx" not in st.session_state: st.session_state.sx = 20
if "sy" not in st.session_state: st.session_state.sy = 20
if "sz" not in st.session_state: st.session_state.sz = 20
if "target_ready" not in st.session_state: st.session_state.target_ready = False
if "ligand_ready" not in st.session_state: st.session_state.ligand_ready = False
if "local_target_path" not in st.session_state: st.session_state.local_target_path = None
if "docking_results_raw" not in st.session_state: st.session_state.docking_results_raw = None
if "serialized_ligand_block" not in st.session_state: st.session_state.serialized_ligand_block = None
if "ligand_summary_text" not in st.session_state: st.session_state.ligand_summary_text = ""
if "smiles_cache" not in st.session_state: st.session_state.smiles_cache = ""
if "selected_native_ligand" not in st.session_state: st.session_state.selected_native_ligand = "None (Manual / Blind Docking)"

# --- MASTER ENVIRONMENT RESET ACTIONS ---
if st.button("🔄 Reset Entire Environment for Fresh Docking", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Clean system background operational scratch files
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb"]:
        if os.path.exists(f): os.remove(f)
    st.success("Dashboard cache and runtime structures completely cleared!")
    st.rerun()

# --- LIGAND LEGENDS QR GAME SCANNER MODULE ---
with st.expander("🃏 Open 'Ligand Legends' Game Card Scanner", expanded=True):
    st.markdown("Hold your NFC/QR game card up to the webcam to instantly auto-load the target receptor and drug molecule into the computational engine!")
    
    if not PYZBAR_AVAILABLE:
        st.error("Missing dependency: The `pyzbar` library is not installed. Run `pip install pyzbar Pillow` in your terminal to enable the card scanner.")
    else:
        scanner_col1, scanner_col2 = st.columns([1, 2])
        
        with scanner_col1:
            camera_image = st.camera_input("Scan Card to Play", key="qr_scanner")
            
        with scanner_col2:
            if camera_image is not None:
                img = Image.open(camera_image)
                decoded_objects = decode(img)
                
                if decoded_objects:
                    qr_text = decoded_objects[0].data.decode('utf-8')
                    try:
                        card_data = json.loads(qr_text)
                        
                        scanned_pdb = card_data.get("pdb_id", "")
                        scanned_smiles = card_data.get("smiles", "")
                        disease = card_data.get("disease", "")
                        drug = card_data.get("drug_name", "")
                        receptor = card_data.get("target_receptor", "")
                        
                        st.success(f"✅ Target Acquired: **{drug}** for {disease}!")
                        st.info(f"🧬 **Receptor:** {receptor} (PDB: {scanned_pdb})")
                        
                        if st.button(f"📥 Initialize '{drug}' Docking Setup", type="primary", use_container_width=True):
                            with st.spinner("Downloading structures and initializing Vina matrices..."):
                                st.session_state.pdb_id_display = scanned_pdb
                                st.session_state.protein_name = receptor
                                st.session_state.smiles_cache = scanned_smiles
                                
                                # Process Protein Data
                                success, path = fetch_pdb_from_rcsb(scanned_pdb)
                                if success:
                                    st.session_state.local_target_path = path
                                    convert_pdb_to_pdbqt(path, "protein.pdbqt")
                                    st.session_state.target_ready = True
                                    
                                # Process Ligand Data
                                pub_data = fetch_ligand_data_from_pubchem(scanned_smiles)
                                ok, _ = convert_smiles_to_pdbqt(scanned_smiles, "ligand.pdbqt")
                                if ok:
                                    st.session_state.ligand_ready = True
                                    with open("ligand.pdbqt", "r") as f:
                                        st.session_state.serialized_ligand_block = f.read()
                                    st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                                
                                st.rerun() # Reload UI with physical card data
                                
                    except json.JSONDecodeError:
                        st.error("Could not parse the QR code data. Please ensure it was generated using the correct JSON format.")
                else:
                    st.warning("No QR code detected in the frame. Please hold the card closer.")

st.write("---")

col_params, col_visual = st.columns([1, 1])

with col_params:
    st.header("1. Target Protein Setup")
    
    # --- TEXT BOXES FOR PROTEIN IDENTIFICATION ---
    current_p_name = st.text_input("Protein Name", placeholder="Hint: Type protein name here...", value=st.session_state.protein_name)
    current_p_id = st.text_input("PDB ID / Code", placeholder="Hint: Type PDB ID here...", value=st.session_state.pdb_id_display)
    
    if current_p_name != st.session_state.protein_name: st.session_state.protein_name = current_p_name
    if current_p_id != st.session_state.pdb_id_display: st.session_state.pdb_id_display = current_p_id
    st.write("---")

    protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
    
    if protein_source == "Type 4-Letter PDB ID":
        pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
        if st.button("📥 Load Target Structure"):
            if pdb_id_input:
                success, path = fetch_pdb_from_rcsb(pdb_id_input)
                if success:
                    st.session_state.local_target_path = path
                    meta = extract_pdb_metadata(path, pdb_id_input.upper())
                    st.session_state.pdb_id_display = meta["id"]
                    st.session_state.protein_name = meta["name"]
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = conv_ok
                    st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
                    st.rerun()
                else: st.error(path)
    else:
        uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
        if uploaded_file:
            path = f"uploaded_{uploaded_file.name}"
            if st.session_state.local_target_path != path:
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.local_target_path = path
                meta = extract_pdb_metadata(path, "Uploaded File")
                st.session_state.pdb_id_display = meta["id"]
                st.session_state.protein_name = meta["name"]
                if uploaded_file.name.endswith(".pdb"):
                    conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = conv_ok
                else:
                    os.replace(path, "protein.pdbqt")
                    st.session_state.target_ready = True
                    st.session_state.local_target_path = None
                st.rerun()

    if st.session_state.target_ready and st.session_state.local_target_path:
        meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
        st.markdown(f"""
        > **Protein Summary Profile:** \n> * **Protein Name:** **{st.session_state.protein_name}** > * **Title:** {meta['title']}  
        > * **PDB ID:** `{st.session_state.pdb_id_display}` | **Classification:** {meta['class']}  
        > * **Organism(s):** *{meta['organism']}* | **Expression System:** {meta['system']}  
        > * **Experimental Method:** {meta['method']} | **Resolution:** **{meta['res']}**
        """)

    st.header("2. Small Molecule Ligand Setup")
    ligand_source = st.radio("Choose Ligand Input Method:", ["SMILES String Input", "Upload Structural File (.pdb, .sdf)"])
    
    smiles_input_val = ""
    uploaded_lig_buffer = None
    uploaded_lig_name = ""

    if ligand_source == "SMILES String Input":
        smiles_input_val = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1").strip()
    else:
        uploaded_lig_file = st.file_uploader("Upload Small Molecule File", type=["pdb", "sdf"])
        if uploaded_lig_file:
            uploaded_lig_buffer = uploaded_lig_file
            uploaded_lig_name = uploaded_lig_file.name

    if st.button("📥 Load Ligand Structure", key="load_ligand_btn"):
        if ligand_source == "SMILES String Input" and smiles_input_val:
            with st.spinner("Querying PubChem Repositories..."):
                pub_data = fetch_ligand_data_from_pubchem(smiles_input_val)
                try:
                    mol = Chem.MolFromSmiles(smiles_input_val)
                    if mol:
                        ok, _ = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
                        if ok:
                            st.session_state.ligand_ready = True
                            st.session_state.smiles_cache = smiles_input_val
                            with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                            st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                            st.success("Ligand metadata mapped from PubChem!")
                            st.rerun()
                except Exception as e: st.error(f"SMILES Parsing Failure: {e}")
            
    elif ligand_source == "Upload Structural File (.pdb, .sdf)" and uploaded_lig_buffer is not None:
            temp_in = f"raw_ligand_{uploaded_lig_name}"
            with open(temp_in, "wb") as f:
                f.write(uploaded_lig_buffer.getbuffer())
            
            mol = Chem.MolFromPDBFile(temp_in, removeHs=False) if uploaded_lig_name.endswith(".pdb") else Chem.SDMolSupplier(temp_in, removeHs=False)[0]
            
            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    AllChem.AssignBondOrdersFromTopology(mol)
                except Exception:
                    pass
                
                if mol.GetNumConformers() == 0:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol)
                
                temp_pdb = "temp_lig_state.pdb"
                Chem.MolToPDBFile(mol, temp_pdb)
                convert_pdb_to_pdbqt(temp_pdb, "ligand.pdbqt", is_ligand=True)
                
                st.session_state.ligand_ready = True
                st.session_state.smiles_cache = temp_in
                st.session_state.ligand_summary_text = "Ligand structure loaded successfully from uploaded file matrix."
                with open("ligand.pdbqt", "r") as f:
                    st.session_state.serialized_ligand_block = f.read()
                
                if os.path.exists(temp_in):
                    os.remove(temp_in)
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
                
                st.success("Structural file loaded and ready for docking!")

    if st.session_state.target_ready and os.path.exists("ligand.pdbqt"):
        st.session_state.ligand_ready = True

    if st.session_state.ligand_ready:
        st.markdown(f"> **Ligand Metric Summary Profile:** \n> {st.session_state.ligand_summary_text}")

    # --- BOUND CO-CRYSTAL SEARCH SITE PANEL ---
    if st.session_state.target_ready and st.session_state.local_target_path:
        bound_ligands_list = parse_bound_ligands(st.session_state.local_target_path)
        if bound_ligands_list:
            st.header("3. Bound Small Molecules in Receptor")
            df_bound = pd.DataFrame(bound_ligands_list)
            df_display = df_bound.copy()
            df_display["Center (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
            df_display["Box (X, Y, Z) Å"] = df_display.apply(lambda r: f"{r['bx']}, {r['by']}, {r['bz']}", axis=1)
            st.dataframe(df_display[["ID", "Chain", "ResSeq", "Atoms", "Center (X, Y, Z) Å", "Box (X, Y, Z) Å"]], hide_index=True, use_container_width=True)
            
            selected_lig_id = st.selectbox("Select native co-crystal target to auto-fill grid box:", options=range(len(bound_ligands_list)), format_func=lambda idx: f"{bound_ligands_list[idx]['ID']} (Chain {bound_ligands_list[idx]['Chain']}-ResSeq {bound_ligands_list[idx]['ResSeq']})")
            if st.button("🎯 Lock Coordinates to Native Site"):
                chosen_target = bound_ligands_list[selected_lig_id]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_target["cx"], chosen_target["cy"], chosen_target["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_target["bx"], chosen_target["by"], chosen_target["bz"]
                st.session_state.selected_native_ligand = f"{chosen_target['ID']} (Chain {chosen_target['Chain']}-ResSeq {chosen_target['ResSeq']})"
                st.success("Grid parameters aligned over pocket boundaries!")
                st.rerun()

    st.header("4. Search Space Mechanics (Grid Box)")
    
    # --- BLIND DOCKING IMPLEMENTATION ---
    if st.button("🌐 Enable Blind Docking (Full Protein Surface)", use_container_width=True):
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            cx, cy, cz, sx, sy, sz = compute_protein_centroid("protein.pdbqt")
            st.session_state.cx, st.session_state.cy, st.session_state.cz = cx, cy, cz
            st.session_state.sx, st.session_state.sy, st.session_state.sz = sx, sy, sz
            st.session_state.selected_native_ligand = "Blind Docking Enabled (Entire Surface Area)"
            st.success("Grid box dynamically expanded to cover the entire macromolecule!")
            st.rerun()
        else:
            st.error("Please load a valid target protein first to enable blind docking.")

    grid_cx = st.number_input("Center X Coordinate", value=float(st.session_state.cx), step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=float(st.session_state.cy), step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=float(st.session_state.cz), step=0.1)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, 40, int(st.session_state.sx))
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, 40, int(st.session_state.sy))
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, 40, int(st.session_state.sz))
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not can_dock)

with col_visual:
    st.header("5. Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None:
        view_tabs = st.tabs(["3D Structural Space", "2D Schematic Topology View"])
        with view_tabs[0]:
            receptor_view_data = ""
            if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
                with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
            render_advanced_modeling_blueprint(receptor_view_data, st.session_state.serialized_ligand_block, mode="cartoon")
        with view_tabs[1]:
            if st.session_state.ligand_ready and st.session_state.smiles_cache:
                try:
                    m_img = Chem.MolFromPDBFile(st.session_state.smiles_cache, removeHs=True) if "raw_ligand" in st.session_state.smiles_cache else Chem.MolFromSmiles(st.session_state.smiles_cache)
                    if m_img:
                        Chem.SanitizeMol(m_img)
                        img_b64 = generate_2d_ligand_img(m_img)
                        if img_b64: st.markdown('<div style="text-align:center; background: white; padding:10px; border-radius:5px;"><img src="data:image/png;base64,{}"/></div>'.format(img_b64), unsafe_html=True)
                except Exception: pass
    else:
        st.subheader("Interactive Complex Viewport")
        if os.path.exists("docking_poses.pdbqt"):
            parsed_poses = split_docking_poses("docking_poses.pdbqt")
            if parsed_poses:
                selected_pose = st.selectbox("Choose Docking Pose to Visualize:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Fit")
                with open("protein.pdbqt", "r") as f: protein_data = f.read()
                
                def get_pose_affinity(stdout_text, idx):
                    for line in stdout_text.split("\n"):
                        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
                        if m and int(m.group(1)) == idx: return m.group(2)
                    return "N/A"
                
                pose_affinity_score = get_pose_affinity(st.session_state.docking_results_raw, selected_pose)
                
                # --- DYNAMIC COLOR LOGIC FOR AFFINITY SCORE ---
                try:
                    aff_val = float(pose_affinity_score)
                    aff_color = "#c62828" if aff_val > 0 else "#1b5e20" # Red if positive, Green if negative
                except ValueError:
                    aff_color = "#1b5e20"

                active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
                
                amino_acid_categories = {"Acidic (-ve)": [], "Basic (+ve)": [], "Polar (Neutral)": [], "Hydrophobic": []}
                for item in active_interactions:
                    res_full = item["Residue Contact"]
                    res_name = "".join([c for c in res_full if c.isalpha()]).upper()
                    if res_name in ["ASP", "GLU"]: amino_acid_categories["Acidic (-ve)"].append(res_full)
                    elif res_name in ["LYS", "ARG", "HIS"]: amino_acid_categories["Basic (+ve)"].append(res_full)
                    elif res_name in ["SER", "THR", "ASN", "GLN", "CYS", "TYR"]: amino_acid_categories["Polar (Neutral)"].append(res_full)
                    else: amino_acid_categories["Hydrophobic"].append(res_full)
                
                breakdown_html = ""
                report_breakdown_text = ""
                has_contacts = False
                for cat_name, res_list in amino_acid_categories.items():
                    if res_list:
                        has_contacts = True
                        labels_joined = ", ".join(list(set(res_list)))
                        breakdown_html += f"<p style='margin:4px 0; font-size:13px;'><b>{cat_name}:</b> <span style='color:#333;'>{labels_joined}</span></p>"
                        report_breakdown_text += f"- {cat_name}: {labels_joined}\n"
                if not has_contacts: 
                    breakdown_html = "<p style='margin:4px 0; color:#777; font-size:13px;'>No pocket interactions detected.</p>"
                    report_breakdown_text = "- No close contacts detected under 3.8 Angstroms.\n"

                html_metric_card = """
                <div style="background-color:#f0f7f4; border-left:6px solid #2e7d32; padding:16px; border-radius:8px; margin-bottom:15px; font-family:sans-serif;">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e0e8e4; padding-bottom:8px; margin-bottom:10px;">
                        <div>
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px;">Active Pose Affinity</span><br>
                            <span style="font-size:36px; font-weight:900; color:{};">{} <span style="font-size:18px; font-weight:normal;">kcal/mol</span></span>
                        </div>
                        <div style="text-align:right; border-left:1px solid #e0e8e4; padding-left:15px;">
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px;">Total Contacts</span><br>
                            <span style="font-size:32px; font-weight:800; color:#2e7d32;">{}</span>
                        </div>
                    </div>
                    <div>
                        <span style="font-size:11px; color:#666; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px; display:block; margin-bottom:4px;">Binding Site Amino Acid Properties Breakdown:</span>
                        {}
                    </div>
                </div>
                """.format(aff_color, pose_affinity_score, len(active_interactions), breakdown_html)
                st.html(html_metric_card)
                
                col_render, col_mesh = st.columns([1, 1])
                with col_render:
                    style_mode = re.sub(r'\W+', '', st.radio("Macromolecule Style Mode:", ["Cartoon Ribbon Mesh", "Spacefill (VDW Configuration)", "Sticks Profile"]).split()[0].lower())
                with col_mesh:
                    surf_toggle = st.checkbox("Overlay Translucent Pocket Cavity Mesh", value=False)
                    
                render_advanced_modeling_blueprint(receptor_data=protein_data, ligand_data=parsed_poses[selected_pose], mode=style_mode, show_surface=surf_toggle, interactions_list=active_interactions)
                
                # --- AUTOMATED COMPREHENSIVE ORIGINAL TEXT REPORT ---
                st.write("---")
                st.subheader("📋 Quick Copy-Paste Citation Report")
                
                report_content = f"""=======================================================
MOLECULAR DOCKING SCREENING ANALYSIS REPORT
Generated dynamically via InSilico BioSphere Docking Tool
Developed by: Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangresearch@gmail.com
=======================================================

1. TARGET RECEPTOR MACROMOLECULE PROFILE
-------------------------------------------------------
- Target Protein Name: {st.session_state.protein_name}
- Target Configuration Identifier (PDB ID): {st.session_state.pdb_id_display}
- Primary Structure Data Source: RCSB Protein Data Bank Server / Local Upload

2. SMALL MOLECULE DRUG LIGAND PROFILE
-------------------------------------------------------
- Input Structural Identity Matrix: {st.session_state.get('smiles_cache', 'Uploaded File Data Track')}
- Compiled Chemical Attributes: {st.session_state.ligand_summary_text.replace('**','')}

3. BOUND SPACE CONFIGURATION MECHANICS (GRID BOX)
-------------------------------------------------------
- Center Coordinates Vector (X, Y, Z): ({grid_cx}, {grid_cy}, {grid_cz})
- Grid Bounding Dimensions (X, Y, Z): ({grid_sx} Å, {grid_sy} Å, {grid_sz} Å)
- Search Algorithm Exhaustiveness Index: {exhaustiveness}
- Grid Alignment Target: {st.session_state.selected_native_ligand}

4. ACTIVE POSE COMPLEX BINDING METRICS (SELECTED MODE)
-------------------------------------------------------
- Target Alignment Selection Mode: Mode {selected_pose} Pose Fit
- Computed Gibbs Free Energy Affinity: {pose_affinity_score} kcal/mol
- Measured Total Spatial Proximity Contact Atoms: {len(active_interactions)}

5. POCKET CONTACT RESIDUES PROACTIVE BREAKDOWN
-------------------------------------------------------
{report_breakdown_text}
=======================================================
Report compiled successfully. Ready for manuscript citation.
**InSilico BioSphere: An Integrated Platform for Automated Molecular Docking.**
    Developed by Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, 
    Shivaji Science College, Nagpur, India.
    Contact - sarangresearch@gmail.com
=======================================================
"""
                st.text_area("Copy Code Summary Report Log Sheet Block directly:", value=report_content, height=250)
                
                # --- NEW PROFESSIONAL HTML REPORT GENERATOR ---
                st.write("---")
                st.subheader("📄 Generate Professional HTML Report")
                
                meta_html = "<p>Data mapping temporarily unavailable.</p>"
                if st.session_state.local_target_path:
                    meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
                    meta_html = f"""
                    <p><b>Protein Name:</b> {st.session_state.protein_name}</p>
                    <p><b>Title:</b> {meta['title']}</p>
                    <p><b>Classification:</b> {meta['class']} | <b>Organism(s):</b> {meta['organism']} | <b>Expression System:</b> {meta['system']}</p>
                    <p><b>Experimental Method:</b> {meta['method']} | <b>Resolution:</b> {meta['res']}</p>
                    """
                
                bound_table_html = "<p>No native co-crystallized ligands detected.</p>"
                if st.session_state.local_target_path:
                    b_ligs = parse_bound_ligands(st.session_state.local_target_path)
                    if b_ligs:
                        df_b = pd.DataFrame(b_ligs)
                        df_b["Center (X, Y, Z) Å"] = df_b.apply(lambda r: f"{r['cx']}, {r['cy']}, {r['cz']}", axis=1)
                        df_b["Box (X, Y, Z) Å"] = df_b.apply(lambda r: f"{r['bx']}, {r['by']}, {r['bz']}", axis=1)
                        bound_table_html = df_b[["ID", "Chain", "ResSeq", "Atoms", "Center (X, Y, Z) Å", "Box (X, Y, Z) Å"]].to_html(index=False, classes="data-table")

                df_results_all = parse_vina_output_with_residues(st.session_state.docking_results_raw)
                docking_table_html = build_styled_html_table(df_results_all)
                
                int_lines_js_offline = ""
                for interact in active_interactions:
                    rc = interact["r_coord"]
                    lc = interact["l_coord"]
                    color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
                    int_lines_js_offline += f"""
                    viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
                    viewer.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
                    """
                
                protein_style_js = ""
                if style_mode == 'cartoon':
                    protein_style_js = "viewer.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
                elif style_mode == 'spacefill':
                    protein_style_js = "viewer.setStyle({model: 0}, {sphere: {colorscheme: 'chain', radius:1.1}});"
                else: 
                    protein_style_js = "viewer.setStyle({model: 0}, {stick: {colorscheme: 'chain', radius:0.25}});"
                    
                surface_js_offline = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if surf_toggle else ""
                    
                html_breakdown_items = ""
                if not has_contacts:
                    html_breakdown_items = "<ul><li>No close contacts detected under 3.8 Angstroms.</li></ul>"
                else:
                    html_breakdown_items = "<ul>"
                    for cat_name, res_list in amino_acid_categories.items():
                        if res_list:
                            labels_joined = ", ".join(list(set(res_list)))
                            html_breakdown_items += f"<li><b>{cat_name}:</b> {labels_joined}</li>"
                    html_breakdown_items += "</ul>"
                    
                int_matrix_html = "<p>No close contacts detected under 3.8 Angstroms.</p>"
                if active_interactions:
                    df_int_html = pd.DataFrame(active_interactions)
                    int_matrix_html = df_int_html[["Residue Contact", "Interaction Type", "Distance (Å)"]].to_html(index=False, classes="data-table")
                
                html_export_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InSilico BioSphere Docking Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 30px; background-color: #fcfcfc; }}
        .header {{ text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 3px solid #1b5e20; }}
        h1 {{ color: #1b5e20; margin-bottom: 5px; font-size: 2.2em; }}
        .subtitle {{ font-size: 1.2em; font-weight: 600; color: #555; }}
        h2 {{ color: #2e7d32; margin-top: 25px; font-size: 1.3em; border-bottom: 1px dashed #ccc; padding-bottom: 5px; }}
        .section {{ background: #ffffff; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid #2e7d32; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        p {{ margin: 8px 0; }}
        .data-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }}
        .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .data-table th {{ background-color: #f2f2f2; color: #333; }}
        .footer {{ margin-top: 50px; font-size: 0.9em; color: #777; border-top: 1px solid #ddd; padding-top: 20px; text-align: center; }}
        #viewer_container {{ height: 500px; width: 100%; position: relative; border-radius:8px; border:1px solid #ccc; background:#fff; margin-top:15px; margin-bottom:20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 InSilico BioSphere Docking Report</h1>
        <div class="subtitle">Department of Chemistry, Shivaji Science College, Nagpur, India</div>
    </div>
    
    <div class="section">
        <h2>1. Target Receptor Profile</h2>
        <p><b>Target Identifier (PDB ID):</b> {st.session_state.pdb_id_display}</p>
        <p><b>Data Source:</b> RCSB Protein Data Bank / Local Upload</p>
        {meta_html}
        <h3>Bound Small Molecules in Receptor</h3>
        {bound_table_html}
    </div>

    <div class="section">
        <h2>2. Small Molecule Ligand Profile</h2>
        <p><b>Structural Matrix:</b> {st.session_state.get('smiles_cache', 'Uploaded File')}</p>
        <p><b>Chemical Attributes:</b> {st.session_state.ligand_summary_text.replace('**','')}</p>
    </div>

    <div class="section">
        <h2>3. Grid Box Mechanics</h2>
        <p><b>Center (X, Y, Z):</b> ({grid_cx}, {grid_cy}, {grid_cz})</p>
        <p><b>Dimensions (X, Y, Z):</b> ({grid_sx} Å, {grid_sy} Å, {grid_sz} Å)</p>
        <p><b>Exhaustiveness:</b> {exhaustiveness}</p>
        <p><b>Grid Alignment Target:</b> {st.session_state.selected_native_ligand}</p>
    </div>

    <div class="section">
        <h2>4. Overall Docking Results</h2>
        {docking_table_html}
    </div>

    <div class="section">
        <h2>5. Selected Pose Analysis (Mode {selected_pose})</h2>
        <p><b>Affinity:</b> <span style="font-size:1.1em; color:{aff_color}; font-weight:bold;">{pose_affinity_score} kcal/mol</span></p>
        <p><b>Total Proximity Contacts:</b> {len(active_interactions)}</p>
        
        <h3>Pocket Contact Residues Breakdown</h3>
        {html_breakdown_items}

        <h3>Interactive 3D Protein-Ligand View</h3>
        <p style="font-size:0.85em; color:#666;">(Use mouse to rotate, scroll to zoom. Cyan lines = van der Waals / pi-Stacking. Yellow lines = Hydrogen Bonds)</p>
        <div id="viewer_container"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
        <script>
            let viewer = $3Dmol.createViewer(document.getElementById('viewer_container'), {{backgroundColor: '#ffffff'}});
            viewer.addModel(`{protein_data}`, 'pdb');
            {protein_style_js}
            {surface_js_offline}
            viewer.addModel(`{parsed_poses[selected_pose]}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
            {int_lines_js_offline}
            viewer.zoomTo(); viewer.render();
        </script>
        
        <h3>Local Contact Residues & Bond Assignments Matrix</h3>
        {int_matrix_html}
    </div>

    <div class="footer">
        <p>Report compiled successfully. Ready for manuscript citation.</p>
        <p><b>InSilico BioSphere: An Integrated Platform for Automated Molecular Docking.</b><br>
        Developed by Mr. Sarang S. Dhote, Assistant Professor, Department of Chemistry,<br>
        Shivaji Science College, Nagpur, India.<br>
        Email: contact - sarangresearch@gmail.com</p>
    </div>
</body>
</html>"""

                st.download_button(
                    label="🔥 Download Full Interactive HTML Report",
                    data=html_export_doc,
                    file_name=f"InSilico_Docking_Report_{st.session_state.pdb_id_display}_Mode{selected_pose}.html",
                    mime="text/html",
                    use_container_width=True,
                    type="primary"
                )

                st.subheader("🧬 Local Contact Residues & Bond Assignments Matrix")
                if active_interactions:
                    df_int = pd.DataFrame(active_interactions)
                    st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
                else:
                    st.info("No close contacts detected within a 3.8 Å threshold radius.")
        else:
            st.info("Initializing active layout matrices workspace pipelines...")

    # --- ENGINE COMPUTATION EXECUTION BOUNDARY ---
if run_btn and can_dock:
    vina_command = [
        "./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
        "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
        "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
        "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"
    ]
    
    # 1. Initialize UI Containers for the Progress Bar and Status text
    progress_bar = st.progress(0, text="Initializing computational engine...")
    status_text = st.empty()
    
    try:
        # 2. Run Popen with raw bytes (unbuffered) to catch real-time asterisk output
        process = subprocess.Popen(
            vina_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT
        )
        
        output_log = []
        progress_count = 0
        current_line = ""
        
        # 3. Read output byte-by-byte
        while True:
            char = process.stdout.read(1).decode("utf-8", errors="ignore")
            
            if not char: # Process has finished
                break
            
            output_log.append(char)
            
            if char == '*':
                progress_count += 1
                percent = min(100, int((progress_count / 50) * 100))
                progress_bar.progress(percent, text=f"Exploring binding modes... {percent}%")
            
            elif char == '\n':
                if "Performing search" in current_line:
                    status_text.info("Executing BFGS optimization and spatial search...")
                elif "Refining" in current_line:
                    status_text.info("Refining top structural poses...")
                current_line = ""
            else:
                current_line += char
        
        process.wait()
        
        if process.returncode == 0:
            progress_bar.progress(100, text="Optimization complete!")
            status_text.empty()
            st.session_state.docking_results_raw = "".join(output_log)
            
            time.sleep(0.8) 
            st.rerun()
        else:
            status_text.empty()
            st.error("Engine encountered a calculation error.")
            st.code("".join(output_log))
            
    except Exception as e:
        st.error(f"Execution pipeline failed: {e}")
        
# --- GLOBAL DATAFRAME ANALYTICS DISPLAY ZONE ---
if st.session_state.docking_results_raw is not None:
    st.write("---")
    st.header("📊 Screening Metrics Dashboard & Data Export")
    
    def parse_vina_output_with_residues_global(stdout_text):
        data = []
        pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)")
        poses_dict = split_docking_poses("docking_poses.pdbqt")
        if not stdout_text: return pd.DataFrame(data)
        for line in stdout_text.split("\n"):
            match = pattern.match(line)
            if match:
                mode_idx = int(match.group(1))
                res_string, bond_types = "N/A", "N/A"
                if mode_idx in poses_dict:
                    ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mode_idx])
                    if ints:
                        res_string = ", ".join(list(set([i["Residue Contact"] for i in ints])))
                        bond_types = ", ".join(list(set([i["Interaction Type"] for i in ints])))
                data.append({"Binding Mode": mode_idx, "Affinity (kcal/mol)": float(match.group(2)), "RMSD l.b.": float(match.group(3)), "RMSD u.b.": float(match.group(4)), "Interacting Residues": res_string, "Contact Bond Types": bond_types})
        return pd.DataFrame(data)

    df_results_global = parse_vina_output_with_residues_global(st.session_state.docking_results_raw)
    
    if not df_results_global.empty:
        col_table, col_export = st.columns([2, 1])
        with col_table: 
            def style_affinity(val):
                try:
                    if float(val) > 0:
                        return 'color: #c62828; font-weight: bold;'
                    else:
                        return 'color: #1b5e20;'
                except Exception:
                    return ''
                    
            try:
                styled_df = df_results_global.style.map(style_affinity, subset=['Affinity (kcal/mol)'])
            except AttributeError: 
                styled_df = df_results_global.style.applymap(style_affinity, subset=['Affinity (kcal/mol)'])
                
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            
        with col_export:
            csv_data = df_results_global.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Data Sheet (.CSV)", data=csv_data, file_name="screening_affinity_report.csv", mime="text/csv", use_container_width=True)
