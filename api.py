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
from rdkit.Chem import AllChem, Draw
from PIL import Image

try:
    from pyzbar.pyzbar import decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False


# --- 1. CLOUD CONTEXT ENGINE MANAGEMENT ---

def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")

ensure_linux_vina_exists()


# --- 2. BIOINFORMATICS PIPELINE FUNCTIONS ---

def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception: pass 
    return metadata

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {"name": "Unknown Protein", "title": "Structure Matrix", "id": pdb_id.upper(), "class": "Unknown", "organism": "Unknown", "method": "N/A", "res": "N/A"}
    if not os.path.exists(file_path): return meta
    with open(file_path, "r") as f:
        title_parts = []
        for line in f:
            if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
            elif line.startswith("HEADER"): meta["class"] = line[10:50].strip().title()
            elif line.startswith("COMPND") and "MOLECULE:" in line:
                if meta["name"] == "Unknown Protein": meta["name"] = line.split("MOLECULE:")[1].split(";")[0].strip().title()
            elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
            elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
            elif "RESOLUTION." in line and "ANGSTROMS." in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match: meta["res"] = f"{match.group(1)} Å"
    if title_parts: meta["title"] = " ".join(title_parts).title()
    if meta["name"] == "Unknown Protein": meta["name"] = meta["title"]
    return meta

def compute_protein_centroid(pdbqt_file):
    """Calculates the geometric center and dimensions for full-protein blind docking."""
    coords = []
    if not os.path.exists(pdbqt_file): return 0.0, 0.0, 0.0, 20.0, 20.0, 20.0
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if not coords: return 0.0, 0.0, 0.0, 20.0, 20.0, 20.0
    arr = np.array(coords)
    center = np.mean(arr, axis=0)
    dims = np.max(arr, axis=0) - np.min(arr, axis=0) + 15.0 # Added slight padding for blind docking
    return center[0], center[1], center[2], dims[0], dims[1], dims[2]

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", local_pdb)
        return True, local_pdb
    except Exception: return False, None

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    autodock_map = {"H":"H", "C":"C", "N":"N", "O":"O", "S":"S", "P":"P", "F":"F", "CL":"Cl", "BR":"Br", "I":"I", "ZN":"Zn", "MG":"Mg"}
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except: torsions = 4
    try:
        with open(input_pdb, "r") as pdb, open(output_pdbqt, "w") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    rec = line[:6].strip()
                    try: aid = int(line[6:11].strip())
                    except: aid = 1
                    aname = line[12:16]
                    rname = line[17:20].strip()
                    chain = line[21].strip() or "A"
                    try: rseq = int(line[22:26].strip())
                    except: rseq = 1
                    try: x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    except: continue
                    elem = line[76:78].strip()
                    if not elem: elem = ''.join([c for c in aname if c.isalpha()])[0]
                    elem = ''.join([c for c in elem if c.isalpha()]).upper()
                    vtype = autodock_map.get(elem, elem.title())
                    if elem == "C" and "AR" in aname.upper(): vtype = "A"
                    pdbqt.write(f"{rec:<6}{aid:>5} {aname:<4} {rname:>3} {chain}{rseq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vtype:<2}\n")
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        return True
    except: return False

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if not mol: return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.MolToPDBFile(mol, "temp_ligand.pdb")
        convert_pdb_to_pdbqt("temp_ligand.pdb", output_filename, is_ligand=True)
        if os.path.exists("temp_ligand.pdb"): os.remove("temp_ligand.pdb")
        return True
    except: return False


# --- 3. RESULTS PARSING & VISUALIZATION ---

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    mode, lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                mode = int(line.split()[1])
                lines = []
            elif line.startswith("ENDMDL"):
                if mode is not None: poses[mode] = "".join(lines)
                mode = None
            else: lines.append(line)
    return poses

def render_mobile_viewer(receptor_data, ligand_data):
    html_content = f"""
    <div id="container" style="height: 60vh; width: 100%; border-radius:10px; border:2px solid #ccc; background:#fff;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            viewer.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
        }}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.3}}}});
        }}
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html_content, height=500)


# --- 4. GAME DASHBOARD WORKSPACE ---

st.set_page_config(page_title="Ligand Legends", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧬 Ligand Legends</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:12px; color:gray;'>Powered by InSilico BioSphere | Developed by Mr. Sarang S. Dhote</p>", unsafe_allow_html=True)

# State Management
if "game_state" not in st.session_state: st.session_state.game_state = "IDLE" # IDLE, DOCKING, FINISHED
if "card_scanned" not in st.session_state: st.session_state.card_scanned = ""
if "affinity_score" not in st.session_state: st.session_state.affinity_score = ""
if "protein_meta" not in st.session_state: st.session_state.protein_meta = {}
if "ligand_meta" not in st.session_state: st.session_state.ligand_meta = {}

# Reset function
def reset_game():
    st.session_state.game_state = "IDLE"
    st.session_state.card_scanned = ""
    st.session_state.affinity_score = ""
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_ligand.pdb"]:
        if os.path.exists(f): os.remove(f)

# --- THE GAME RESULT POPUP (Shows when docking completes) ---
if st.session_state.game_state == "FINISHED":
    aff_val = float(st.session_state.affinity_score)
    color = "#2e7d32" if aff_val < 0 else "#c62828" # Green if negative, Red if positive
    
    st.markdown(f"""
    <div style="background-color:#f0f7f4; border: 4px solid {color}; padding:20px; border-radius:15px; margin-bottom:20px; text-align:center; box-shadow: 0px 8px 16px rgba(0,0,0,0.2);">
        <h2 style="margin-top:0; color:#333;">🎉 DOCKING COMPLETE!</h2>
        <h4 style="color:#666; margin-bottom:5px;">{st.session_state.ligand_meta.get('name', 'Drug')} ➔ {st.session_state.protein_meta.get('name', 'Receptor')}</h4>
        <p style="font-size:14px; color:gray; text-transform:uppercase; letter-spacing:1px;">Binding Affinity Score</p>
        <h1 style="font-size:55px; font-weight:900; color:{color}; margin:0;">{st.session_state.affinity_score}</h1>
        <p style="font-size:16px; color:{color};">kcal/mol</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("❌ Close & Play Next Card", use_container_width=True, type="primary"):
        reset_game()
        st.rerun()
    st.write("---")


# --- THE SCANNER & AUTO-DOCK PIPELINE ---
if st.session_state.game_state == "IDLE":
    if not PYZBAR_AVAILABLE:
        st.error("Missing dependency: `pyzbar`. Please install via terminal to enable the scanner.")
    else:
        st.write("### 📸 Scan your Ligand Card")
        camera_image = st.camera_input("Hold QR Card to Camera", key="qr")
        
        if camera_image is not None:
            img = Image.open(camera_image)
            decoded = decode(img)
            
            if decoded:
                qr_text = decoded[0].data.decode('utf-8')
                try:
                    card_data = json.loads(qr_text)
                    pdb_id = card_data.get("pdb_id", "")
                    smiles = card_data.get("smiles", "")
                    
                    # Prevent scanning the exact same card in an infinite loop
                    if pdb_id and smiles and qr_text != st.session_state.card_scanned:
                        st.session_state.card_scanned = qr_text
                        st.session_state.game_state = "DOCKING"
                        
                        st.session_state.scanned_disease = card_data.get("disease", "")
                        st.session_state.scanned_drug = card_data.get("drug_name", "")
                        st.session_state.scanned_pdb = pdb_id
                        st.session_state.scanned_smiles = smiles
                        
                        st.rerun() # Trigger the docking block below
                except json.JSONDecodeError:
                    st.error("Invalid QR format detected.")


# --- DOCKING EXECUTION BLOCK ---
if st.session_state.game_state == "DOCKING":
    st.info(f"Target Acquired: **{st.session_state.scanned_drug}** for {st.session_state.scanned_disease}")
    
    progress_bar = st.progress(5, text="Downloading Protein & Ligand Data...")
    
    # 1. Fetch & Prepare Data
    success, local_pdb = fetch_pdb_from_rcsb(st.session_state.scanned_pdb)
    if success:
        st.session_state.protein_meta = extract_pdb_metadata(local_pdb, st.session_state.scanned_pdb)
        convert_pdb_to_pdbqt(local_pdb, "protein.pdbqt")
        
    st.session_state.ligand_meta = fetch_ligand_data_from_pubchem(st.session_state.scanned_smiles)
    convert_smiles_to_pdbqt(st.session_state.scanned_smiles, "ligand.pdbqt")
    
    # 2. Setup Blind Docking Grid
    progress_bar.progress(20, text="Calculating Blind Docking Grid Space...")
    cx, cy, cz, sx, sy, sz = compute_protein_centroid("protein.pdbqt")
    
    # 3. Run Vina
    progress_bar.progress(30, text="Igniting InSilico BioSphere Engine...")
    vina_cmd = [
        "./vina", "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz), 
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz), 
        "--exhaustiveness", "8", "--out", "docking_poses.pdbqt"
    ]
    
    try:
        process = subprocess.Popen(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_log = ""
        p_count = 0
        
        while True:
            char = process.stdout.read(1).decode("utf-8", errors="ignore")
            if not char: break
            output_log += char
            if char == '*':
                p_count += 1
                percent = min(99, 30 + int((p_count / 50) * 70))
                progress_bar.progress(percent, text=f"Simulating Binding Modes... {percent}%")
                
        process.wait()
        
        if process.returncode == 0:
            # Extract top score
            match = re.search(r"^\s*1\s+([-+]?\d+\.\d+)", output_log, re.MULTILINE)
            st.session_state.affinity_score = match.group(1) if match else "N/A"
            st.session_state.game_state = "FINISHED"
            progress_bar.progress(100, text="Docking Complete!")
            time.sleep(0.5)
            st.rerun() # Reloads app to show the Results Popup at the top
        else:
            st.error("Engine Calculation Failed.")
            if st.button("Reset"): reset_game(); st.rerun()
            
    except Exception as e:
        st.error(f"Failed to execute docking: {e}")
        if st.button("Reset"): reset_game(); st.rerun()


# --- DATA PROFILES DISPLAY (Read-Only) ---
if st.session_state.game_state in ["DOCKING", "FINISHED"] and st.session_state.protein_meta:
    p_meta = st.session_state.protein_meta
    l_meta = st.session_state.ligand_meta
    
    st.write("### 🧬 Combatant Data")
    st.markdown(f"""
    **Receptor:** {p_meta.get('name', 'Unknown')} (`{p_meta.get('id', '')}`)  
    *Organism:* {p_meta.get('organism', 'Unknown')} | *Res:* {p_meta.get('res', 'N/A')}
    
    **Ligand:** {l_meta.get('name', st.session_state.scanned_drug)}  
    *Formula:* {l_meta.get('formula', 'N/A')} | *Weight:* {l_meta.get('mw', 'N/A')}
    """)


# --- 3D VIEWER (Shifted to absolute bottom for mobile scrolling) ---
if st.session_state.game_state == "FINISHED" and os.path.exists("docking_poses.pdbqt"):
    st.write("---")
    st.write("### 🔬 Final Docked Complex")
    st.write("Scroll around to view the binding pocket. (Top pose selected)")
    
    poses = split_docking_poses("docking_poses.pdbqt")
    top_pose = poses.get(1, "") # Get the best pose (Mode 1)
    
    p_data = ""
    if os.path.exists("protein.pdbqt"):
        with open("protein.pdbqt", "r") as f: p_data = f.read()
        
    render_mobile_viewer(p_data, top_pose)
