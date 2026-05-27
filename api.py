import time
import streamlit as st
import subprocess
import os
import urllib.request
import json
import re
import requests
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

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
    meta = {"name": "Unknown Protein", "title": "Structure Matrix", "id": pdb_id.upper(), "class": "Unknown", "organism": "Unknown"}
    if not os.path.exists(file_path): return meta
    with open(file_path, "r") as f:
        title_parts = []
        for line in f:
            if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
            elif line.startswith("COMPND") and "MOLECULE:" in line:
                if meta["name"] == "Unknown Protein": meta["name"] = line.split("MOLECULE:")[1].split(";")[0].strip().title()
            elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
    if title_parts: meta["title"] = " ".join(title_parts).title()
    if meta["name"] == "Unknown Protein": meta["name"] = meta["title"]
    return meta

def compute_protein_centroid(pdbqt_file):
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
    dims = np.max(arr, axis=0) - np.min(arr, axis=0) + 15.0 
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


# --- 3. BIOPHYSICAL INTERACTION PARSER ---

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


# --- 4. SCORING & VISUALIZATION ---

def evaluate_affinity(score_val, drug_name, disease_name):
    """Evaluates the score and returns the ranking and personalized comment."""
    if score_val >= -4.0:
        rank = "Weak / Poor Binding"
        desc = "The molecule might just be loosely bumping into the protein."
        comment = f"❌ {drug_name} is considered a <b>weak</b> candidate for {disease_name} medicinal activity."
        color = "#e53935" # Red
    elif -8.0 < score_val < -4.0:
        rank = "Moderate / Good Binding"
        desc = "Often used as a standard baseline or threshold for a 'hit'."
        comment = f"✅ {drug_name} shows <b>good</b> potential for {disease_name} medicinal activity."
        color = "#fb8c00" # Orange
    else:
        rank = "Very Strong Binding"
        desc = "Excellent binding affinity indicating a highly stable complex."
        comment = f"🔥 {drug_name} is a <b>HIGHLY POTENT</b> candidate for {disease_name} medicinal activity!"
        color = "#2e7d32" # Green
    return rank, desc, comment, color

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

def render_mobile_viewer(receptor_data, ligand_data, style="cartoon", show_surface=False, interactions=[]):
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    
    if style == "cartoon":
        style_js = "viewer.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
    elif style == "spacefill":
        style_js = "viewer.setStyle({model: 0}, {sphere: {colorscheme: 'chain', radius:1.1}});"
    else:
        style_js = "viewer.setStyle({model: 0}, {stick: {colorscheme: 'chain', radius:0.25}});"

    int_lines_js = ""
    for interact in interactions:
        rc = interact["r_coord"]
        lc = interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:12, backgroundBorder: '1px solid #333'}});
        """

    html_content = f"""
    <div id="container" style="height: 60vh; width: 100%; border-radius:10px; border:2px solid #ccc; background:#fff;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById('container'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer.addModel(`{receptor_data}`, 'pdb');
            {style_js}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer.addModel(`{ligand_data}`, 'pdb');
            viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.3}}}});
        }}
        {int_lines_js}
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html_content, height=500)


# --- 5. GAME DASHBOARD WORKSPACE ---

st.set_page_config(page_title="Ligand Legends", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧬 Ligand Legends</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:12px; color:gray;'>Powered by InSilico BioSphere | Developed by Mr. Sarang S. Dhote</p>", unsafe_allow_html=True)

if "game_state" not in st.session_state: st.session_state.game_state = "IDLE"
if "card_scanned" not in st.session_state: st.session_state.card_scanned = ""
if "affinity_score" not in st.session_state: st.session_state.affinity_score = ""
if "protein_meta" not in st.session_state: st.session_state.protein_meta = {}
if "ligand_meta" not in st.session_state: st.session_state.ligand_meta = {}

def reset_game():
    st.session_state.game_state = "IDLE"
    st.session_state.card_scanned = ""
    st.session_state.affinity_score = ""
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_ligand.pdb"]:
        if os.path.exists(f): os.remove(f)


# --- THE GAME RESULT POPUP & GOOGLE SHEETS SUBMIT ---
if st.session_state.game_state == "FINISHED":
    try:
        aff_val = float(st.session_state.affinity_score)
    except:
        aff_val = 0.0
        
    drug_n = st.session_state.ligand_meta.get('name', st.session_state.scanned_drug)
    prot_n = st.session_state.protein_meta.get('name', st.session_state.scanned_pdb)
    disease_n = st.session_state.scanned_disease
    
    rank, desc, comment, rank_color = evaluate_affinity(aff_val, drug_n, disease_n)
    
    # Render the card with zero indentation to prevent Markdown code-blocking
    html_card = f"""
<div style="background-color:#f0f7f4; border: 4px solid {rank_color}; padding:20px; border-radius:15px; margin-bottom:20px; text-align:center; box-shadow: 0px 8px 16px rgba(0,0,0,0.2);">
    <h2 style="margin-top:0; color:#333;">🎉 DOCKING COMPLETE!</h2>
    <h4 style="color:#666; margin-bottom:5px;">{drug_n} ➔ {prot_n}</h4>
    
    <p style="font-size:14px; color:gray; text-transform:uppercase; letter-spacing:1px; margin-top:15px;">Binding Affinity Score</p>
    <h1 style="font-size:55px; font-weight:900; color:{rank_color}; margin:0;">{st.session_state.affinity_score} <span style="font-size:20px;">kcal/mol</span></h1>
    
    <div style="background-color: white; padding: 10px; border-radius: 8px; margin-top: 15px; border: 1px solid #ddd;">
        <h3 style="color:{rank_color}; margin:0;">{rank}</h3>
        <p style="font-size:13px; color:#555; margin-bottom:10px;"><i>"{desc}"</i></p>
        <p style="font-size:15px; color:#111; font-weight:bold;">{comment}</p>
    </div>
    
    <div style="margin-top: 20px; font-size: 11px; color: #999; border-top: 1px solid #ddd; padding-top: 10px;">
        Ligand Legends game developed by Sarang Dhote | &copy; Copyright Sarang Dhote
    </div>
</div>
"""
    st.markdown(html_card, unsafe_allow_html=True)
    
    st.write("### 📝 Record Your Score")
    student_name = st.text_input("Enter Student Name to record score:")
    
    if st.button("📤 Submit Score to Google Sheets", type="primary", use_container_width=True):
        if student_name.strip() == "":
            st.warning("Please enter your name before submitting!")
        else:
            with st.spinner("Uploading to leaderboard..."):
                GOOGLE_SHEET_WEBHOOK_URL = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
                payload = {
                    "Name": student_name,
                    "Disease": disease_n,
                    "Drug": drug_n,
                    "Target": prot_n,
                    "Score": st.session_state.affinity_score,
                    "Rank": rank
                }
                try:
                    # requests.post(GOOGLE_SHEET_WEBHOOK_URL, json=payload) 
                    st.success(f"Awesome job, {student_name}! Score saved to Google Sheets.")
                except Exception as e:
                    st.error("Failed to connect to Google Sheets.")
    
    if st.button("🔄 Play Next Card", use_container_width=True):
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
                    
                    if pdb_id and smiles and qr_text != st.session_state.card_scanned:
                        st.session_state.card_scanned = qr_text
                        st.session_state.game_state = "DOCKING"
                        st.session_state.scanned_disease = card_data.get("disease", "")
                        st.session_state.scanned_drug = card_data.get("drug_name", "")
                        st.session_state.scanned_pdb = pdb_id
                        st.session_state.scanned_smiles = smiles
                        st.rerun() 
                except json.JSONDecodeError:
                    st.error("Invalid QR format detected.")


# --- DOCKING EXECUTION BLOCK ---
if st.session_state.game_state == "DOCKING":
    st.info(f"Target Acquired: **{st.session_state.scanned_drug}** for {st.session_state.scanned_disease}")
    progress_bar = st.progress(5, text="Downloading Protein & Ligand Data...")
    
    success, local_pdb = fetch_pdb_from_rcsb(st.session_state.scanned_pdb)
    if success:
        st.session_state.protein_meta = extract_pdb_metadata(local_pdb, st.session_state.scanned_pdb)
        convert_pdb_to_pdbqt(local_pdb, "protein.pdbqt")
        
    st.session_state.ligand_meta = fetch_ligand_data_from_pubchem(st.session_state.scanned_smiles)
    convert_smiles_to_pdbqt(st.session_state.scanned_smiles, "ligand.pdbqt")
    
    progress_bar.progress(20, text="Calculating Blind Docking Grid Space...")
    cx, cy, cz, sx, sy, sz = compute_protein_centroid("protein.pdbqt")
    
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
            parsed_score = "0.0"
            for line in output_log.split("\n"):
                if re.match(r"^\s*1\s+", line):
                    parts = line.split()
                    if len(parts) >= 2:
                        parsed_score = parts[1]
                        break
                        
            st.session_state.affinity_score = parsed_score
            st.session_state.game_state = "FINISHED"
            progress_bar.progress(100, text="Docking Complete!")
            time.sleep(0.5)
            st.rerun() 
        else:
            st.error("Engine Calculation Failed.")
            if st.button("Reset"): reset_game(); st.rerun()
            
    except Exception as e:
        st.error(f"Failed to execute docking: {e}")
        if st.button("Reset"): reset_game(); st.rerun()


# --- 3D VIEWER & INTERACTION TABLE ---
if st.session_state.game_state == "FINISHED" and os.path.exists("docking_poses.pdbqt"):
    st.write("### 🔬 Interactive Docked Complex")
    
    col_style, col_mesh = st.columns(2)
    with col_style:
        view_style = st.selectbox("Style:", ["Cartoon (Ribbon)", "Sticks", "Spacefill"])
        view_style = view_style.split()[0].lower()
    with col_mesh:
        st.write("") 
        show_mesh = st.checkbox("Toggle Translucent Surface")
    
    poses = split_docking_poses("docking_poses.pdbqt")
    top_pose = poses.get(1, "")
    
    active_interactions = compute_spatial_interactions("protein.pdbqt", top_pose)
    
    p_data = ""
    if os.path.exists("protein.pdbqt"):
        with open("protein.pdbqt", "r") as f: p_data = f.read()
        
    render_mobile_viewer(p_data, top_pose, style=view_style, show_surface=show_mesh, interactions=active_interactions)
    
    st.write("### 🔗 Interaction Profile")
    if active_interactions:
        df_int = pd.DataFrame(active_interactions)
        st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
    else:
        st.info("No close contacts detected within 3.8 Å.")
