import streamlit as st
import json
from PIL import Image
try:
    from pyzbar.pyzbar import decode
except ImportError:
    st.error("Please run: pip install pyzbar Pillow")

# --- LIGAND LEGENDS: QR SCANNER MODULE ---
st.header("🃏 Scan 'Ligand Legends' Game Card")
st.write("Hold your NFC/QR card up to the webcam to instantly load the target receptor and drug molecule!")

# 1. Open the camera input
camera_image = st.camera_input("Scan Card to Play", key="qr_scanner")

if camera_image is not None:
    # 2. Convert the image to a format pyzbar can read
    img = Image.open(camera_image)
    decoded_objects = decode(img)
    
    if decoded_objects:
        # 3. Extract the text from the QR Code
        qr_text = decoded_objects[0].data.decode('utf-8')
        
        try:
            # 4. Parse the JSON
            card_data = json.loads(qr_text)
            
            # Extract data
            scanned_pdb = card_data.get("pdb_id", "")
            scanned_smiles = card_data.get("smiles", "")
            disease = card_data.get("disease", "")
            drug = card_data.get("drug_name", "")
            receptor = card_data.get("target_receptor", "")
            
            # 5. UI Feedback
            st.success(f"✅ Card Scanned Successfully: **{drug}** for {disease}!")
            st.info(f"🧬 **Targeting:** {receptor} (PDB: {scanned_pdb})")
            
            # 6. Inject the values directly into your existing session state pipeline
            # This perfectly bridges the physical card to your computational backend!
            if st.button(f"📥 Initialize {drug} Docking Setup", type="primary"):
                st.session_state.pdb_id_display = scanned_pdb
                st.session_state.protein_name = receptor
                
                # We can place the SMILES right into the cache so your engine picks it up
                st.session_state.smiles_cache = scanned_smiles
                
                # Fetch structures using your existing functions
                success, path = fetch_pdb_from_rcsb(scanned_pdb)
                if success:
                    st.session_state.local_target_path = path
                    convert_pdb_to_pdbqt(path, "protein.pdbqt")
                    st.session_state.target_ready = True
                    
                pub_data = fetch_ligand_data_from_pubchem(scanned_smiles)
                ok, _ = convert_smiles_to_pdbqt(scanned_smiles, "ligand.pdbqt")
                if ok:
                    st.session_state.ligand_ready = True
                    with open("ligand.pdbqt", "r") as f:
                        st.session_state.serialized_ligand_block = f.read()
                    st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                
                st.rerun() # Refresh the UI with the loaded setup
                
        except json.JSONDecodeError:
            st.error("Could not parse the QR code data. Please ensure it was generated using the correct JSON format.")
    else:
        st.warning("No QR code detected in the frame. Please hold the card closer.")
