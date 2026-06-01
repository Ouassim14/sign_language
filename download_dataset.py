import kagglehub
import shutil
import os

# Télécharge le dataset
print("Téléchargement du dataset ASL...")
path = kagglehub.dataset_download("madhavanair/american-sign-language-dataset")
print(f"Dataset téléchargé dans : {path}")

# Copie vers le dossier attendu par le projet
dest = "data/asl_alphabet_train"
os.makedirs(dest, exist_ok=True)

# Ce dataset utilise des dossiers numérotés (0-27) au lieu de noms de classes.
# Mapping : 0-25 → A-Z, 26 → del, 27 → nothing (space absent de ce dataset)
MAPPING = {str(i): chr(ord('A') + i) for i in range(26)}
MAPPING["26"] = "del"
MAPPING["27"] = "nothing"

# Cherche le dossier "data" contenant les sous-dossiers numérotés
source_data_dir = None
for root, dirs, files in os.walk(path):
    if set(dirs) & {"0", "1", "2"}:   # Présence de dossiers numérotés = bonne structure
        source_data_dir = root
        break

if source_data_dir is None:
    # Fallback : cherche asl_alphabet_train comme prévu initialement
    for root, dirs, files in os.walk(path):
        if "asl_alphabet_train" in dirs:
            src = os.path.join(root, "asl_alphabet_train")
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"Dataset (structure nommée) copié vers : {dest}")
            break
    else:
        print("[ERREUR] Structure du dataset non reconnue. Contenu du path téléchargé :")
        for item in os.listdir(path):
            print(f"  {item}")
else:
    print(f"Structure numérique détectée dans : {source_data_dir}")
    print("Copie et renommage des dossiers en cours...")

    for num_folder, class_name in MAPPING.items():
        src_dir = os.path.join(source_data_dir, num_folder)
        dst_dir = os.path.join(dest, class_name)

        if not os.path.isdir(src_dir):
            print(f"  [AVERTISSEMENT] Dossier source manquant : {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)
        files = os.listdir(src_dir)
        for f in files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

        print(f"  {num_folder:>3} -> {class_name:<10} : {len(files)} images copiees")

    print(f"\nDataset réorganisé dans : {dest}")

print("\nPrêt ! Lance maintenant : python src/train.py")
