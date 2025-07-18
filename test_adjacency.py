import trimesh
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
import io # Per caricare la mesh da una stringa

# --- 1. Definiamo una mesh di esempio semplice (Tetraedro) in formato OBJ ---
# Un tetraedro ha 4 vertici e 4 facce triangolari.
# Ogni faccia è adiacente alle altre 3.
tetrahedron_obj_data = """

v 0.0 0.0 0.0   
v 1.0 0.0 0.0   
v 0.5 1.0 0.0   
v 0.5 0.5 1.0   


f 1 2 3  
f 1 2 4  
f 1 3 4  
f 2 3 4  
"""

print("--- Caricamento Mesh di Esempio (Tetraedro) ---")

mesh = None
try:
    # Carichiamo la mesh dalla stringa, specificando il tipo e assicurando il processing
    # Nota: usiamo io.StringIO per trattare la stringa come se fosse un file
    mesh = trimesh.load_mesh(file_obj=io.StringIO(tetrahedron_obj_data), 
                             file_type='obj', 
                             process=True) # process=True è importante
    
    # Verifica del caricamento
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        print("Errore: La mesh non è stata caricata correttamente come Trimesh o è vuota.")
        exit()
        
    print(f"Mesh caricata con successo.")
    print(f"Numero vertici: {len(mesh.vertices)}")
    print(f"Numero facce (triangoli): {len(mesh.faces)}")
    # Stampiamo le facce caricate (dovrebbero usare indici 0-based)
    print("Facce caricate (indici 0-based dei vertici):")
    print(mesh.faces)
    # Atteso: [[0 1 2], [0 1 3], [0 2 3], [1 2 3]]

except Exception as e:
    print(f"Errore fatale durante il caricamento della mesh: {e}")
    exit()

print("\n--- Calcolo Matrice di Adiacenza da trimesh.face_adjacency ---")

num_faces = len(mesh.faces)
adjacency_matrix = None # Inizializza

try:
    # 1. Ottieni le coppie di facce adiacenti per spigolo da trimesh
    adj_pairs = mesh.face_adjacency
    print(f"Coppie di facce adiacenti (indici 0-based) trovate da trimesh:\n{adj_pairs}")
    # Atteso per tetraedro: [[0 1], [0 2], [0 3], [1 2], [1 3], [2 3]] (o ordine diverso)

    if adj_pairs.size == 0 and num_faces > 0:
        print("Warning: trimesh.face_adjacency restituito vuoto.")
        adjacency_matrix = lil_matrix((num_faces, num_faces), dtype=int)
    elif num_faces == 0:
         print("Warning: La mesh non ha facce.")
         adjacency_matrix = lil_matrix((0, 0), dtype=int)
    else:
        # 2. Prepara i dati per coo_matrix
        num_connections = len(adj_pairs)
        data = np.ones(num_connections * 2, dtype=np.int8) # Valori '1'
        # Indici di riga (f1 -> f2 e f2 -> f1)
        rows = np.concatenate((adj_pairs[:, 0], adj_pairs[:, 1])) 
        # Indici di colonna (f1 -> f2 e f2 -> f1)
        cols = np.concatenate((adj_pairs[:, 1], adj_pairs[:, 0])) 
        
        # 3. Crea la matrice COO
        coo_adj = coo_matrix((data, (rows, cols)), shape=(num_faces, num_faces))
        print(f"Matrice COO creata con {len(data)} elementi non-zero.")
        
        # 4. Converti in formato LIL (per accesso efficiente alle righe)
        adjacency_matrix = coo_adj.tolil() 
        print("Matrice convertita in formato LIL.")
        # Opzionale: Stampa la matrice densa (solo per mesh piccole!)
        print("Matrice di adiacenza (densa):\n", adjacency_matrix.toarray())

except Exception as e:
    print(f"Errore durante la creazione della matrice di adiacenza: {e}")
    # Crea matrice vuota in caso di errore
    if adjacency_matrix is None:
         adjacency_matrix = lil_matrix((num_faces, num_faces), dtype=int)


print("\n--- Estrazione Vicini di Primo Anello per Ogni Faccia ---")

if adjacency_matrix is not None:
    for face_index in range(num_faces):
        try:
            # Estrai la riga dalla matrice LIL
            row = adjacency_matrix.getrow(face_index)
            # Trova gli indici delle colonne non-zero
            neighbors = row.nonzero()[1]
            
            print(f"Faccia {face_index}: Vicini di primo anello -> {list(neighbors)}")
            # Per il tetraedro, ogni faccia dovrebbe avere le altre 3 come vicini.
            # Es: Faccia 0 dovrebbe avere vicini [1, 2, 3] (in ordine crescente)
            # Es: Faccia 1 dovrebbe avere vicini [0, 2, 3] 
            # etc.
            
        except IndexError:
             print(f"Errore: face_index {face_index} fuori dai limiti per la matrice A.")
        except Exception as e:
             print(f"Errore nell'elaborare la faccia {face_index}: {e}")
else:
    print("Matrice di adiacenza non disponibile.")

print("\n--- Script Terminato ---")