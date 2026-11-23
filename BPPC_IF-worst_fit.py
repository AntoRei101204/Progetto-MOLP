import json
import time
import math

# Tolleranza numerica per i confronti tra numeri float
EPS = 1e-9

def leggi_istanza(nome_file):
    """
    Legge l'istanza dal file JSON e restituisce:
    - V: Insieme degli oggetti
    - w: dizionario dei pesi degli oggetti
    - W: capacità di ogni bin
    - E: insieme delle coppie di oggetti in conflitto
    - U: numero massimo di bin utilizzabili
    """
    with open(nome_file, "r", encoding = "utf-8") as file:
        dati = json.load(file)
    
    # Convertire i pesi in float
    W = float(dati["capacita"])
    w = {
        item_id : float(weight)
        for item_id, weight in dati["oggetti"].items()
    }
    
    V = list(w.keys())

    E = [
        (pair[0], pair[1])
        for pair in dati["conflitti"]
    ]
    
    # Controlli sui dati dell'istanza
    if W <= 0:
        raise ValueError("La capacità dei bin deve essere positiva.")

    if not V:
        raise ValueError("L'istanza deve contenere almeno un oggetto.")

    if any(weight <= 0 for weight in w.values()):
        raise ValueError("I pesi degli oggetti devono essere positivi.")

    for i, k in E:
        if i not in w or k not in w:
            raise ValueError(f"La coppia di conflitto {(i, k)} contiene oggetti inesistenti.")
        if i == k:
            raise ValueError("Un oggetto non può essere in conflitto con se stesso.")

    # Numero massimo di bin:
    # U = somma_i ceil(w_i / W)
    U = sum(math.ceil(weight / W) for weight in w.values())
    
    return V, w, W, E, U


# Ordinamento degli oggetti
def ordina_oggetti(V, w, conflitti, ordine):
    # "input"       : ordine del file (Worst Fit classico)
    # "decrescente" : peso decrescente (Worst Fit Decreasing)
    # "conflitti"   : grado di conflitto decrescente, a parità peso decrescente
    if ordine == "input":
        return list(V)
    if ordine == "decrescente":
        return sorted(V, key = lambda i: -w[i])
    if ordine == "conflitti":
        return sorted(V, key = lambda i: (-len(conflitti[i]), -w[i]))
    raise ValueError(f"Ordine sconosciuto: {ordine}")


# Euristica Worst Fit per il BPPC-IF
def worst_fit(V, w, W, E, ordine):
    
    # Costruzione della lista dei conflitti di ogni oggetto
    conflitti = {i: set() for i in V}
    for i, k in E:
        conflitti[i].add(k)
        conflitti[k].add(i)
    
    bins = []     # bins[j] = dizionario {oggetto: dimensione del frammento}
    carichi = []  # carichi[j] = carico attuale del bin j
    
    # Ogni oggetto viene inserito seguendo l'ordine scelto
    for i in ordina_oggetti(V, w, conflitti, ordine):
        resto = w[i]  # parte dell'oggetto i ancora da assegnare
        
        while resto > EPS:
            
            # Cerco tra i bin già aperti quello COMPATIBILE con il MAGGIORE spazio libero
            migliore = None
            max_libero = EPS
            for j in range(len(bins)):
                # Se nel bin c'è un oggetto in conflitto con i, il bin è vietato
                if conflitti[i] & bins[j].keys():
                    continue
                libero = W - carichi[j]
                if libero > max_libero:
                    migliore = j
                    max_libero = libero
            
            if migliore is not None:
                # Inserisco un frammento nel bin più vuoto
                q = min(max_libero, resto)
                bins[migliore][i] = q
                carichi[migliore] += q
            else:
                # Nessun bin aperto è utilizzabile: apro un nuovo bin
                q = min(W, resto)
                bins.append({i: q})
                carichi.append(q)
            
            resto -= q
    
    # Riordino i bin per carico decrescente (come il vincolo 5 del modello)
    bins.sort(key = lambda b: -sum(b.values()))
    
    return bins


# Verifica della Soluzione
def verifica_soluzione(bins, V, w, W, E, tol = 1e-6):
    errori = []
    
    for j, b in enumerate(bins, start = 1):
        # Capacità del bin
        if sum(b.values()) > W + tol:
            errori.append(f"Bin {j}: capacità superata")
        # Conflitti nel bin
        for i, k in E:
            if i in b and k in b:
                errori.append(f"Bin {j}: conflitto tra {i} e {k}")
    
    # Ogni oggetto deve essere assegnato per intero
    for i in V:
        totale = sum(b.get(i, 0.0) for b in bins)
        if abs(totale - w[i]) > tol:
            errori.append(f"Oggetto {i}: assegnato {totale:.6f} invece di {w[i]}")
    
    return errori


# Stampa dei Risultati
def prepara_output(bins, V, w, W, E, tempo, ordine):
    errori = verifica_soluzione(bins, V, w, W, E)
    
    righe = [
        "Soluzione Euristica Worst Fit:",
        f"\tOrdine oggetti = {ordine}",
        f"\tTempo impiegato = {tempo:.6f} secondi"
    ]
    
    if errori:
        righe.append("\tSoluzione NON ammissibile:")
        for e in errori:
            righe.append(f"\t\t{e}")
    else:
        righe.append("\tSoluzione ammissibile")
    
    # Valore della funzione obiettivo = numero di bin utilizzati
    righe.append(f"\tValore funzione obiettivo = {len(bins)}")
    
    righe.append("\n========== DETTAGLI DELLA SOLUZIONE ==========")
    
    for j, b in enumerate(bins, start = 1):
        righe.append(
            f"\tBin {j}: carico = {sum(b.values()):.3f}, "
            f"oggetti = {list(b.keys())}"
        )
        # Quantità assegnata di ciascun oggetto
        for i, q in b.items():
            righe.append(f"\tOggetto {i}: frammento = {q:.3f}")
    
    return righe


def scrivi_output(righe, nome_file):
    with open(nome_file, "w") as f:
        for riga in righe:
            f.write(riga + "\n")


def main():
    # Caricamento dell'istanza
    file_input = "istanza_09_OS2.json"  # Nome del file di input
    V, w, W, E, U = leggi_istanza(file_input)
    ordine = "decrescente"  # oggetti ordinati per peso decrescente
    file_output = "risultati_WorstFit_BPPC-IF.txt"  # Nome del file di output

    print("========== DATI DELL'ISTANZA ==========")
    print("Oggetti:", V)
    print("Pesi:", w)
    print("Capacità dei bin:", W)
    print("Coppie in conflitto:", E)
    print("Numero massimo di bin:", U)

    # Risoluzione con Worst Fit
    t0 = time.perf_counter()  # tempo di inizio
    bins = worst_fit(V, w, W, E, ordine)
    tempo = time.perf_counter() - t0  # tempo impiegato

    # Stampa della soluzione
    righe = prepara_output(bins, V, w, W, E, tempo, ordine)
    
    for riga in righe:
        print(riga)
        
    scrivi_output(righe, file_output)
    

if __name__ == "__main__":
    main()