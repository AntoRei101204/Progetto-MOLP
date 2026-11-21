import json
import time
import math
import random 
import pyomo.environ as pyo
import xpress
from pyomo.opt import SolverStatus, TerminationCondition


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
    
    # Gli ID JSON sono stringhe: li manteniamo stringhe anche nei conflitti.
    W = float(dati["capacita"])
    w = {
        item_id : float(weight)
        for item_id, weight in dati["oggetti"].items()
    }
    
    V = list(w.keys())

    E = [
        (str(pair[0]), str(pair[1]))
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

# Costruzione del Modello Matematico
def crea_modello (V, w, W, E, U):
    
    # Creazione del modello
    model = pyo.ConcreteModel(name = "BPPC-IF")
    
    # Insiemi
    model.V = pyo.Set(initialize = V)  # insieme degli oggetti
    model.B = pyo.RangeSet(1, U) # insieme dei bin
    model.E = pyo.Set(dimen = 2, initialize = E)  # insieme dei lati
    
    # Parametri
    model.w = pyo.Param(model.V, initialize = w)
    model.W = pyo.Param(initialize = W)
    
    # Variabili Decisionali
    # x[i, j] = 1 se un frammento di i è messo nel bin j, 0 altrimenti
    model.x = pyo.Var(model.V, model.B, domain = pyo.Binary)
    
    # f[i, j] = grandezza del frammento di i messo nel bin j
    model.f = pyo.Var(model.V, model.B, domain = pyo.NonNegativeReals)
    
    # y[j] = 1 se il bin j viene utilizzato, 0 altrimenti
    model.y = pyo.Var(model.B, domain = pyo.Binary)
    
    # Funzione Obiettivo: Minimizzare il numero di bin utilizzati
    model.obj = pyo.Objective(expr = sum(model.y[j] for j in model.B), sense = pyo.minimize)
    
    # Vincolo 1: Capacità dei bin
    def vincolo_capacita(model, j):
        return sum(model.f[i, j] for i in model.V) <= model.W * model.y[j]
    
    model.capacity = pyo.Constraint(model.B, rule = vincolo_capacita)
    
    # Vincolo 2: Tutti i frammenti di ogni oggetto devono essere assegnati 
    def vincolo_assegnamento(model, i):
        return sum(model.f[i, j] for j in model.B) == model.w[i]
    
    model.assignment = pyo.Constraint(model.V, rule = vincolo_assegnamento)
    
    # Vincolo 3: Collegamento tra f e x
    def vincolo_collegamento(model, i, j):
        return model.f[i, j] <= model.w[i] * model.x[i, j]
    
    model.collegamento = pyo.Constraint(model.V, model.B, rule = vincolo_collegamento)
    
    # Vincolo 4: Conflitti tra oggetti
    def vincolo_conflitto(model, i, k, j):
        return model.x[i, j] + model.x[k, j] <= model.y[j]
    
    model.conflitti = pyo.Constraint(model.E, model.B, rule = vincolo_conflitto)
    
    # Vincolo 5: Ordinamento dei carichi dei bin
    model.ordinamento = pyo.ConstraintList()
    for j in range(2, U + 1):
        model.ordinamento.add( sum(model.f[i, j] for i in model.V) <= sum(model.f[i, j-1] for i in model.V) )
        
    # Vincolo 6: Ogni oggetto può essere assegnato a un bin solo se il bin è utilizzato
    def vincolo_bin_utilizzato(model, i, j):
        return model.x[i, j] <= model.y[j]
    
    model.bin_utilizzato = pyo.Constraint(model.V, model.B, rule = vincolo_bin_utilizzato)
    
    return model


# Risoluzione del Modello
def risolvi_modello(model, time_limit):
    opt = pyo.SolverFactory('xpress') # creo un oggetto solver per risolvere il modello (utilizzo il solver xpress)
    opt.options["timelimit"] = time_limit  # limite di tempo di 60 secondi
        
    t0 = time.perf_counter()  # tempo di inizio della risoluzione
    results = opt.solve(model)  # risolvo il modello
    t1 = time.perf_counter()  # tempo di fine della risoluzione
        
    return results, t1 - t0  # ritorno i risultati e il tempo impiegato per risolvere il modello

# Verifica della Soluzione 
def verifica_soluzione_milp(model, results, tempo, time_limit):
    status = results.solver.status
    termination_condition = results.solver.termination_condition
    
    # Valore presente solo se Xpress ha caricato una soluzione.
    z = pyo.value(model.obj, exception=False)
    
    if status == SolverStatus.ok and termination_condition == TerminationCondition.optimal:
        return "Soluzione Ottima", True
    if termination_condition ==TerminationCondition.infeasible:
        return "Problema Inammissibile", False
    if termination_condition == TerminationCondition.unbounded:
        return "Problema Illimitato", False
    
    if z is not None:
        if tempo >= time_limit:
            return "Time Limit raggiunto, soluzione ammissibile, ma non è detto che sia ottima", True
        return "Soluzione ammissibile, non ottima", True
    
    if tempo >= time_limit:
        return "Time Limit raggiunto, nessuna soluzione trovata", False
    return "Nessuna soluzione trovata", False
    

# Stampa dei Risultati
def prepara_output(model, results, tempo, time_limit):

    status = results.solver.status
    termination_condition = results.solver.termination_condition

    righe = [
        "Soluzione Modello:",
        f"\tStato = {status}",
        f"\tCondizione di terminazione = {termination_condition}",
        f"\tTempo impiegato = {tempo:.3f} secondi",
        f"\tLimite di tempo = {time_limit} secondi"
    ]

    # Verifica se esiste una soluzione caricata
    messaggio, disponibile = verifica_soluzione_milp(model, results, tempo, time_limit)
    righe.append(f"\t{messaggio}")
    
    if not disponibile:
        return righe

    # Valore della funzione obiettivo
    z = pyo.value(model.obj, exception = False)

    if z is not None:
        righe.append(f"\tValore funzione obiettivo = {z}")
    
    righe.append("\n========== DETTAGLI DELLA SOLUZIONE ==========")

    for j in model.B:

        # Verifica se il bin è utilizzato
        y_value = pyo.value(model.y[j], exception=False)
        if y_value is not None and y_value > 0.5:
            oggetti_bin = []
            carico_bin = 0
            for i in model.V:
                x_value = pyo.value(model.x[i, j], exception=False)
                f_value = pyo.value(model.f[i, j], exception=False)
                if x_value is not None and f_value is not None and x_value > 0.5 and f_value > 1e-6:
                    oggetti_bin.append(i)
                    carico_bin += f_value
            righe.append(
                f"\tBin {j}: carico = {carico_bin:.3f}, "
                f"oggetti = {oggetti_bin}"
            )

            # Quantità assegnata di ciascun oggetto
            for i in oggetti_bin:
                quantita = pyo.value(model.f[i, j], exception=False)
                righe.append(
                    f"\tOggetto {i}: frammento = {quantita:.3f}"
                )
    return righe


def scrivi_output(righe, nome_file):
    with open(nome_file, "w", encoding="utf-8") as f:
        for riga in righe:
            f.write(riga + "\n")


# ============================================================
# EURISTICHE COSTRUTTIVE: First Fit e Next Fit 
# ============================================================
EPS = 1e-9


def ordina_oggetti(V, w, conflitti, ordine):
    # Ordine per peso decrescente oppure grado di conflitto decrescente.
    if ordine == "input":
        return list(V)
    if ordine == "decrescente":
        return sorted(V, key=lambda i: -w[i])
    if ordine == "conflitti":
        return sorted(V, key=lambda i: (-len(conflitti[i]), -w[i]))
    raise ValueError(f"Ordine sconosciuto: {ordine}")


def first_fit(V, w, W, E, ordine):
    # First Fit adattato a conflitti e frammentazione
    conflitti = {i: set() for i in V}
    for i, k in E:
        conflitti[i].add(k)
        conflitti[k].add(i)

    bins = []
    carichi = []

    for i in ordina_oggetti(V, w, conflitti, ordine):
        resto = w[i]

        # Prova tutti i bin già aperti, dal primo all'ultimo.
        for j in range(len(bins)):
            if resto <= EPS:
                break
            if conflitti[i] & bins[j].keys():
                continue
            libero = W - carichi[j]
            if libero <= EPS:
                continue
            q = min(libero, resto)
            bins[j][i] = bins[j].get(i, 0.0) + q
            carichi[j] += q
            resto -= q

        # Apre nuovi bin se la quantità restante non trova posto.
        while resto > EPS:
            q = min(W, resto)
            bins.append({i: q})
            carichi.append(q)
            resto -= q

    bins.sort(key=lambda b: -sum(b.values()))
    return bins


def next_fit(V, w, W, E, ordine):
    # Next Fit adattato: utilizza solo l'ultimo bin aperto.
    conflitti = {i: set() for i in V}
    for i, k in E:
        conflitti[i].add(k)
        conflitti[k].add(i)

    bins = [{}]
    carichi = [0.0]

    for i in ordina_oggetti(V, w, conflitti, ordine):
        resto = w[i]
        while resto > EPS:
            in_conflitto = bool(conflitti[i] & bins[-1].keys())
            pieno = (W - carichi[-1]) <= EPS
            if in_conflitto or pieno:
                bins.append({})
                carichi.append(0.0)

            libero = W - carichi[-1]
            q = min(libero, resto)
            bins[-1][i] = bins[-1].get(i, 0.0) + q
            carichi[-1] += q
            resto -= q

    bins.sort(key=lambda b: -sum(b.values()))
    return bins


def verifica_soluzione_euristica(bins, V, w, W, E, tol=1e-6):
    # Controlla ogni bin, i conflitti e l'assegnamento completo degli oggetti.
    errori = []
    totale = {i: 0.0 for i in V}
    if not isinstance(bins, list) or not bins:
        return ["La soluzione deve essere una lista non vuota di bin."]

    for j, b in enumerate(bins, start=1):
        if not isinstance(b, dict) or not b:
            errori.append(f"Bin {j}: vuoto o non valido")
            continue

        carico = 0.0
        presenti = set()
        for i, q in b.items():
            if i not in totale:
                errori.append(f"Bin {j}: oggetto sconosciuto {i}")
                continue
            if not isinstance(q, (int, float)) or not math.isfinite(q):
                errori.append(f"Bin {j}: quantità non valida per {i}")
                continue
            if q <= 0:
                errori.append(f"Bin {j}: quantità non positiva per {i}")
                continue
            carico += q
            totale[i] += q
            presenti.add(i)

        if carico > W + tol:
            errori.append(f"Bin {j}: capacità superata ({carico:.6f} > {W})")
        for i, k in E:
            if i in presenti and k in presenti:
                errori.append(f"Bin {j}: conflitto tra {i} e {k}")

    for i in V:
        if abs(totale[i] - w[i]) > tol:
            errori.append(f"Oggetto {i}: assegnato {totale[i]:.6f} invece di {w[i]}")
    return errori


def prepara_output_euristica(nome, bins, V, w, W, E, tempo, ordine):
    # Stampa i risultati come con il modello
    errori = verifica_soluzione_euristica(bins, V, w, W, E)
    righe = [
        f"Soluzione Euristica {nome}:",
        f"\tOrdine oggetti = {ordine}",
        f"\tTempo impiegato = {tempo:.6f} secondi",
        "\tSoluzione ammissibile" if not errori else "\tSoluzione NON ammissibile",
        f"\tValore funzione obiettivo = {len(bins)}"
    ]
    for errore in errori:
        righe.append(f"\tERRORE: {errore}")

    righe.append("\n========== DETTAGLI DELLA SOLUZIONE ==========")
    for j, b in enumerate(bins, start=1):
        righe.append(f"\tBin {j}: carico = {sum(b.values()):.3f}, oggetti = {list(b.keys())}")
        for i, q in b.items():
            righe.append(f"\tOggetto {i}: frammento = {q:.3f}")
    return righe

def first_fit_randomizzato(V, w, W, E, generatore):
    # First Fit con randomizzazione: mescola l'ordine degli oggetti prima di applicare First Fit.
    
    # Creo una copia dell'insieme degli oggetti per non modificare l'originale del First Fit
    V_random = list(V)
    
    # Mescolo casualmente l'ordine degli oggetti utilizzando il generatore fornito
    generatore.shuffle(V_random)
    
    # Richiamo il First Fit già implementato. Con "input", First Fit rispetta l'ordine della lista che gli viene passata.
    return first_fit(V_random, w, W, E, "input")

def prova_elimina_bin(bins, indice_da_eliminare, W, conflitti, generatore):
    # Prova a riassegnare gli oggetti del bin da eliminare agli altri bin esistenti. 
    # Restituisce la nuova soluzione se riesce a riassegnare i frammentie e elimnare il bin, altrimenti restituisce None.
    
    # Creo una copia di tutti i bin tranne quello da eliminare
    altri_bin = [dict(b) for j, b in enumerate(bins) if j != indice_da_eliminare]
    
    # Recuoero tutti i frammenti del bin da eliminare 
    frammenti = list(bins[indice_da_eliminare].items())
    
    # Mescolo l'ordine degli oggetti da riassegnare per introdurre casualità
    generatore.shuffle(frammenti)
    
    # Provo a reassegnare ogni frammento agli altri bin esistenti
    for i, quantita in frammenti:
        resto = quantita
        # Creo un ordine casuale dei bin disponibiliper tentare la riassegnazione
        posizioni = list(range(len(altri_bin)))
        generatore.shuffle(posizioni)
        
        for j in posizioni:
            if resto <= EPS:
                break
            b = altri_bin[j]
            
            # Controllo conflitti e capacità: Non posso inserire un oggetto i in un bin che contiene un oggetto incompatibile
            if conflitti[i] & b.keys():
                continue  # l'oggetto è già presente in questo bin, quindi c'è un conflitto
            # calcola capacità residua del bin 
            spazio = W - sum(b.values())
            
            if spazio <= EPS:
                continue  # il bin è pieno
            
            # Inserisco la quantità massima possibile.
            q = min(spazio, resto)

            # Aggiungo il frammento al bin selezionato.
            b[i] = b.get(i, 0.0) + q

            # Aggiorno la quantità ancora da collocare.
            resto -= q
            
        # Se non riesco a collocare tutto il frammento, il tentativo di eliminazione fallisce.
        if resto > EPS:
            return None

    # Tutti i frammenti sono stati ricollocati.
    # Riordino i bin per carico decrescente.
    altri_bin.sort(key=lambda b: -sum(b.values()))

    return altri_bin


def Local_Search_elimina_bin(bins, V, w, W, E, generatore, max_tentativi=5):
    """
    Migliora una soluzione provando a ridurre
    il numero di bin utilizzati.

    Si tenta prima di eliminare i bin meno carichi.
    Dopo ogni eliminazione riuscita, la ricerca ricomincia.
    """

    # Costruisco il dizionario dei conflitti.
    conflitti = {i: set() for i in V}
 
    for i, k in E:
        conflitti[i].add(k)
        conflitti[k].add(i)

    # Copio la soluzione iniziale.
    soluzione = [dict(b) for b in bins]

    # Continuo la ricerca finché ci sono almeno due bin.
    while len(soluzione) > 1:

        migliorata = False

        # Ordino gli indici dei bin per carico crescente. Parto da quelli meno carichi perché, in genere, richiedono meno quantità da redistribuire.
        indici = sorted(range(len(soluzione)), key=lambda j: sum(soluzione[j].values()))

        # Provo a eliminare i bin nell'ordine stabilito.
        for indice in indici:

            # Per ogni bin provo diverse redistribuzioni casuali.
            for _ in range(max_tentativi):

                candidata = prova_elimina_bin(soluzione, indice, W, conflitti, generatore)

                # Se candidata non è None, l'eliminazione del bin è riuscita.
                if candidata is not None:

                    # Verifico che la nuova soluzione rispetti capacità, conflitti e assegnamenti.
                    errori = verifica_soluzione_euristica(candidata, V, w, W, E)

                    if errori:
                        raise RuntimeError("Ricerca locale non ammissibile: " + str(errori))

                    # Accetto la soluzione migliorata.
                    soluzione = candidata
                    migliorata = True
                    break

            # Se ho eliminato un bin, interrompo la visita e ricomincio dalla nuova soluzione.
            if migliorata:
                break

        # Se nessun bin può essere eliminato con i tentativi disponibili, interrompo la ricerca.
        if not migliorata:
            break

    return soluzione
   
    
def LS_Fragment_Relocation(bins, V, w, W, E, max_mosse=30):
    """
    Sposta frammenti da bin meno carichi verso bin piu' carichi.

    La mossa mantiene ammissibilita' e numero di bin, tranne quando
    svuota completamente un bin. L'obiettivo ausiliario e' aumentare
    la concentrazione dei carichi, per favorire Bin Elimination.

    Per evitare cicli si accettano solo mosse che aumentano
    strettamente la somma dei quadrati dei carichi.
    """
    
    # controllo sul numero massimo di mosse
    if max_mosse < 0:
        raise ValueError("max_mosse deve essere non negativo")

    # creo il dizionario dei conflitti e per ogni oggetto memorizzo quelli incompatibili
    conflitti = {i: set() for i in V}
    for i, k in E:
        conflitti[i].add(k)
        conflitti[k].add(i)

    # copio la soluzione iniziale per non modificare l'originale
    soluzione = [dict(b) for b in bins]
    
    # eseguo al massimo max_mosse mosse di ricollocazione dei frammenti
    for _ in range(max_mosse):
        # calcolo del carico attuale di ogni bin
        carichi = [sum(b.values()) for b in soluzione]
        migliore = None
        miglioramento_massimo = EPS  # inizializzo il miglioramento massimo a un valore molto piccolo 
                                     # (incremento minimo per accettare una mossa)

        # Individuo la mossa ammissibile con maggior incremento di somma(carico_bin ** 2). Non apro mai nuovi bin.
        for origine, b_origine in enumerate(soluzione):
            for i, quantita in b_origine.items():                # considero ogni frammento del bin 
                for destinazione, b_dest in enumerate(soluzione):  # esamino tutti i possibili bin di destinazione
                    if origine == destinazione:   # non sposto un frammento nello stesso bin
                        continue
                    if conflitti[i] & b_dest.keys():  # controllo che il bin di destinazione non abbia oggetti in conflitto con l'oggetto i
                        continue
                    spazio = W - carichi[destinazione] # calcolo lo spazio disponibile nel bin di destinazione
                    if spazio <= EPS:
                        continue
                    q = min(quantita, spazio)  # quanto frammento posso spostare dal bin di origine al bin di destinazione (tutto il frammento o solo una parte?)
                    if q <= EPS:
                        continue

                    # Variazione della somma dei quadrati dei carichi.
                    a = carichi[origine]
                    b = carichi[destinazione]
                    incremento = (a - q)**2 + (b + q)**2 - a**2 - b**2
                    if incremento > miglioramento_massimo:
                        migliore = (origine, destinazione, i, q)
                        miglioramento_massimo = incremento          # conservo la mossa che migliora l'incremento
        # se nessuna mossa migliora il criterio, interrompo la ricerca
        if migliore is None:
            break  # Nessuna ricollocazione che migliori il criterio ausiliario

        origine, destinazione, i, q = migliore
        soluzione[destinazione][i] = soluzione[destinazione].get(i, 0.0) + q
        soluzione[origine][i] -= q
        if soluzione[origine][i] <= EPS:
            del soluzione[origine][i]
        if not soluzione[origine]:
            soluzione.pop(origine)  # Un bin completamente svuotato si elimina

    soluzione.sort(key=lambda b: -sum(b.values()))
    errori = verifica_soluzione_euristica(soluzione, V, w, W, E)
    if errori:
        raise RuntimeError("Fragment Relocation non ammissibile: " + str(errori))
    return soluzione


def ricerca_locale_combinata(bins, V, w, W, E, generatore, max_tentativi=5, max_mosse=30):
    """
    Alterna Bin Elimination, Fragment Relocation e Bin Elimination.

    In questo ordine verifichiamo prima le eliminazioni immediate;
    poi ridistribuiamo i frammenti per concentrare il carico e
    riproviamo a eliminare bin. Non peggiora il numero di bin.
    """
    soluzione = Local_Search_elimina_bin(bins, V, w, W, E, generatore, max_tentativi)
    soluzione = LS_Fragment_Relocation(soluzione, V, w, W, E, max_mosse)
    soluzione = Local_Search_elimina_bin(soluzione, V, w, W, E, generatore, max_tentativi)
    return soluzione
    
    
def multi_start_first_fit(V, w, W, E, iterazioni=100, seme=42, usa_ricerca_locale=True, max_tentativi=5, max_mosse=30):
    """
    Esegue più costruzioni First Fit randomizzate.
    Ogni soluzione viene eventualmente migliorata
    tramite la ricerca locale Bin Elimination.
    Restituisce:
    - migliore_costruita: migliore soluzione prima della LS
    - migliore_finale: migliore soluzione dopo la LS
    - tempo: tempo totale del Multi-Start
    - miglioramenti_locali: numero di partenze migliorate
    """

    # Controllo che il numero di iterazioni e tentativi siano valori positivi.
    if iterazioni < 1 or max_tentativi < 1:
        raise ValueError(
            "Iterazioni e max_tentativi devono essere positivi"
        )

    # Creo il generatore casuale.
    generatore = random.Random(seme)

    # Inizialmente non ho soluzioni.
    migliore_costruita = None
    migliore_finale = None

    # Contatore delle partenze migliorate dalla LS.
    miglioramenti_locali = 0

    # Avvio la misurazione del tempo.
    t0 = time.perf_counter()

    # Ripeto la procedura per il numero stabilito di partenze casuali.
    for _ in range(iterazioni):

        # 1. Costruzione di una soluzione casuale.
        iniziale = first_fit_randomizzato(V, w, W, E, generatore)

        # Verifico la soluzione costruita.
        errori = verifica_soluzione_euristica(iniziale, V, w, W, E)

        if errori:
            raise RuntimeError("First Fit randomizzato non ammissibile: " + str(errori))

        # Aggiorno la migliore soluzione costruttiva.
        if (migliore_costruita is None or len(iniziale) < len(migliore_costruita)):
            migliore_costruita = [dict(b) for b in iniziale]

        # 2. Applico la ricerca locale, se abilitata.
        if usa_ricerca_locale:
            finale = ricerca_locale_combinata(iniziale, V, w, W, E, generatore, max_tentativi, max_mosse)
        else:
            finale = iniziale

        # Registro se la LS ha migliorato questa partenza.
        if len(finale) < len(iniziale):
            miglioramenti_locali += 1

        # 3. Aggiorno la migliore soluzione complessiva.
        if (migliore_finale is None or len(finale) < len(migliore_finale)):
            migliore_finale = [dict(b) for b in finale]

    # Calcolo il tempo totale.
    tempo = time.perf_counter() - t0

    return migliore_costruita, migliore_finale, tempo, miglioramenti_locali


# ============================================================
# META-EURISTICA ILS: PERTURBAZIONE
# ============================================================

def perturbazione_ils(bins, W, E, generatore, num_bin_perturbati=3):
    """
    Perturba una soluzione ammissibile del BPPC-IF.

    Seleziona casualmente alcuni bin, rimuove i loro frammenti
    e prova a reinserirli negli altri bin disponibili.

    Se lo spazio non è sufficiente, apre nuovi bin.

    La perturbazione può generare una soluzione peggiore
    rispetto a quella iniziale: questo è consentito in ILS.

    Parametri:
    - bins: soluzione corrente
    - W: capacità di ogni bin
    - E: insieme delle coppie di oggetti in conflitto
    - generatore: generatore di numeri casuali
    - num_bin_perturbati: numero di bin da rimuovere

    Restituisce:
    - nuova: soluzione perturbata
    """

    # Controllo dei parametri.
    if num_bin_perturbati < 1:
        raise ValueError("num_bin_perturbati deve essere almeno 1")

    if not bins:
        raise ValueError("La soluzione da perturbare non può essere vuota")

    # Non posso selezionare più bin di quelli esistenti.
    quanti = min(num_bin_perturbati, len(bins))

    # Scelgo casualmente quali bin rimuovere. sample() seleziona indici distinti, senza ripetizioni.
    indici = set(generatore.sample(range(len(bins)), quanti))

    # Copio tutti i bin NON selezionati. Uso dict(b) per non modificare la soluzione originale.
    nuova = [dict(b) for j, b in enumerate(bins) if j not in indici]

    # Raccolgo tutti i frammenti contenuti nei bin rimossi. Ogni elemento è una coppia (oggetto, quantità).
    frammenti = [(i, quantita) for j in indici for i, quantita in bins[j].items()]

    # Cambio casualmente l'ordine di reinserimento.
    generatore.shuffle(frammenti)

    # Costruisco le liste di incompatibilità.
    conflitti = {i: set() for b in bins for i in b}

    for i, k in E:
        conflitti[i].add(k)
        conflitti[k].add(i)

    # Reinserisco i frammenti nella soluzione.
    for i, quantita in frammenti:
        # Quantità dell'oggetto ancora da collocare.
        resto = quantita
        # Considero i bin rimasti in ordine casuale.
        ordine_bin = list(range(len(nuova)))
        generatore.shuffle(ordine_bin)

        for j in ordine_bin:
            # Se il frammento è stato assegnato interamente, non devo cercare altri bin.
            if resto <= EPS:
                break
            
            b = nuova[j]
            # Il bin non è utilizzabile se contiene un oggetto in conflitto con i.
            if conflitti[i] & b.keys():
                continue

            # Capacità residua del bin.
            libero = W - sum(b.values())
            if libero <= EPS:
                continue

            # Inserisco la massima quantità possibile.
            q = min(libero, resto)

            # Se l'oggetto è già presente nel bin, sommo il nuovo frammento a quello esistente.
            b[i] = b.get(i, 0.0) + q

            # Aggiorno la quantità ancora da assegnare.
            resto -= q

        # Se rimane del peso non assegnato, apro uno o più nuovi bin.
        while resto > EPS:
            q = min(W, resto)
            nuova.append({i: q})
            resto -= q

    # Riordino i bin per carico decrescente.
    nuova.sort(key=lambda b: -sum(b.values()))

    return nuova


# ============================================================
# META-EURISTICA ILS: CRITERIO DI ACCETTAZIONE
# ============================================================

def accetta_soluzione_ils(candidata, corrente, generatore, probabilita_peggiori=0.10):
    """
    Decide se la soluzione candidata diventa la nuova
    soluzione corrente dell'ILS.

    Parametri:
    - candidata: soluzione ottenuta dopo perturbazione e Local Search
    - corrente: soluzione dalla quale siamo partiti
    - generatore: generatore di numeri casuali
    - probabilita_peggiori: probabilità di accettare
      una soluzione con più bin della corrente

    Restituisce:
    - True se la candidata viene accettata
    - False altrimenti
    """

    # La probabilità deve essere compresa tra 0 e 1.
    if not 0 <= probabilita_peggiori <= 1:
        raise ValueError("probabilita_peggiori deve essere compresa tra 0 e 1")

    # Accetto sempre le soluzioni migliori o equivalenti in termini di numero di bin utilizzati.
    if len(candidata) <= len(corrente):
        return True

    # Se la soluzione è peggiorativa, genero un numero casuale compreso tra 0 e 1.
    # Accetto la soluzione soltanto se il valore casuale è inferiore alla probabilità stabilita.
    return generatore.random() < probabilita_peggiori

# ============================================================
# META-EURISTICA: ITERATED LOCAL SEARCH
# ============================================================

def ils(V, w, W, E, iterazioni=100, seme=42, num_bin_perturbati=3, probabilita_peggiori=0.10, max_tentativi=5, max_mosse=30):
    """
    Implementazione della metaeuristica Iterated Local Search.

    Procedura:
    1. Costruisco una soluzione iniziale con First Fit Decreasing.
    2. Applico la ricerca locale combinata.
    3. Perturbo la soluzione corrente.
    4. Applico nuovamente la ricerca locale.
    5. Aggiorno la migliore soluzione trovata.
    6. Decido se accettare la candidata.
    7. Ripeto fino al numero massimo di iterazioni.

    Parametri:
    - V, w, W, E: dati dell'istanza BPPC-IF
    - iterazioni: numero di iterazioni ILS
    - seme: seme del generatore casuale
    - num_bin_perturbati: numero di bin coinvolti
      in ogni perturbazione
    - probabilita_peggiori: probabilità di accettare
      una soluzione che utilizza più bin
    - max_tentativi: tentativi per bin in Bin Elimination
    - max_mosse: massimo numero di mosse Fragment Relocation

    Restituisce:
    - iniziale: soluzione First Fit Decreasing
    - migliore: migliore soluzione ILS
    - tempo: tempo totale di esecuzione
    - miglioramenti: aggiornamenti della migliore soluzione
    - peggioramenti_accettati: numero di soluzioni peggiorative
      accettate come nuove soluzioni correnti
    """

    # Controllo dei parametri.
    if iterazioni < 1:
        raise ValueError("Le iterazioni ILS devono essere positive")

    if max_tentativi < 1 or max_mosse < 0:
        raise ValueError("Parametri di ricerca locale non validi")

    if num_bin_perturbati < 1:
        raise ValueError("num_bin_perturbati deve essere positivo")

    if not 0 <= probabilita_peggiori <= 1:
        raise ValueError("probabilita_peggiori deve essere tra 0 e 1")

    # Creo un generatore pseudo-casuale con seme fissato.
    generatore = random.Random(seme)

    # Avvio il cronometro.
    t0 = time.perf_counter()

    # FASE 1: COSTRUZIONE DELLA SOLUZIONE INIZIALE
    # Utilizzo First Fit Decreasing già implementato.
    iniziale = first_fit(V, w, W, E, "decrescente")

    # FASE 2: PRIMA RICERCA LOCALE
    # Applico Bin Elimination e Fragment Relocation.
    corrente = ricerca_locale_combinata(iniziale, V, w, W, E, generatore, max_tentativi, max_mosse)

    # Memorizzo la migliore soluzione trovata finora. Deve essere una copia indipendente dalla corrente.
    migliore = [dict(b) for b in corrente]

    # Contatori per l'analisi sperimentale.
    miglioramenti = 0
    peggioramenti_accettati = 0

    # FASE 3: CICLO PRINCIPALE ILS
    for _ in range(iterazioni):

        # 3A. Perturbo la soluzione corrente, non necessariamente la migliore trovata.
        perturbata = perturbazione_ils(corrente, W, E, generatore, num_bin_perturbati)

        # Verifico che la perturbazione sia ammissibile.
        errori = verifica_soluzione_euristica(perturbata, V, w, W, E)

        if errori:
            raise RuntimeError("Perturbazione ILS non ammissibile: " + str(errori))

        # 3B. Applico la ricerca locale alla soluzione perturbata.
        candidata = ricerca_locale_combinata(perturbata, V, w, W, E, generatore, max_tentativi, max_mosse)

        # Verifico anche la soluzione dopo la ricerca locale.
        errori = verifica_soluzione_euristica(candidata, V, w, W, E)

        if errori:
            raise RuntimeError("ILS non ammissibile: " + str(errori))

        # 3C. Aggiorno la migliore soluzione globale. Questo controllo è indipendente dall'accettazione.
        if len(candidata) < len(migliore):

            migliore = [dict(b) for b in candidata]

            miglioramenti += 1

        # 3D. Decido se continuare dalla soluzione candidata.
        if accetta_soluzione_ils(candidata, corrente, generatore, probabilita_peggiori):
            # Registro le accettazioni peggiorative.
            if len(candidata) > len(corrente):
                peggioramenti_accettati += 1
            # La candidata diventa la nuova soluzione corrente.
            corrente = candidata

    # FASE 4: RISULTATI FINALI
    tempo = time.perf_counter() - t0
    return ( iniziale, migliore, tempo, miglioramenti, peggioramenti_accettati )


def main():
    # E' necessario cambiare nome dell'istanza e del file di output a seconda dei casi.
    file_input = "istanza_09_OS2.json"
    time_limit = 300
    ordine = "decrescente"  # inserire "input", "decrescente", "conflitti" a seconda dell'euristica da testare
    file_output = "risultati_BPPC-IF.txt"
    # Parametri del Multi-Start e della ricerca locale
    iterazioni_ms = 100  # Numero di soluzioni iniziali da generare
    seme_ms = 42         # Seme per rendere riproducibili i risultati
    tentativi_ls = 5     # Tentativi di eliminazione per ciascun bin
    mosse_fr = 30       # Numero massimo di mosse per la ricerca locale di frammenti
    # Parametri ILS
    iterazioni_ils = 100
    seme_ils = 42   # Seme per rendere riproducibili le scelte casuali.
    bin_perturbati_ils = 3       # Numero di bin da rimuovere e reinserire durante ogni perturbazione.
    probabilita_peggiori_ils = 0.10     # Probabilità di accettare una soluzione peggiorativa. 0.10 equivale al 10%.


    V, w, W, E, U = leggi_istanza(file_input)
    print("========== DATI DELL'ISTANZA ==========")
    print("Oggetti:", V)
    print("Pesi:", w)
    print("Capacità dei bin:", W)
    print("Coppie in conflitto:", E)
    print("Numero massimo di bin:", U)

    # Modello MILP
    model = crea_modello(V, w, W, E, U)
    
    # Inizialmente non abbiamo risultati del MILP
    results = None
    tempo_milp = None

    try:
        # Provo a risolvere il modello utilizzando Xpress
        results, tempo_milp = risolvi_modello(model, time_limit)

        # Se Xpress riesce a terminare senza errori, preparo normalmente l'output del MILP
        righe = prepara_output(model, results, tempo_milp, time_limit)

    except xpress.SolverError as errore:
        # Se Xpress genera un errore (ad esempio per il limite dimensionale della licenza), non interrompo il programma
        righe = [
            "========== RISULTATO MODELLO MILP ==========",
            "MILP non risolto da Xpress.",
            f"Motivo: {errore}",
            "First Fit e Next Fit verranno comunque eseguiti."
        ]
    # results, tempo_milp = risolvi_modello(model, time_limit)
    # righe = prepara_output(model, results, tempo_milp, time_limit)


    # First Fit
    t0 = time.perf_counter()
    bins_ff = first_fit(V, w, W, E, ordine)
    tempo_ff = time.perf_counter() - t0
    righe += ["", ""] + prepara_output_euristica(
        "First Fit", bins_ff, V, w, W, E, tempo_ff, ordine
    )

    # Next Fit
    t0 = time.perf_counter()
    bins_nf = next_fit(V, w, W, E, ordine)
    tempo_nf = time.perf_counter() - t0
    righe += ["", ""] + prepara_output_euristica(
        "Next Fit", bins_nf, V, w, W, E, tempo_nf, ordine
    )


    # ============================================================
    # RISOLUZIONE CON MULTI-START E RICERCA LOCALE (First Fit randomizzato + Bin Elimination)
    # ============================================================

    # Eseguo il Multi-Start con First Fit randomizzato e Bin Elimination applicata a ogni soluzione iniziale.
    bins_ms_base, bins_ms, tempo_ms, miglioramenti_ls = multi_start_first_fit(
        V, w, W, E,
        iterazioni=iterazioni_ms,
        seme=seme_ms,
        usa_ricerca_locale=True,
        max_tentativi=tentativi_ls, 
        max_mosse=mosse_fr
    )

    # Preparo il riepilogo dei risultati.
    righe += ["", "========== MULTI-START E RICERCA LOCALE First Fit =========="]

    righe.append(f"Numero di partenze casuali = {iterazioni_ms}")
    righe.append(f"Seme casuale = {seme_ms}")
    righe.append(f"Tentativi per bin da eliminare = {tentativi_ls}")
    righe.append(f"Numero massimo di mosse per frammenti = {mosse_fr}")

    # Migliore soluzione prima della ricerca locale
    righe.append(f"Miglior First Fit randomizzato = {len(bins_ms_base)} bin")

    # Migliore soluzione dopo la ricerca locale
    righe.append(f"Miglior Multi-Start + ricerca locale = {len(bins_ms)} bin")

    # Numero di partenze migliorate dalla ricerca locale
    righe.append(f"Partenze migliorate dalla ricerca locale = {miglioramenti_ls}/{iterazioni_ms}")

    # Utilizzo la funzione di output già presente per stampare anche i dettagli della soluzione Multi-Start.
    righe += prepara_output_euristica( "Multi-Start + Ricerca Locale", bins_ms, V, w, W, E, tempo_ms, "casuale")
    
    
    # ============================================================
    # RISOLUZIONE CON ITERATED LOCAL SEARCH (ILS)
    # ============================================================

    # Eseguo ILS partendo da First Fit Decreasing. L'algoritmo applica la ricerca locale iniziale, quindi ripete perturbazione, ricerca locale e accettazione.

    bins_ils_iniziale, bins_ils, tempo_ils, miglioramenti_ils, peggioramenti_ils = ils(
        V, w, W, E,
        iterazioni=iterazioni_ils,
        seme=seme_ils,
        num_bin_perturbati=bin_perturbati_ils,
        probabilita_peggiori=probabilita_peggiori_ils,
        max_tentativi=tentativi_ls,
        max_mosse=mosse_fr
    )

    # Preparo il riepilogo dei risultati.
    righe += ["", "========== ITERATED LOCAL SEARCH =========="]

    # Riporto i parametri usati nell'esperimento.
    righe.append(f"Numero iterazioni ILS = {iterazioni_ils}")
    righe.append(f"Seme casuale = {seme_ils}")
    righe.append(f"Bin perturbati per iterazione = {bin_perturbati_ils}")
    righe.append(f"Probabilità accettazione peggioramenti = {probabilita_peggiori_ils:.2f}")

    # Risultati dell'esecuzione.
    righe.append(f"Soluzione iniziale FFD = {len(bins_ils_iniziale)} bin")

    righe.append(f"Migliore soluzione ILS = {len(bins_ils)} bin")

    righe.append(f"Numero di miglioramenti globali = {miglioramenti_ils}")

    righe.append(f"Numero di peggioramenti accettati = {peggioramenti_ils}")

    # Utilizzo la funzione già presente per stampare i dettagli della migliore soluzione trovata da ILS.
    righe += prepara_output_euristica("Iterated Local Search", bins_ils, V, w, W, E, tempo_ils, "FFD + perturbazione + ricerca locale")


    # Confronto: GAP rispetto al MILP SOLO se l'ottimalità è certificata.
    righe += ["", "========== CONFRONTO GAP =========="]
    righe.append(f"First Fit: {len(bins_ff)} bin, {tempo_ff:.6f} s")
    righe.append(f"Next Fit:  {len(bins_nf)} bin, {tempo_nf:.6f} s")
    righe.append(f"Multi-Start + Ricerca Locale: {len(bins_ms)} bin, {tempo_ms:.6f} s")
    righe.append(f"Iterated Local Search: {len(bins_ils)} bin, {tempo_ils:.6f} s")
    # Calcolo il GAP soltanto quando il MILP ha restituito
    # un risultato e Xpress ha certificato l'ottimalità.
    if (results is not None and results.solver.termination_condition == TerminationCondition.optimal):
        z_ottimo = pyo.value(model.obj, exception=False)
        if z_ottimo is not None and z_ottimo > 0:
            righe.append(f"MILP ottimo: {z_ottimo:.0f} bin, {tempo_milp:.3f} s")
            if not verifica_soluzione_euristica(bins_ff, V, w, W, E):
                righe.append(f"GAP First Fit = {(len(bins_ff) - z_ottimo) / z_ottimo * 100:.2f}%")
            if not verifica_soluzione_euristica(bins_nf, V, w, W, E):
                righe.append(f"GAP Next Fit  = {(len(bins_nf) - z_ottimo) / z_ottimo * 100:.2f}%")
            if not verifica_soluzione_euristica(bins_ms, V, w, W, E):
                righe.append(f"GAP Multi-Start + Ricerca Locale = {(len(bins_ms) - z_ottimo) / z_ottimo * 100:.2f}%")
            if not verifica_soluzione_euristica(bins_ils, V, w, W, E):
                righe.append(f"GAP Iterated Local Search = {(len(bins_ils) - z_ottimo) / z_ottimo * 100:.2f}%")
    else:
        if results is None:
            righe.append("MILP non risolto: benchmark MILP non disponibile; GAP non calcolato.")
        else:
            righe.append("MILP: ottimalità non certificata; GAP rispetto all'ottimo non calcolato."
            )
            # Il programma continua a confrontare il numero di bin e i tempi di First Fit e Next Fit, 
            # ma non inventa un benchmark quando Xpress non è riuscito a risolvere il modello.
   
    for riga in righe:
        print(riga)
        
    scrivi_output(righe, file_output)


if __name__ == "__main__":
    main()
