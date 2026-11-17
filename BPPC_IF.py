import json
import time
import math
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
    
    # Convertire gli identificativi degli oggetti in interi e i pesi in float
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
def verifica_soluzione(model, results, tempo, time_limit):
    status = results.solver.status
    termination_condition = results.solver.termination_condition
    
    z = pyo.value(model.obj, exception=False) # ottengo il valore della funzione obiettivo
    
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
    messaggio, disponibile = verifica_soluzione(model, results, tempo, time_limit)
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
        if y_value > 0.5:
            oggetti_bin = []
            carico_bin = 0
            for i in model.V:
                x_value = pyo.value(model.x[i, j], exception=False)
                f_value = pyo.value(model.f[i, j], exception=False)
                if x_value > 0.5:
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
    with open(nome_file, "w") as f:
        for riga in righe:
            f.write(riga + "\n")


def main():
    # Caricamento dell'istanza
    file_input = "istanza_01_SI.json"
    V, w, W, E, U = leggi_istanza(file_input)
    time_limit = 300  # Limite di tempo in secondi (5 minuti)
    file_output = "risultati_BPPC-IF.txt"  # Nome del file di output

    print("========== DATI DELL'ISTANZA ==========")
    print("Oggetti:", V)
    print("Pesi:", w)
    print("Capacità dei bin:", W)
    print("Coppie in conflitto:", E)
    print("Numero massimo di bin:", U)

    # Costruzione del modello
    model = crea_modello(V, w, W, E, U)

    # Risoluzione
    results, tempo = risolvi_modello(model, time_limit)

    # Stampa della soluzione
    righe = prepara_output(model, results, tempo, time_limit)
    
    for riga in righe:
        print(riga)
        
    scrivi_output(righe, file_output)
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
        
        
    