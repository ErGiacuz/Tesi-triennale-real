"""
Esperimento Connect 4 a singola posizione (non partite intere):

1) Genera N posizioni casuali costruendo direttamente la griglia: sceglie
   un numero totale di pedine (minimo 20), calcola quante ne servono per
   simbolo in base a chi deve muovere, e le piazza a caso rispettando solo
   la gravita' (nessuna simulazione mossa-per-mossa). Scarta le posizioni
   dove per caso e' gia' presente un 4-in-fila.
2) Per ogni posizione, usa il solver esaustivo per calcolare quali mosse
   sono oggettivamente ottimali (potrebbero essere piu' di una).
3) Per ogni variante di prompt (Elo), chiede all'LLM di scegliere una mossa
   sulla STESSA posizione, e verifica se la sua scelta e' tra quelle ottimali.
   Questa fase (le chiamate all'LLM) gira IN PARALLELO, fino a max_worker
   richieste contemporanee, perche' ogni (posizione, variante) e' indipendente
   dalle altre.
4) Salva tutto in CSV.

Le stesse posizioni vengono riusate per tutte le varianti, cosi' il confronto
tra Elo diversi non e' influenzato da posizioni piu' o meno difficili capitate
per caso a una variante piuttosto che a un'altra.
"""

import random
import re
import csv
import os
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from Connect4Solver import (
    trova_mossa_ottimale,
    gioca_mossa,
    controlla_vittoria,
    colonne_disponibili,
    RIGHE,
    COLONNE,
)

load_dotenv()

# Rende visibili i retry automatici e silenziosi del client (timeout, errori
# di connessione, errori del server) che altrimenti non vedresti mai, ma che
# contano comunque come richieste reali sulla dashboard di OpenCode.
logging.basicConfig(level=logging.INFO, format="[RETRY LOG] %(message)s")
logging.getLogger("openai._base_client").setLevel(logging.INFO)

client = OpenAI(
    base_url="https://opencode.ai/zen/go/v1",
    api_key=os.getenv("OPENCODE_API_KEY"),
)
MODELLO = "mimo-v2.5"
MAX_TENTATIVI_PER_MOSSA = 3
MOSSE_MINIME = 20  # sotto questa soglia il solver puo' occasionalmente richiedere minuti (verificato empiricamente)
MOSSE_MASSIME = 38


def genera_una_posizione(mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME, seed=None):
    """Costruisce direttamente una griglia valida, invece di simulare un'intera
    partita mossa per mossa. Sceglie un numero totale di pedine k a caso tra
    mosse_minime e mosse_massime, calcola quante pedine deve avere ciascun
    simbolo (rispettando l'alternanza dei turni: se k e' pari tocca al simbolo 1,
    se k e' dispari tocca al simbolo 2), e piazza le pedine a caso rispettando
    solo la gravita' (ogni pedina si appoggia sulla prima libera dal basso).

    Non simula i turni intermedi: controlla SOLO che la griglia finale non abbia
    gia' un 4-in-fila per nessuno dei due simboli. Se lo scarto, il chiamante
    generera' un'altra posizione."""

    if seed is not None:
        random.seed(seed)

    k = random.randint(mosse_minime, mosse_massime)
    if k % 2 == 0:
        conteggio_1, conteggio_2 = k // 2, k // 2
        simbolo_da_muovere = 1
    else:
        conteggio_1, conteggio_2 = (k + 1) // 2, (k - 1) // 2
        simbolo_da_muovere = 2

    pedine_da_piazzare = [1] * conteggio_1 + [2] * conteggio_2
    random.shuffle(pedine_da_piazzare)

    griglia = [[0] * COLONNE for _ in range(RIGHE)]

    for simbolo_pedina in pedine_da_piazzare:
        mosse = colonne_disponibili(griglia)
        if not mosse:
            return None  # non dovrebbe succedere se k <= 42, ma per sicurezza
        c = random.choice(mosse)
        griglia = gioca_mossa(griglia, c, simbolo_pedina)

    # Controllo finale: la posizione non deve avere gia' un vincitore
    if controlla_vittoria(griglia, 1) or controlla_vittoria(griglia, 2):
        return None

    return griglia, simbolo_da_muovere, None  # non teniamo traccia di una "sequenza" reale


MASSIMO_MOSSE_OTTIMALI = 2  # posizioni con piu' opzioni ottimali di questa soglia vengono scartate


def genera_n_posizioni(n_posizioni, mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME):
    """Genera n_posizioni valide, ritentando quando una generazione fallisce
    o quando la posizione ha troppe mosse ugualmente ottimali (troppo facile
    da indovinare per caso). Questa parte resta sequenziale: e' locale e
    veloce, non ha bisogno di parallelismo."""

    posizioni = []
    tentativi = 0
    scartate_per_troppe_ottimali = 0

    while len(posizioni) < n_posizioni:
        tentativi += 1
        risultato = genera_una_posizione(mosse_minime, mosse_massime, seed=None)
        if risultato is None:
            continue

        griglia, simbolo_da_muovere, _ = risultato
        avversario = 2 if simbolo_da_muovere == 1 else 1
        numero_pedine = sum(riga.count(1) + riga.count(2) for riga in griglia)

        print(f"  (tentativo {tentativi}) posizione valida trovata con {numero_pedine} pedine, "
              f"risoluzione in corso...")

        _, valore_ottimale, punteggi_per_mossa, tempo_solver = trova_mossa_ottimale(
            griglia, simbolo_da_muovere, avversario
        )
        punteggio_massimo = max(punteggi_per_mossa.values())
        colonne_ottimali = sorted(c + 1 for c, p in punteggi_per_mossa.items() if p == punteggio_massimo)

        if len(colonne_ottimali) > MASSIMO_MOSSE_OTTIMALI:
            scartate_per_troppe_ottimali += 1
            print(f"    -> scartata: {len(colonne_ottimali)} mosse ugualmente ottimali "
                  f"(soglia massima: {MASSIMO_MOSSE_OTTIMALI})")
            continue

        posizioni.append({
            "griglia": griglia,
            "simbolo_da_muovere": simbolo_da_muovere,
            "numero_mosse_giocate": numero_pedine,
            "colonne_ottimali": colonne_ottimali,
            "valore_teorico": valore_ottimale,
            "tempo_risoluzione": tempo_solver,
        })

        print(f"  Posizione {len(posizioni)}/{n_posizioni} generata "
              f"({numero_pedine} pedine, risolta in {tempo_solver:.2f}s, "
              f"mosse ottimali: {colonne_ottimali}, valore: {valore_ottimale})")

    print(f"\nGenerazione completata: {len(posizioni)} posizioni valide da {tentativi} tentativi totali "
          f"({scartate_per_troppe_ottimali} scartate per troppe mosse ottimali).")
    return posizioni


def griglia_come_testo(griglia):
    simboli = {0: " ", 1: "X", 2: "O"}
    righe_testo = []
    for r in range(RIGHE):
        riga_str = " | ".join(simboli[griglia[r][c]] for c in range(COLONNE))
        righe_testo.append("| " + riga_str + " |")
    intestazione = "  " + "   ".join(str(c + 1) for c in range(COLONNE))
    return intestazione + "\n" + "\n".join(righe_testo)


def costruisci_prompt(griglia, colonne_disp, istruzioni_stile, simbolo_llm_char):
    return f"""{istruzioni_stile}

Stai giocando a Forza 4 (Connect 4). Il tuo simbolo e' "{simbolo_llm_char}".
Ecco la griglia attuale (le righe sono numerate dall'alto in basso,
la riga piu' in basso e' quella su cui i dischi si appoggiano):

{griglia_come_testo(griglia)}

Le colonne disponibili in questo momento sono: {', '.join(map(str, colonne_disp))}

Scegli una colonna tra quelle elencate sopra per far cadere il tuo disco.
Rispondi SOLO con il numero della colonna (esempio: 4), senza altro testo,
senza spiegazioni."""


def estrai_colonna(testo_risposta, colonne_valide):
    numeri_trovati = re.findall(r'\b([1-7])\b', testo_risposta)
    numeri_validi = [int(n) for n in numeri_trovati if int(n) in colonne_valide]
    return numeri_validi[-1] if numeri_validi else None


def chiedi_mossa_llm(griglia, istruzioni_stile, simbolo_llm_char, mostra_dettagli=False):
    colonne_disp = colonne_disponibili(griglia)
    colonne_disp_1_7 = [c + 1 for c in colonne_disp]
    prompt = costruisci_prompt(griglia, colonne_disp_1_7, istruzioni_stile, simbolo_llm_char)

    for tentativo in range(1, MAX_TENTATIVI_PER_MOSSA + 1):
        try:
            risposta = client.chat.completions.create(
                model=MODELLO,
                max_tokens=15000,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            # Errore lato server/connessione che ha superato anche i retry
            # automatici della libreria (es. 500, timeout persistente).
            # Non facciamo crashare tutto lo script: contiamo questo come
            # un tentativo fallito e proviamo ancora (fino a MAX_TENTATIVI_PER_MOSSA).
            if mostra_dettagli:
                print(f"    (tentativo {tentativo}) ERRORE API: {type(e).__name__}: {e}")
            continue

        testo = risposta.choices[0].message.content or ""
        motivo_fine = risposta.choices[0].finish_reason
        colonna = estrai_colonna(testo, colonne_disp_1_7)

        if mostra_dettagli:
            print(f"    (tentativo {tentativo}) finish_reason: {motivo_fine}")
            print(f"    (tentativo {tentativo}) LLM ha risposto: {testo.strip()!r} -> estratto: {colonna}")
            if motivo_fine == "length":
                print("    !!! ATTENZIONE: risposta troncata per limite di token (15000) !!!")

        if colonna is not None:
            return colonna, tentativo

        prompt += "\n\nLa colonna scelta non e' valida o non e' tra quelle disponibili. Riprova."

    return None, MAX_TENTATIVI_PER_MOSSA


_lock_csv = threading.Lock()  # protegge la scrittura sul CSV quando piu' thread finiscono insieme


def esegui_esperimento(posizioni, varianti_prompt, file_csv="risultati_connect4_posizioni.csv",
                        mostra_dettagli=False, max_worker=10):
    """Le VARIANTI vengono processate una alla volta, in ordine (prima tutta
    elo_800, poi tutta elo_1800, ecc.) - come in Othello. DENTRO ogni variante,
    le posizioni vengono interrogate IN PARALLELO, fino a max_worker richieste
    contemporanee. Parti prudente (es. 10) e alza gradualmente se non vedi
    errori di 'troppe richieste' (rate limit)."""

    file_esiste_gia = os.path.exists(file_csv)
    colonne_csv = ["timestamp", "etichetta_variante", "modello", "posizione_id",
                   "numero_mosse_giocate",
                   "colonne_ottimali", "valore_teorico", "colonna_scelta_llm",
                   "e_ottimale", "tentativi_falliti"]

    if not file_esiste_gia:
        with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=colonne_csv).writeheader()

    conteggi_per_variante = {}

    def esegui_una_posizione(pos_id, pos, istruzioni_stile):
        simbolo_llm_char = "X" if pos["simbolo_da_muovere"] == 1 else "O"
        colonna_scelta, tentativi_usati = chiedi_mossa_llm(
            pos["griglia"], istruzioni_stile, simbolo_llm_char, mostra_dettagli=mostra_dettagli
        )
        return pos_id, pos, colonna_scelta, tentativi_usati

    for etichetta, istruzioni_stile in varianti_prompt.items():
        print(f"\n=========== VARIANTE: {etichetta} ===========")
        print(f"Avvio {len(posizioni)} posizioni, fino a {max_worker} in parallelo...\n")

        conteggi_per_variante[etichetta] = {"ottimali": 0, "totali": 0}
        completati = 0

        with ThreadPoolExecutor(max_workers=max_worker) as pool:
            futures = [pool.submit(esegui_una_posizione, pos_id, pos, istruzioni_stile)
                       for pos_id, pos in enumerate(posizioni, start=1)]

            for future in as_completed(futures):
                pos_id, pos, colonna_scelta, tentativi_usati = future.result()

                tentativi_falliti = tentativi_usati - 1 if colonna_scelta is not None else MAX_TENTATIVI_PER_MOSSA
                e_ottimale = colonna_scelta in pos["colonne_ottimali"] if colonna_scelta is not None else False

                riga = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "etichetta_variante": etichetta,
                    "modello": MODELLO,
                    "posizione_id": pos_id,
                    "numero_mosse_giocate": pos["numero_mosse_giocate"],
                    "colonne_ottimali": " ".join(map(str, pos["colonne_ottimali"])),
                    "valore_teorico": pos["valore_teorico"],
                    "colonna_scelta_llm": colonna_scelta,
                    "e_ottimale": e_ottimale,
                    "tentativi_falliti": tentativi_falliti,
                }

                with _lock_csv:
                    with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=colonne_csv).writerow(riga)

                conteggi_per_variante[etichetta]["totali"] += 1
                if e_ottimale:
                    conteggi_per_variante[etichetta]["ottimali"] += 1

                completati += 1
                esito = "OTTIMALE" if e_ottimale else "NON ottimale"
                print(f"[{completati}/{len(posizioni)}] Posizione {pos_id}: LLM ha scelto "
                      f"{colonna_scelta} (ottimali: {pos['colonne_ottimali']}) -> {esito}")

        ok = conteggi_per_variante[etichetta]["ottimali"]
        tot = conteggi_per_variante[etichetta]["totali"]
        print(f"\n  >>> Riepilogo {etichetta}: {ok}/{tot} ottimali ({100*ok/tot:.1f}%) | "
              f"{tot-ok}/{tot} non ottimali ({100*(tot-ok)/tot:.1f}%)")

    print("\n\n=========== RIEPILOGO FINALE (tutte le varianti) ===========")
    for etichetta, dati in conteggi_per_variante.items():
        ok, tot = dati["ottimali"], dati["totali"]
        if tot == 0:
            continue
        print(f"  {etichetta}: {ok}/{tot} ottimali ({100*ok/tot:.1f}%) | {tot-ok}/{tot} non ottimali "
              f"({100*(tot-ok)/tot:.1f}%)")

    return conteggi_per_variante


if __name__ == "__main__":

    N_POSIZIONI = 200
    MAX_WORKER = 10  # chiamate LLM in parallelo - alza gradualmente se non vedi errori di rate limit

    VARIANTI_PROMPT = {
        "elo_800": "Sei un giocatore di Forza 4 con un rating Elo di circa 800. ",

        "elo_1800": "Sei un giocatore di Forza 4 con un rating Elo di circa 1800. ",

        "elo_2800": "Sei un giocatore di Forza 4 con un rating Elo di circa 2800. "
    }

    print(f"=== Generazione di {N_POSIZIONI} posizioni casuali (minimo {MOSSE_MINIME} mosse giocate) ===\n")
    posizioni = genera_n_posizioni(N_POSIZIONI, mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME)

    esegui_esperimento(posizioni, VARIANTI_PROMPT, file_csv="risultati_connect4_posizioni.csv",
                        mostra_dettagli=False, max_worker=MAX_WORKER)