"""
Esperimento Othello 6x6 a singola posizione (non partite intere):

1) Genera N posizioni valide SIMULANDO vere partite (mosse casuali, gestione
   corretta dei "pass"), fermandosi a un numero di mosse vere scelto a caso
   nel range [MOSSE_MINIME, MOSSE_MASSIME]. A differenza di Connect4, qui non
   si puo' costruire la griglia direttamente: il legame tra numero di pedine
   e turno si rompe a causa dei pass, quindi la simulazione e' l'unico modo
   sicuro per garantire posizioni raggiungibili.

   Vengono scartate le posizioni dove:
   - la partita e' gia' finita prima di raggiungere il target
   - chi deve muovere non ha mosse legali disponibili (dovrebbe passare,
     non e' un caso utile per "qual e' la mossa migliore")
   - ci sono piu' di MASSIMO_MOSSE_OTTIMALI mosse ugualmente ottimali

2) Per ogni posizione valida, il solver esaustivo calcola la mossa
   oggettivamente ottimale.
3) Per ogni variante di prompt (Elo), chiede all'LLM di scegliere una mossa
   sulla STESSA posizione, e verifica se coincide con quella ottimale.
   Le varianti vengono processate in blocco, una alla volta; DENTRO ogni
   variante, le posizioni vengono interrogate in parallelo.
4) Salva tutto in CSV.
"""

import random
import re
import csv
import os
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from OthelloSolver import (
    griglia_iniziale,
    mosse_legali,
    gioca_mossa,
    partita_finita,
    trova_mossa_ottimale,
    DIMENSIONE,
)

load_dotenv()

# Log puliti: mostra SOLO i retry automatici veri (silenzia il rumore delle
# richieste riuscite normalmente, che altrimenti riempirebbero il terminale).
logging.basicConfig(level=logging.WARNING, format="[RETRY LOG] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)          # silenzia i "200 OK" normali
logging.getLogger("openai._base_client").setLevel(logging.INFO)  # ma mostra i veri retry

client = OpenAI(
    base_url="https://opencode.ai/zen/go/v1",
    api_key=os.getenv("OPENCODE_API_KEY"),
)

MODELLO = "mimo-v2.5"
MAX_TENTATIVI_PER_MOSSA = 3
MOSSE_MINIME = 20   # range scelto empiricamente: buon compromesso tempo/percentuale utilizzabile
MOSSE_MASSIME = 25
MASSIMO_MOSSE_OTTIMALI = 1  # esattamente 1 mossa ottimale, nessuna ambiguita'


def genera_una_posizione(mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME, seed=None):
    """Simula una vera partita, fermandosi a un numero di mosse VERE scelto
    a caso nel range. Ritorna (griglia, simbolo_da_muovere) oppure None se
    la posizione risultante non e' utilizzabile."""

    if seed is not None:
        random.seed(seed)

    numero_mosse_target = random.randint(mosse_minime, mosse_massime)
    griglia = griglia_iniziale()
    turno = 1
    mosse_vere_fatte = 0

    for _ in range(200):
        if mosse_vere_fatte >= numero_mosse_target:
            break
        if partita_finita(griglia):
            return None
        mosse = mosse_legali(griglia, turno)
        if not mosse:
            turno = 2 if turno == 1 else 1
            continue
        r, c = random.choice(mosse)
        griglia = gioca_mossa(griglia, r, c, turno)
        mosse_vere_fatte += 1
        turno = 2 if turno == 1 else 1

    if partita_finita(griglia):
        return None

    if not mosse_legali(griglia, turno):
        return None

    return griglia, turno


def genera_n_posizioni(n_posizioni, mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME):
    """Genera n_posizioni valide, ritentando quando una generazione fallisce
    o quando la posizione ha troppe mosse ugualmente ottimali."""

    posizioni = []
    tentativi = 0
    scartate_generazione = 0
    scartate_troppe_ottimali = 0

    while len(posizioni) < n_posizioni:
        tentativi += 1
        risultato = genera_una_posizione(mosse_minime, mosse_massime, seed=None)
        if risultato is None:
            scartate_generazione += 1
            continue

        griglia, simbolo_da_muovere = risultato
        avversario = 2 if simbolo_da_muovere == 1 else 1
        numero_pedine = sum(riga.count(1) + riga.count(2) for riga in griglia)

        _, valore_ottimale, punteggi_per_mossa, tempo_solver = trova_mossa_ottimale(
            griglia, simbolo_da_muovere, avversario
        )
        punteggio_massimo = max(punteggi_per_mossa.values())
        mosse_ottimali = [rc for rc, p in punteggi_per_mossa.items() if p == punteggio_massimo]

        if len(mosse_ottimali) > MASSIMO_MOSSE_OTTIMALI:
            scartate_troppe_ottimali += 1
            continue

        posizioni.append({
            "griglia": griglia,
            "simbolo_da_muovere": simbolo_da_muovere,
            "numero_pedine": numero_pedine,
            "mosse_ottimali": mosse_ottimali,
            "valore_teorico": valore_ottimale,
            "tempo_risoluzione": tempo_solver,
        })

        print(f"  Posizione {len(posizioni)}/{n_posizioni} generata "
              f"({numero_pedine} pedine, risolta in {tempo_solver:.3f}s, "
              f"mossa ottimale: {cella_a_testo(*mosse_ottimali[0])}, valore: {valore_ottimale})")

    print(f"\nGenerazione completata: {len(posizioni)} posizioni valide da {tentativi} tentativi totali "
          f"({scartate_generazione} scartate per generazione, {scartate_troppe_ottimali} per troppe mosse ottimali).")
    return posizioni


def cella_a_testo(r, c):
    return f"{chr(ord('a') + c)}{r + 1}"


def testo_a_cella(testo):
    match = re.fullmatch(r'([a-fA-F])([1-6])', testo.strip())
    if not match:
        return None
    lettera, numero = match.groups()
    return (int(numero) - 1, ord(lettera.lower()) - ord('a'))


def griglia_come_testo(griglia):
    simboli = {0: ".", 1: "B", 2: "W"}
    lettere_colonne = "  " + " ".join(chr(ord('a') + c) for c in range(DIMENSIONE))
    righe_testo = [lettere_colonne]
    for r in range(DIMENSIONE):
        riga_str = " ".join(simboli[griglia[r][c]] for c in range(DIMENSIONE))
        righe_testo.append(f"{r+1} {riga_str}")
    return "\n".join(righe_testo)


def costruisci_prompt(griglia, mosse_disp, istruzioni_stile, simbolo_llm_char):
    mosse_testo = [cella_a_testo(r, c) for r, c in mosse_disp]
    return f"""{istruzioni_stile}

Stai giocando a Othello (Reversi) su una scacchiera {DIMENSIONE}x{DIMENSIONE}.
Il tuo simbolo e' "{simbolo_llm_char}".
Ecco la griglia attuale (colonne a-f, righe 1-6):

{griglia_come_testo(griglia)}

Le mosse legali disponibili in questo momento sono: {', '.join(mosse_testo)}

Scegli una mossa tra quelle elencate sopra. Rispondi SOLO con la casella
(esempio: c4), senza altro testo, senza spiegazioni."""


def estrai_mossa(testo_risposta, mosse_legali_lista):
    celle_legali_testo = {cella_a_testo(r, c) for r, c in mosse_legali_lista}
    candidati = re.findall(r'\b([a-fA-F][1-6])\b', testo_risposta)
    validi = [c for c in candidati if c.lower() in {x.lower() for x in celle_legali_testo}]
    if not validi:
        return None
    return testo_a_cella(validi[-1])


def chiedi_mossa_llm(griglia, mosse_disp, istruzioni_stile, simbolo_llm_char, mostra_dettagli=False):
    prompt = costruisci_prompt(griglia, mosse_disp, istruzioni_stile, simbolo_llm_char)

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
            # un tentativo fallito, come una risposta non valida, e proviamo
            # ancora (fino a MAX_TENTATIVI_PER_MOSSA).
            if mostra_dettagli:
                print(f"    (tentativo {tentativo}) ERRORE API: {type(e).__name__}: {e}")
            continue

        testo = risposta.choices[0].message.content or ""
        motivo_fine = risposta.choices[0].finish_reason
        mossa = estrai_mossa(testo, mosse_disp)

        if mostra_dettagli:
            print(f"    (tentativo {tentativo}) finish_reason: {motivo_fine}")
            print(f"    (tentativo {tentativo}) LLM ha risposto: {testo.strip()!r} -> estratto: {mossa}")
            if motivo_fine == "length":
                print("    !!! ATTENZIONE: risposta troncata per limite di token (15000) !!!")

        if mossa is not None:
            return mossa, tentativo

        prompt += "\n\nLa mossa scelta non e' valida o non e' tra quelle disponibili. Riprova."

    return None, MAX_TENTATIVI_PER_MOSSA


_lock_csv = threading.Lock()


def esegui_esperimento(posizioni, varianti_prompt, file_csv="risultati_othello_posizioni.csv",
                        mostra_dettagli=False, max_worker=10):

    file_esiste_gia = os.path.exists(file_csv)
    colonne_csv = ["timestamp", "etichetta_variante", "modello", "posizione_id",
                   "numero_pedine", "mossa_ottimale", "valore_teorico",
                   "mossa_scelta_llm", "e_ottimale", "tentativi_falliti"]

    if not file_esiste_gia:
        with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=colonne_csv).writeheader()

    conteggi_per_variante = {}

    def esegui_una_posizione(pos_id, pos, istruzioni_stile):
        simbolo_llm_char = "B" if pos["simbolo_da_muovere"] == 1 else "W"
        mosse_disp = mosse_legali(pos["griglia"], pos["simbolo_da_muovere"])
        mossa_scelta, tentativi_usati = chiedi_mossa_llm(
            pos["griglia"], mosse_disp, istruzioni_stile, simbolo_llm_char, mostra_dettagli=mostra_dettagli
        )
        return pos_id, pos, mossa_scelta, tentativi_usati

    for etichetta, istruzioni_stile in varianti_prompt.items():
        print(f"\n=========== VARIANTE: {etichetta} ===========")
        print(f"Avvio {len(posizioni)} posizioni, fino a {max_worker} in parallelo...\n")

        conteggi_per_variante[etichetta] = {"ottimali": 0, "totali": 0}
        completati = 0

        with ThreadPoolExecutor(max_workers=max_worker) as pool:
            futures = [pool.submit(esegui_una_posizione, pos_id, pos, istruzioni_stile)
                       for pos_id, pos in enumerate(posizioni, start=1)]

            for future in as_completed(futures):
                pos_id, pos, mossa_scelta, tentativi_usati = future.result()

                tentativi_falliti = tentativi_usati - 1 if mossa_scelta is not None else MAX_TENTATIVI_PER_MOSSA
                e_ottimale = mossa_scelta in pos["mosse_ottimali"] if mossa_scelta is not None else False

                riga = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "etichetta_variante": etichetta,
                    "modello": MODELLO,
                    "posizione_id": pos_id,
                    "numero_pedine": pos["numero_pedine"],
                    "mossa_ottimale": cella_a_testo(*pos["mosse_ottimali"][0]),
                    "valore_teorico": pos["valore_teorico"],
                    "mossa_scelta_llm": cella_a_testo(*mossa_scelta) if mossa_scelta else None,
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
                mossa_testo = cella_a_testo(*mossa_scelta) if mossa_scelta else "NESSUNA"
                print(f"[{completati}/{len(posizioni)}] Posizione {pos_id}: LLM ha scelto {mossa_testo} "
                      f"(ottimale: {cella_a_testo(*pos['mosse_ottimali'][0])}) -> {esito}")

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
    MAX_WORKER = 10

    VARIANTI_PROMPT = {
        "elo_800": "Sei un giocatore di Othello con un rating Elo di circa 800. ",

        "elo_1800": "Sei un giocatore di Othello con un rating Elo di circa 1800. ",

        "elo_2800": "Sei un giocatore di Othello con un rating Elo di circa 2800. "
    }

    print(f"=== Generazione di {N_POSIZIONI} posizioni (range {MOSSE_MINIME}-{MOSSE_MASSIME} mosse, "
          f"max {MASSIMO_MOSSE_OTTIMALI} mosse ottimali) ===\n")
    posizioni = genera_n_posizioni(N_POSIZIONI, mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME)

    esegui_esperimento(posizioni, VARIANTI_PROMPT, file_csv="risultati_othello_posizioni.csv",
                        mostra_dettagli=False, max_worker=MAX_WORKER)