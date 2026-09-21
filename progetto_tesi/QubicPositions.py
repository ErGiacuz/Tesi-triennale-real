"""
Esperimento Qubic (tris 3D, 4x4x4) a singola posizione:

1) Genera N posizioni valide con COSTRUZIONE DIRETTA (non simulazione):
   a differenza di Othello, qui non esistono "pass" (ogni cella vuota e'
   sempre una mossa legale), quindi qualunque assegnazione di simboli alle
   celle che rispetti i conteggi giusti per turno e' sempre raggiungibile
   con un ordine di mosse valido - possiamo costruire la griglia direttamente,
   come gia' fatto per Connect4.

   Range di mosse verificato empiricamente: 30-34 (sotto e' intrattabile per
   il solver, sopra la percentuale di posizioni ancora "vive" - non gia'
   decise - crolla rapidamente).

2) Per ogni posizione valida, il solver esaustivo calcola la mossa
   oggettivamente ottimale (validato contro una versione brute-force
   indipendente: 12/12 concordanze).
3) Per ogni variante di prompt (Elo), chiede all'LLM di scegliere una mossa
   sulla STESSA posizione. Le varianti vengono processate in blocco, una
   alla volta; DENTRO ogni variante, le posizioni vengono interrogate in
   parallelo.
4) Salva tutto in CSV.
"""

import random
import re
import csv
import os
import threading
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

_SESSION_ID_STABILE = str(uuid.uuid4())

from QubicSolver import (
    DIM,
    NUM_CELLE,
    griglia_vuota,
    celle_disponibili,
    gioca_mossa,
    controlla_vittoria,
    trova_mossa_ottimale,
)

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="[RETRY LOG] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.INFO)

client = OpenAI(
    base_url="https://opencode.ai/zen/go/v1",
    api_key=os.getenv("OPENCODE_API_KEY"),
    default_headers={"x-opencode-session": _SESSION_ID_STABILE},
)

MODELLO = "mimo-v2.5"  # <-- METTI QUI IL NOME ESATTO DEL MODELLO
MAX_TENTATIVI_PER_MOSSA = 3
MOSSE_MINIME = 36
MOSSE_MASSIME = 42
MASSIMO_MOSSE_OTTIMALI = 1


def genera_una_posizione(mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME, seed=None):
    if seed is not None:
        random.seed(seed)

    k = random.randint(mosse_minime, mosse_massime)
    if k % 2 == 0:
        conteggio_1, conteggio_2 = k // 2, k // 2
        simbolo_da_muovere = 1
    else:
        conteggio_1, conteggio_2 = (k + 1) // 2, (k - 1) // 2
        simbolo_da_muovere = 2

    tutte_le_celle = list(range(NUM_CELLE))
    celle_scelte = random.sample(tutte_le_celle, k)
    random.shuffle(celle_scelte)

    griglia = griglia_vuota()
    for i, cella in enumerate(celle_scelte):
        simbolo = 1 if i < conteggio_1 else 2
        griglia = gioca_mossa(griglia, cella, simbolo)

    if controlla_vittoria(griglia, 1) or controlla_vittoria(griglia, 2):
        return None

    return griglia, simbolo_da_muovere


import threading as _threading_timeout

TIMEOUT_SOLVER_SECONDI = 5  # se una posizione richiede piu' di cosi', la scartiamo


def trova_mossa_ottimale_con_timeout(griglia, simbolo, avversario, timeout=TIMEOUT_SOLVER_SECONDI):
    """Come trova_mossa_ottimale, ma con un tetto di tempo di sicurezza.
    Usa un thread demone: se va in timeout, il thread abbandonato continua
    a lavorare in background ma NON impedisce allo script di chiudersi
    (a differenza di un thread normale/di un ThreadPoolExecutor)."""
    risultato_contenitore = {}

    def bersaglio():
        risultato_contenitore["esito"] = trova_mossa_ottimale(griglia, simbolo, avversario)

    thread = _threading_timeout.Thread(target=bersaglio, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None  # timeout: il thread demone continuera' da solo, ma non ci blocchiamo ad aspettarlo
    return risultato_contenitore.get("esito")


def genera_n_posizioni(n_posizioni, mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME):
    posizioni = []
    tentativi = 0
    scartate_generazione = 0
    scartate_troppe_ottimali = 0
    scartate_per_timeout = 0

    while len(posizioni) < n_posizioni:
        tentativi += 1
        risultato = genera_una_posizione(mosse_minime, mosse_massime, seed=None)
        if risultato is None:
            scartate_generazione += 1
            continue

        griglia, simbolo_da_muovere = risultato
        avversario = 2 if simbolo_da_muovere == 1 else 1
        numero_pedine = sum(1 for c in griglia if c != 0)

        esito_solver = trova_mossa_ottimale_con_timeout(griglia, simbolo_da_muovere, avversario)
        if esito_solver is None:
            scartate_per_timeout += 1
            continue
        _, valore_ottimale, punteggi_per_mossa, tempo_solver = esito_solver

        punteggio_massimo = max(punteggi_per_mossa.values())
        mosse_ottimali = [cella for cella, p in punteggi_per_mossa.items() if p == punteggio_massimo]

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
              f"mossa ottimale: {cella_a_testo(mosse_ottimali[0])}, valore: {valore_ottimale})")

    print(f"\nGenerazione completata: {len(posizioni)} posizioni valide da {tentativi} tentativi totali "
          f"({scartate_generazione} scartate per vittoria gia' presente, "
          f"{scartate_per_timeout} scartate per timeout del solver (>{TIMEOUT_SOLVER_SECONDI}s), "
          f"{scartate_troppe_ottimali} per troppe mosse ottimali).")
    return posizioni


def cella_a_xyz(cella):
    z = cella // (DIM * DIM)
    resto = cella % (DIM * DIM)
    y = resto // DIM
    x = resto % DIM
    return x, y, z


def xyz_a_cella(x, y, z):
    return z * DIM * DIM + y * DIM + x


def cella_a_testo(cella):
    x, y, z = cella_a_xyz(cella)
    return f"{x+1},{y+1},{z+1}"


def testo_a_cella(testo):
    match = re.fullmatch(r'([1-4])\s*,\s*([1-4])\s*,\s*([1-4])', testo.strip())
    if not match:
        return None
    x, y, z = (int(v) - 1 for v in match.groups())
    return xyz_a_cella(x, y, z)


def griglia_come_testo(griglia):
    simboli = {0: ".", 1: "X", 2: "O"}
    blocchi = []
    for z in range(DIM):
        righe = [f"Livello z={z+1}:"]
        for y in range(DIM):
            riga = " ".join(simboli[griglia[xyz_a_cella(x, y, z)]] for x in range(DIM))
            righe.append(riga)
        blocchi.append("\n".join(righe))
    return "\n\n".join(blocchi)


def costruisci_prompt(griglia, celle_disp, istruzioni_stile, simbolo_llm_char):
    celle_testo = [cella_a_testo(c) for c in celle_disp]
    return f"""{istruzioni_stile}

Stai giocando a Qubic (tris tridimensionale) su una griglia {DIM}x{DIM}x{DIM}.
Il tuo simbolo e' "{simbolo_llm_char}". Vince chi allinea 4 simboli lungo
una qualsiasi retta (orizzontale, verticale, diagonale di un piano, o
diagonale dello spazio) in una qualsiasi delle 4 "fette" (livelli) o
attraverso di esse.

Ogni cella e' identificata da tre coordinate x,y,z (ciascuna da 1 a 4).
Ecco la griglia attuale, mostrata livello per livello (z=1..4), ogni
livello e' una griglia 4x4 nelle coordinate x (colonne) e y (righe):

{griglia_come_testo(griglia)}

Le celle disponibili in questo momento sono: {', '.join(celle_testo)}

Scegli una cella tra quelle elencate sopra. Rispondi SOLO con le coordinate
nel formato x,y,z (esempio: 2,3,1), senza altro testo, senza spiegazioni."""


def estrai_mossa(testo_risposta, celle_disp):
    celle_valide_testo = {cella_a_testo(c) for c in celle_disp}
    candidati = re.findall(r'\b([1-4]\s*,\s*[1-4]\s*,\s*[1-4])\b', testo_risposta)
    candidati_normalizzati = [re.sub(r'\s*,\s*', ',', c) for c in candidati]
    validi = [c for c in candidati_normalizzati if c in celle_valide_testo]
    if not validi:
        return None
    return testo_a_cella(validi[-1])


def chiedi_mossa_llm(griglia, celle_disp, istruzioni_stile, simbolo_llm_char, mostra_dettagli=False):
    prompt = costruisci_prompt(griglia, celle_disp, istruzioni_stile, simbolo_llm_char)

    for tentativo in range(1, MAX_TENTATIVI_PER_MOSSA + 1):
        try:
            risposta = client.chat.completions.create(
                model=MODELLO,
                max_tokens=15000,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            if mostra_dettagli:
                print(f"    (tentativo {tentativo}) ERRORE API: {type(e).__name__}: {e}")
            continue

        testo = risposta.choices[0].message.content or ""
        motivo_fine = risposta.choices[0].finish_reason
        mossa = estrai_mossa(testo, celle_disp)

        if mostra_dettagli:
            print(f"    (tentativo {tentativo}) finish_reason: {motivo_fine}")
            print(f"    (tentativo {tentativo}) LLM ha risposto: {testo.strip()!r} -> estratto: {mossa}")
            if motivo_fine == "length":
                print("    !!! ATTENZIONE: risposta troncata per limite di token (15000) !!!")

        if mossa is not None:
            return mossa, tentativo

        prompt += "\n\nLa cella scelta non e' valida o non e' tra quelle disponibili. Riprova."

    return None, MAX_TENTATIVI_PER_MOSSA


_lock_csv = threading.Lock()


def esegui_esperimento(posizioni, varianti_prompt, file_csv="risultati_qubic_posizioni.csv",
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
        simbolo_llm_char = "X" if pos["simbolo_da_muovere"] == 1 else "O"
        celle_disp = celle_disponibili(pos["griglia"])
        mossa_scelta, tentativi_usati = chiedi_mossa_llm(
            pos["griglia"], celle_disp, istruzioni_stile, simbolo_llm_char, mostra_dettagli=mostra_dettagli
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
                    "mossa_ottimale": cella_a_testo(pos["mosse_ottimali"][0]),
                    "valore_teorico": pos["valore_teorico"],
                    "mossa_scelta_llm": cella_a_testo(mossa_scelta) if mossa_scelta is not None else None,
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
                mossa_testo = cella_a_testo(mossa_scelta) if mossa_scelta is not None else "NESSUNA"
                print(f"[{completati}/{len(posizioni)}] Posizione {pos_id}: LLM ha scelto {mossa_testo} "
                      f"(ottimale: {cella_a_testo(pos['mosse_ottimali'][0])}) -> {esito}")

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
        "baseline": "",
    }

    print(f"=== Generazione di {N_POSIZIONI} posizioni (range {MOSSE_MINIME}-{MOSSE_MASSIME} mosse, "
          f"max {MASSIMO_MOSSE_OTTIMALI} mosse ottimali) ===\n")
    posizioni = genera_n_posizioni(N_POSIZIONI, mosse_minime=MOSSE_MINIME, mosse_massime=MOSSE_MASSIME)

    esegui_esperimento(posizioni, VARIANTI_PROMPT, file_csv="risultati_qubic_posizioni.csv",
                        mostra_dettagli=False, max_worker=MAX_WORKER)
