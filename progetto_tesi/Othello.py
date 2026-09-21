"""
Othello (Reversi) 6x6: LLM vs bot casuale, stessa logica degli esperimenti
su scacchi e Connect4, adattata a questo gioco.

Le regole (griglia, cattura nelle 8 direzioni, gestione del "pass" quando
un giocatore non ha mosse legali) sono scritte da zero in questo file,
nella classe Othello. Non esiste una libreria standard equivalente a
python-chess per questo gioco.
"""

import random
import re
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://opencode.ai/zen/go/v1",
    api_key=os.getenv("OPENCODE_API_KEY"),
)

MODELLO = "mimo-v2.5"
MAX_TENTATIVI_PER_MOSSA = 3
DIMENSIONE = 6  # scacchiera 6x6

DIREZIONI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Othello:
    """Rappresenta una partita di Othello su scacchiera DIMENSIONE x DIMENSIONE.
    griglia[riga][colonna]: 0 = vuota, 1 = simbolo giocatore 1, 2 = simbolo giocatore 2.
    Il giocatore 1 (nero) muove per primo, per convenzione standard del gioco."""

    def __init__(self):
        self.griglia = [[0] * DIMENSIONE for _ in range(DIMENSIONE)]
        meta = DIMENSIONE // 2
        self.griglia[meta - 1][meta - 1] = 1
        self.griglia[meta][meta] = 1
        self.griglia[meta - 1][meta] = 2
        self.griglia[meta][meta - 1] = 2

    def _dentro_griglia(self, r, c):
        return 0 <= r < DIMENSIONE and 0 <= c < DIMENSIONE

    def _catture_in_direzione(self, r, c, dr, dc, simbolo):
        avversario = 2 if simbolo == 1 else 1
        catture = []
        rr, cc = r + dr, c + dc
        while self._dentro_griglia(rr, cc) and self.griglia[rr][cc] == avversario:
            catture.append((rr, cc))
            rr, cc = rr + dr, cc + dc
        if catture and self._dentro_griglia(rr, cc) and self.griglia[rr][cc] == simbolo:
            return catture
        return []

    def mossa_valida(self, r, c, simbolo):
        if self.griglia[r][c] != 0:
            return False
        for dr, dc in DIREZIONI:
            if self._catture_in_direzione(r, c, dr, dc, simbolo):
                return True
        return False

    def mosse_legali(self, simbolo):
        mosse = []
        for r in range(DIMENSIONE):
            for c in range(DIMENSIONE):
                if self.mossa_valida(r, c, simbolo):
                    mosse.append((r, c))
        return mosse

    def gioca_mossa(self, r, c, simbolo):
        self.griglia[r][c] = simbolo
        for dr, dc in DIREZIONI:
            catture = self._catture_in_direzione(r, c, dr, dc, simbolo)
            for (rr, cc) in catture:
                self.griglia[rr][cc] = simbolo

    def conta(self, simbolo):
        return sum(riga.count(simbolo) for riga in self.griglia)

    def partita_finita(self):
        return not self.mosse_legali(1) and not self.mosse_legali(2)

    def griglia_come_testo(self):
        simboli = {0: ".", 1: "B", 2: "W"}
        lettere_colonne = "  " + " ".join(chr(ord('a') + c) for c in range(DIMENSIONE))
        righe_testo = [lettere_colonne]
        for r in range(DIMENSIONE):
            riga_str = " ".join(simboli[self.griglia[r][c]] for c in range(DIMENSIONE))
            righe_testo.append(f"{r+1} {riga_str}")
        return "\n".join(righe_testo)


def scegli_mossa_casuale(mosse_legali):
    return random.choice(mosse_legali)


def cella_a_testo(r, c):
    return f"{chr(ord('a') + c)}{r + 1}"


def testo_a_cella(testo):
    match = re.fullmatch(r'([a-fA-F])([1-6])', testo.strip())
    if not match:
        return None
    lettera, numero = match.groups()
    c = ord(lettera.lower()) - ord('a')
    r = int(numero) - 1
    return (r, c)


def costruisci_prompt(partita, mosse_legali, istruzioni_stile, simbolo_llm_char):
    mosse_testo = [cella_a_testo(r, c) for r, c in mosse_legali]
    return f"""{istruzioni_stile}

Stai giocando a Othello (Reversi) su una scacchiera {DIMENSIONE}x{DIMENSIONE}.
Il tuo simbolo e' "{simbolo_llm_char}".
Ecco la griglia attuale (colonne a-f, righe 1-6):

{partita.griglia_come_testo()}

Le mosse legali disponibili in questo momento sono: {', '.join(mosse_testo)}

Scegli una mossa tra quelle elencate sopra. Rispondi SOLO con la casella
(esempio: c4), senza altro testo, senza spiegazioni."""


def estrai_mossa(testo_risposta, mosse_legali):
    celle_testo_legali = {cella_a_testo(r, c) for r, c in mosse_legali}
    candidati = re.findall(r'\b([a-fA-F][1-6])\b', testo_risposta)
    validi = [c for c in candidati if c.lower() in {x.lower() for x in celle_testo_legali}]
    if not validi:
        return None
    return testo_a_cella(validi[-1])


def chiedi_mossa_llm(partita, mosse_legali, istruzioni_stile, simbolo_llm_char, mostra_dettagli=True):
    prompt = costruisci_prompt(partita, mosse_legali, istruzioni_stile, simbolo_llm_char)

    for tentativo in range(1, MAX_TENTATIVI_PER_MOSSA + 1):
        risposta = client.chat.completions.create(
            model=MODELLO,
            max_tokens=10000,  # generoso per lasciare spazio al reasoning, che teniamo attivo
            messages=[{"role": "user", "content": prompt}],
        )
        testo = risposta.choices[0].message.content or ""
        motivo_fine = risposta.choices[0].finish_reason
        mossa = estrai_mossa(testo, mosse_legali)

        if mostra_dettagli:
            print(f"    (tentativo {tentativo}) finish_reason: {motivo_fine}")
            print(f"    (tentativo {tentativo}) LLM ha risposto: {testo.strip()!r} -> estratto: {mossa}")
            if motivo_fine == "length":
                print("    !!! ATTENZIONE: risposta troncata anche a 10000 token !!!")

        if mossa is not None:
            return mossa, tentativo

        prompt += "\n\nLa mossa scelta non e' valida o non e' tra quelle disponibili. Riprova."

    return None, MAX_TENTATIVI_PER_MOSSA


def gioca_una_partita(istruzioni_stile, mostra_mosse=True, llm_gioca_primo=True):
    partita = Othello()
    numero_mosse = 0
    numero_pass = 0
    tentativi_falliti_totali = 0
    sequenza_mosse = []
    LIMITE_MOSSE = DIMENSIONE * DIMENSIONE * 2

    simbolo_llm = 1 if llm_gioca_primo else 2
    simbolo_random = 2 if llm_gioca_primo else 1

    def esito_comune(risultato, motivo):
        return {
            "risultato": risultato,
            "motivo": motivo,
            "numero_mosse": numero_mosse,
            "numero_pass": numero_pass,
            "tentativi_falliti_totali": tentativi_falliti_totali,
            "istruzioni_stile": istruzioni_stile,
            "modello": MODELLO,
            "sequenza_mosse": " ".join(sequenza_mosse),
        }

    simbolo_di_turno = 1

    while numero_mosse < LIMITE_MOSSE:

        if partita.partita_finita():
            break

        mosse_legali = partita.mosse_legali(simbolo_di_turno)

        if not mosse_legali:
            numero_pass += 1
            sequenza_mosse.append("pass")
            if mostra_mosse:
                chi = "LLM" if simbolo_di_turno == simbolo_llm else "random"
                print(f"  ({chi}) nessuna mossa legale disponibile -> pass")
            simbolo_di_turno = 2 if simbolo_di_turno == 1 else 1
            continue

        if simbolo_di_turno == simbolo_llm:
            mossa, tentativi_usati = chiedi_mossa_llm(
                partita, mosse_legali, istruzioni_stile, "B" if simbolo_llm == 1 else "W",
                mostra_dettagli=mostra_mosse
            )
            tentativi_falliti_totali += (tentativi_usati - 1)
            if mossa is None:
                if mostra_mosse:
                    print(f"Mossa {numero_mosse + 1}: LLM non ha prodotto una mossa valida -> PARTITA PERSA per l'LLM")
                risultato = "VINCE_RANDOM"
                return esito_comune(risultato, "MOSSA_NON_VALIDA_LLM")
        else:
            mossa = scegli_mossa_casuale(mosse_legali)

        r, c = mossa
        partita.gioca_mossa(r, c, simbolo_di_turno)
        sequenza_mosse.append(cella_a_testo(r, c))
        numero_mosse += 1

        if mostra_mosse:
            chi = "LLM" if simbolo_di_turno == simbolo_llm else "random"
            print(f"Mossa {numero_mosse} ({chi}): {cella_a_testo(r, c)}")
            print(partita.griglia_come_testo())

        simbolo_di_turno = 2 if simbolo_di_turno == 1 else 1

    conteggio_llm = partita.conta(simbolo_llm)
    conteggio_random = partita.conta(simbolo_random)

    if conteggio_llm > conteggio_random:
        risultato = "VINCE_LLM"
    elif conteggio_random > conteggio_llm:
        risultato = "VINCE_RANDOM"
    else:
        risultato = "PATTA"

    return esito_comune(risultato, f"FINE_PARTITA (LLM: {conteggio_llm}, random: {conteggio_random})")


def gioca_n_partite(istruzioni_stile, n_partite, file_csv="risultati_othello.csv",
                     mostra_mosse=False, etichetta_variante="", llm_gioca_primo=True):
    file_esiste_gia = os.path.exists(file_csv)

    with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
        colonne_csv = ["timestamp", "etichetta_variante", "modello", "istruzioni_stile", "risultato",
                       "motivo", "numero_mosse", "numero_pass", "tentativi_falliti_totali", "sequenza_mosse"]
        writer = csv.DictWriter(f, fieldnames=colonne_csv)
        if not file_esiste_gia:
            writer.writeheader()

        conteggio = {"VINCE_LLM": 0, "VINCE_RANDOM": 0, "PATTA": 0}

        for i in range(1, n_partite + 1):
            print(f"\n--- Partita {i}/{n_partite} ({etichetta_variante}) ---")
            dati = gioca_una_partita(istruzioni_stile, mostra_mosse=mostra_mosse, llm_gioca_primo=llm_gioca_primo)
            dati["timestamp"] = datetime.now().isoformat(timespec="seconds")
            dati["etichetta_variante"] = etichetta_variante
            writer.writerow(dati)
            f.flush()
            conteggio[dati["risultato"]] = conteggio.get(dati["risultato"], 0) + 1
            print(f"Risultato: {dati['risultato']} | Motivo: {dati['motivo']} | Mosse: {dati['numero_mosse']} | "
                  f"Pass: {dati['numero_pass']} | Tentativi falliti: {dati['tentativi_falliti_totali']}")

        print(f"\n  >>> Riepilogo {etichetta_variante}: {conteggio['VINCE_LLM']} vittorie LLM, "
              f"{conteggio['VINCE_RANDOM']} vittorie random, {conteggio['PATTA']} patte "
              f"(su {n_partite} partite)")

    return conteggio


_lock_csv = threading.Lock()  # protegge la scrittura sul file quando piu' thread finiscono insieme


def gioca_n_partite_parallelo(istruzioni_stile, n_partite, file_csv="risultati_othello.csv",
                               etichetta_variante="", llm_gioca_primo=True, max_worker=5):
    """Come gioca_n_partite, ma fa girare piu' partite CONTEMPORANEAMENTE (fino a max_worker
    alla volta), invece di una alla volta in sequenza.

    Attenzione: mostra_mosse non e' disponibile qui (con piu' thread che stampano insieme
    l'output sarebbe illeggibile) - i risultati si vedono solo a partita completata.

    Parti con max_worker basso (es. 5) e alza gradualmente se non vedi errori di
    'troppe richieste' (rate limit) dal servizio."""

    file_esiste_gia = os.path.exists(file_csv)
    colonne_csv = ["timestamp", "etichetta_variante", "modello", "istruzioni_stile", "risultato",
                   "motivo", "numero_mosse", "numero_pass", "tentativi_falliti_totali", "sequenza_mosse"]

    if not file_esiste_gia:
        with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=colonne_csv).writeheader()

    conteggio = {"VINCE_LLM": 0, "VINCE_RANDOM": 0, "PATTA": 0}
    completate = 0

    def esegui_una_partita(indice):
        return indice, gioca_una_partita(istruzioni_stile, mostra_mosse=False, llm_gioca_primo=llm_gioca_primo)

    print(f"\n--- Avvio {n_partite} partite ({etichetta_variante}), fino a {max_worker} in parallelo ---")

    with ThreadPoolExecutor(max_workers=max_worker) as pool:
        futures = [pool.submit(esegui_una_partita, i) for i in range(1, n_partite + 1)]

        for future in as_completed(futures):
            indice, dati = future.result()
            dati["timestamp"] = datetime.now().isoformat(timespec="seconds")
            dati["etichetta_variante"] = etichetta_variante

            with _lock_csv:
                with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=colonne_csv).writerow(dati)

            conteggio[dati["risultato"]] = conteggio.get(dati["risultato"], 0) + 1
            completate += 1
            print(f"[{completate}/{n_partite}] Partita {indice} completata -> {dati['risultato']} | "
                  f"Mosse: {dati['numero_mosse']} | Pass: {dati['numero_pass']} | "
                  f"Tentativi falliti: {dati['tentativi_falliti_totali']}")

    print(f"\n  >>> Riepilogo {etichetta_variante}: {conteggio['VINCE_LLM']} vittorie LLM, "
          f"{conteggio['VINCE_RANDOM']} vittorie random, {conteggio['PATTA']} patte "
          f"(su {n_partite} partite)")

    return conteggio


if __name__ == "__main__":

    VARIANTI_PROMPT = {
        "elo_800": "Sei un giocatore di Othello con un rating Elo di circa 800. ",

        "elo_1800": "Sei un giocatore di Othello con un rating Elo di circa 1800. ",

        "elo_2800": "Sei un giocatore di Othello con un rating Elo di circa 2800. "
    }

    N_PARTITE_PER_VARIANTE = 10
    LLM_GIOCA_PRIMO = True
    MAX_WORKER = 10  # partite in parallelo - parti prudente, alza se non vedi errori di rate limit

    conteggi_per_variante = {}

    for etichetta, istruzioni_stile in VARIANTI_PROMPT.items():
        print(f"\n\n=========== VARIANTE: {etichetta} ===========")
        conteggi_per_variante[etichetta] = gioca_n_partite_parallelo(
            istruzioni_stile,
            n_partite=N_PARTITE_PER_VARIANTE,
            file_csv="risultati_othello.csv",
            etichetta_variante=etichetta,
            llm_gioca_primo=LLM_GIOCA_PRIMO,
            max_worker=MAX_WORKER,
        )

    print("\n\n=========== RIEPILOGO FINALE (tutte le varianti) ===========")
    for etichetta, conteggio in conteggi_per_variante.items():
        tot = conteggio["VINCE_LLM"] + conteggio["VINCE_RANDOM"] + conteggio["PATTA"]
        print(f"  {etichetta}: {conteggio['VINCE_LLM']} vittorie LLM, "
              f"{conteggio['VINCE_RANDOM']} vittorie random, {conteggio['PATTA']} patte (su {tot} partite)")