"""
Connect 4 (Forza 4): LLM vs bot casuale, stessa logica dell'esperimento
sugli scacchi ma adattata a questo gioco.

Non esiste una libreria standard come python-chess per Connect 4, quindi
le regole (griglia, caduta dei dischi, controllo vittoria) sono scritte
da zero in questo file, nella classe Connect4.
"""

import random
import re
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODELLO = "google/gemini-2.5-flash-lite"   # cambia qui per provare un altro modello
MAX_TENTATIVI_PER_MOSSA = 3
RIGHE = 6
COLONNE = 7


class Connect4:
    """Rappresenta una partita di Connect 4. La griglia e' una lista di liste:
    griglia[riga][colonna], con riga 0 in alto e riga RIGHE-1 in basso.
    Le celle vuote sono " ", i dischi sono "X" (LLM) o "O" (random)."""

    def __init__(self):
        self.griglia = [[" " for _ in range(COLONNE)] for _ in range(RIGHE)]
        self.ultima_mossa = None

    def colonne_disponibili(self):
        """Ritorna la lista delle colonne (numerate 1-7) non ancora piene."""
        return [c + 1 for c in range(COLONNE) if self.griglia[0][c] == " "]

    def gioca_mossa(self, colonna_1_7, simbolo):
        """Fa cadere un disco 'simbolo' nella colonna indicata (1-7).
        Ritorna True se applicata, False se la colonna era piena."""
        c = colonna_1_7 - 1
        if c < 0 or c >= COLONNE or self.griglia[0][c] != " ":
            return False
        for r in range(RIGHE - 1, -1, -1):
            if self.griglia[r][c] == " ":
                self.griglia[r][c] = simbolo
                self.ultima_mossa = (r, c)
                return True
        return False

    def griglia_come_testo(self):
        righe_testo = []
        for r in range(RIGHE):
            righe_testo.append("| " + " | ".join(self.griglia[r]) + " |")
        intestazione = "  " + "   ".join(str(c + 1) for c in range(COLONNE))
        return intestazione + "\n" + "\n".join(righe_testo)

    def controlla_vittoria(self, simbolo):
        g = self.griglia

        for r in range(RIGHE):
            for c in range(COLONNE - 3):
                if all(g[r][c + i] == simbolo for i in range(4)):
                    return True
        for c in range(COLONNE):
            for r in range(RIGHE - 3):
                if all(g[r + i][c] == simbolo for i in range(4)):
                    return True
        for r in range(RIGHE - 3):
            for c in range(COLONNE - 3):
                if all(g[r + i][c + i] == simbolo for i in range(4)):
                    return True
        for r in range(3, RIGHE):
            for c in range(COLONNE - 3):
                if all(g[r - i][c + i] == simbolo for i in range(4)):
                    return True
        return False

    def griglia_piena(self):
        return all(self.griglia[0][c] != " " for c in range(COLONNE))


def scegli_colonna_casuale(partita):
    return random.choice(partita.colonne_disponibili())


def costruisci_prompt(partita, istruzioni_stile, simbolo_llm):
    colonne_disponibili = partita.colonne_disponibili()
    return f"""{istruzioni_stile}

Stai giocando a Forza 4 (Connect 4). Il tuo simbolo e' "{simbolo_llm}".
Ecco la griglia attuale (le righe sono numerate dall'alto in basso,
la riga piu' in basso e' quella su cui i dischi si appoggiano):

{partita.griglia_come_testo()}

Le colonne disponibili in questo momento sono: {', '.join(map(str, colonne_disponibili))}

Scegli una colonna tra quelle elencate sopra per far cadere il tuo disco.
Rispondi SOLO con il numero della colonna (esempio: 4), senza altro testo,
senza spiegazioni."""


def estrai_colonna(testo_risposta, colonne_valide):
    """Prendiamo l'ULTIMO numero valido menzionato nel testo, che rappresenta
    la decisione finale del modello anche se si e' corretto a meta' frase."""
    numeri_trovati = re.findall(r'\b([1-7])\b', testo_risposta)
    numeri_validi = [int(n) for n in numeri_trovati if int(n) in colonne_valide]
    return numeri_validi[-1] if numeri_validi else None


def chiedi_mossa_llm(partita, istruzioni_stile, simbolo_llm, mostra_dettagli=True):
    """Ritorna (colonna, numero_tentativi_usati). colonna e' None se fallisce del tutto."""
    prompt = costruisci_prompt(partita, istruzioni_stile, simbolo_llm)
    colonne_valide = partita.colonne_disponibili()

    for tentativo in range(1, MAX_TENTATIVI_PER_MOSSA + 1):
        risposta = client.chat.completions.create(
            model=MODELLO,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        testo = risposta.choices[0].message.content or ""
        motivo_fine = risposta.choices[0].finish_reason
        colonna = estrai_colonna(testo, colonne_valide)

        if mostra_dettagli:
            print(f"    (tentativo {tentativo}) finish_reason: {motivo_fine}")
            print(f"    (tentativo {tentativo}) LLM ha risposto: {testo.strip()!r} -> estratto: {colonna}")
            if motivo_fine == "length":
                print("    !!! ATTENZIONE: risposta troncata per limite di token !!!")

        if colonna is not None:
            return colonna, tentativo

        prompt += "\n\nLa colonna scelta non e' valida o non e' tra quelle disponibili. Riprova."

    return None, MAX_TENTATIVI_PER_MOSSA


def gioca_una_partita(istruzioni_stile, mostra_mosse=True, llm_gioca_primo=True):
    """Fa giocare l'LLM contro il bot casuale a Connect 4.
    Se llm_gioca_primo=True, l'LLM ha il simbolo 'X' e muove per primo.
    Se llm_gioca_primo=False, il bot casuale ('X') muove per primo, e l'LLM e' 'O'."""

    partita = Connect4()
    numero_mosse = 0
    tentativi_falliti_totali = 0
    sequenza_mosse = []
    LIMITE_MOSSE = RIGHE * COLONNE  # 42: la griglia si riempie al massimo dopo questo numero di mosse

    simbolo_llm = "X" if llm_gioca_primo else "O"
    simbolo_random = "O" if llm_gioca_primo else "X"

    if llm_gioca_primo:
        ordine_turni = [("LLM", simbolo_llm), ("random", simbolo_random)]
    else:
        ordine_turni = [("random", simbolo_random), ("LLM", simbolo_llm)]

    def esito_comune(risultato, motivo):
        return {
            "risultato": risultato,
            "motivo": motivo,
            "numero_mosse": numero_mosse,
            "tentativi_falliti_totali": tentativi_falliti_totali,
            "istruzioni_stile": istruzioni_stile,
            "modello": MODELLO,
            "sequenza_mosse": " ".join(sequenza_mosse),
        }

    while numero_mosse < LIMITE_MOSSE:
        for chi_muove, simbolo in ordine_turni:

            if chi_muove == "LLM":
                colonna, tentativi_usati = chiedi_mossa_llm(partita, istruzioni_stile, simbolo_llm, mostra_dettagli=mostra_mosse)
                tentativi_falliti_totali += (tentativi_usati - 1)
                if colonna is None:
                    if mostra_mosse:
                        print(f"Mossa {numero_mosse + 1}: LLM non ha prodotto una colonna valida -> PARTITA PERSA per l'LLM")
                    return esito_comune("PERDE_LLM", "COLONNA_NON_VALIDA_LLM")
            else:
                colonna = scegli_colonna_casuale(partita)

            partita.gioca_mossa(colonna, simbolo)
            sequenza_mosse.append(str(colonna))
            numero_mosse += 1
            if mostra_mosse:
                print(f"Mossa {numero_mosse} ({chi_muove}, {simbolo}): colonna {colonna}")
                print(partita.griglia_come_testo())

            if partita.controlla_vittoria(simbolo):
                vincitore = "LLM" if simbolo == simbolo_llm else "random"
                if mostra_mosse:
                    print(f"Vittoria di {vincitore} (simbolo {simbolo})!")
                risultato = "VINCE_LLM" if vincitore == "LLM" else "VINCE_RANDOM"
                return esito_comune(risultato, "QUATTRO_IN_LINEA")

            if partita.griglia_piena():
                if mostra_mosse:
                    print("Griglia piena, partita patta.")
                return esito_comune("PATTA", "GRIGLIA_PIENA")

    return esito_comune("PATTA", "LIMITE_MOSSE_RAGGIUNTO")


def gioca_n_partite(istruzioni_stile, n_partite, file_csv="risultati_connect4.csv",
                     mostra_mosse=False, etichetta_variante="", llm_gioca_primo=True):
    """Fa giocare n_partite di fila con le stesse istruzioni di stile,
    e aggiunge ogni risultato in coda al file CSV (lo crea se non esiste)."""

    file_esiste_gia = os.path.exists(file_csv)

    with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
        colonne_csv = ["timestamp", "etichetta_variante", "modello", "istruzioni_stile", "risultato",
                       "motivo", "numero_mosse", "tentativi_falliti_totali", "sequenza_mosse"]
        writer = csv.DictWriter(f, fieldnames=colonne_csv)

        if not file_esiste_gia:
            writer.writeheader()

        for i in range(1, n_partite + 1):
            print(f"\n--- Partita {i}/{n_partite} ({etichetta_variante}) ---")
            dati = gioca_una_partita(istruzioni_stile, mostra_mosse=mostra_mosse, llm_gioca_primo=llm_gioca_primo)
            dati["timestamp"] = datetime.now().isoformat(timespec="seconds")
            dati["etichetta_variante"] = etichetta_variante
            writer.writerow(dati)
            f.flush()
            print(f"Risultato: {dati['risultato']} | Motivo: {dati['motivo']} | Mosse: {dati['numero_mosse']} | Tentativi falliti: {dati['tentativi_falliti_totali']}")


if __name__ == "__main__":

    VARIANTI_PROMPT = {
         "elo_800": "Sei un giocatore di Forza 4 con un rating Elo di circa 800. "
                   "Non ti accorgi bene delle minacce dell'avversario e fatichi "
                   "a costruire una strategia chiara per allineare 4 dischi.",
 
        "elo_1800": "Sei un giocatore di Forza 4 con un rating Elo di circa 1800. "
                    "Riconosci bene le minacce dell'avversario e costruisci le tue "
                    "mosse con un piano chiaro per allineare 4 dischi.",
 
        "elo_2800": "Sei un giocatore di Forza 4 con un rating Elo di circa 2800. "
                    "Riconosci immediatamente ogni minaccia dell'avversario e giochi "
                    "con un piano chiarissimo per allineare 4 dischi nel modo piu' rapido possibile.",
    }

    N_PARTITE_PER_VARIANTE = 60
    LLM_GIOCA_PRIMO = False   # True = l'LLM muove per primo (simbolo X)

    for etichetta, istruzioni_stile in VARIANTI_PROMPT.items():
        print(f"\n\n=========== VARIANTE: {etichetta} ===========")
        gioca_n_partite(
            istruzioni_stile,
            n_partite=N_PARTITE_PER_VARIANTE,
            file_csv="risultati_connect4.csv",
            mostra_mosse=False,
            etichetta_variante=etichetta,
            llm_gioca_primo=LLM_GIOCA_PRIMO,
        )