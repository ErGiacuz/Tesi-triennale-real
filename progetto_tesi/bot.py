
import chess
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

# Esempi di identificativi modello su OpenRouter (nota il prefisso provider/):
# "google/gemini-2.5-flash-lite"
# "deepseek/deepseek-chat-v3.1"
# "openai/gpt-5-mini"
# "anthropic/claude-haiku-4.5"
# "anthropic/claude-sonnet-4.6"
MODELLO = "google/gemini-2.5-flash-lite"   #deepseek/deepseek-v4-pro, z-ai/glm-5.2, xiaomi/mimo-v2.5-pro, xiaomi/mimo-v2.5, deepseek/deepseek-v4
MAX_TENTATIVI_PER_MOSSA = 3
LIMITE_MOSSE = 200


def scegli_mossa_casuale(scacchiera):
    mosse_legali = list(scacchiera.legal_moves)
    return random.choice(mosse_legali)


def costruisci_prompt(scacchiera, istruzioni_stile):
    mosse_legali = [m.uci() for m in scacchiera.legal_moves]
    return f"""{istruzioni_stile}

Stai giocando una partita a scacchi. Ecco la posizione attuale in notazione FEN:
{scacchiera.fen()}

Le mosse legali disponibili in questo momento sono:
{', '.join(mosse_legali)}

Scegli una mossa tra quelle elencate sopra. Rispondi SOLO con la mossa
in formato UCI (esempio: e2e4), senza altro testo, senza spiegazioni."""


def estrai_mossa(testo_risposta):
    """Cerca nel testo la mossa in formato UCI. Se il modello ne menziona piu' di una
    (es. si corregge a meta' frase, tipo 'c6c7... aspetta, c7d6'), prendiamo l'ULTIMA
    menzionata, che rappresenta la sua decisione finale, non il primo tentativo scartato."""
    pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
    tutte_le_mosse_trovate = re.findall(pattern, testo_risposta.lower())
    return tutte_le_mosse_trovate[-1] if tutte_le_mosse_trovate else None


def chiedi_mossa_llm(scacchiera, istruzioni_stile, mostra_dettagli=True):
    """Ritorna (mossa, numero_tentativi_usati). mossa e' None se fallisce del tutto."""
    prompt = costruisci_prompt(scacchiera, istruzioni_stile)

    for tentativo in range(1, MAX_TENTATIVI_PER_MOSSA + 1):
        risposta = client.chat.completions.create(
            model=MODELLO,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        testo = risposta.choices[0].message.content or ""
        motivo_fine = risposta.choices[0].finish_reason  # equivalente di stop_reason
        mossa_uci = estrai_mossa(testo)

        if mostra_dettagli:
            print(f"    (tentativo {tentativo}) finish_reason: {motivo_fine}")
            print(f"    (tentativo {tentativo}) LLM ha risposto: {testo.strip()!r} -> estratto: {mossa_uci}")
            if motivo_fine == "length":
                print("    !!! ATTENZIONE: risposta troncata per limite di token, probabile causa del fallimento !!!")

        if mossa_uci:
            try:
                mossa = chess.Move.from_uci(mossa_uci)
                if mossa in scacchiera.legal_moves:
                    return mossa, tentativo
            except ValueError:
                pass

        prompt += f"\n\nLa mossa '{mossa_uci}' non è valida o non è nella lista delle mosse legali. Riprova."

    return None, MAX_TENTATIVI_PER_MOSSA


def gioca_una_partita(istruzioni_stile, mostra_mosse=True):
    """Fa giocare l'LLM (bianco) contro il bot casuale (nero).
    Ritorna un dizionario con tutte le metriche della partita, inclusa
    la sequenza completa delle mosse per poterla ricostruire in seguito."""

    scacchiera = chess.Board()
    numero_mosse = 0
    tentativi_falliti_totali = 0
    sequenza_mosse = []  # lista di tutte le mosse in formato UCI, nell'ordine giocato

    while not scacchiera.is_game_over() and numero_mosse < LIMITE_MOSSE:

        mossa_llm, tentativi_usati = chiedi_mossa_llm(scacchiera, istruzioni_stile, mostra_dettagli=mostra_mosse)
        tentativi_falliti_totali += (tentativi_usati - 1)

        if mossa_llm is None:
            if mostra_mosse:
                print(f"Mossa {numero_mosse + 1}: LLM non ha prodotto una mossa valida -> PARTITA PERSA per l'LLM")
            return {
                "risultato": "0-1",
                "motivo": "MOSSA_ILLEGALE_LLM",
                "numero_mosse": numero_mosse,
                "tentativi_falliti_totali": tentativi_falliti_totali,
                "istruzioni_stile": istruzioni_stile,
                "modello": MODELLO,
                "sequenza_mosse": " ".join(sequenza_mosse),
            }

        scacchiera.push(mossa_llm)
        sequenza_mosse.append(mossa_llm.uci())
        numero_mosse += 1
        if mostra_mosse:
            print(f"Mossa {numero_mosse} (LLM): {mossa_llm.uci()}")

        if scacchiera.is_game_over():
            break

        mossa_random = scegli_mossa_casuale(scacchiera)
        scacchiera.push(mossa_random)
        sequenza_mosse.append(mossa_random.uci())
        numero_mosse += 1
        if mostra_mosse:
            print(f"Mossa {numero_mosse} (random): {mossa_random.uci()}")

    outcome = scacchiera.outcome()
    if outcome:
        risultato = outcome.result()
        motivo = outcome.termination.name
    else:
        risultato = "*"
        motivo = "LIMITE_MOSSE_RAGGIUNTO"

    return {
        "risultato": risultato,
        "motivo": motivo,
        "numero_mosse": numero_mosse,
        "tentativi_falliti_totali": tentativi_falliti_totali,
        "istruzioni_stile": istruzioni_stile,
        "modello": MODELLO,
        "sequenza_mosse": " ".join(sequenza_mosse),
    }


def gioca_n_partite(istruzioni_stile, n_partite, file_csv="risultati.csv", mostra_mosse=False, etichetta_variante=""):
    """Fa giocare n_partite di fila con le stesse istruzioni di stile,
    e aggiunge ogni risultato in coda al file CSV (lo crea se non esiste)."""

    file_esiste_gia = os.path.exists(file_csv)

    with open(file_csv, mode="a", newline="", encoding="utf-8") as f:
        colonne = ["timestamp", "etichetta_variante", "modello", "istruzioni_stile", "risultato",
                   "motivo", "numero_mosse", "tentativi_falliti_totali", "sequenza_mosse"]
        writer = csv.DictWriter(f, fieldnames=colonne)

        if not file_esiste_gia:
            writer.writeheader()

        for i in range(1, n_partite + 1):
            print(f"\n--- Partita {i}/{n_partite} ({etichetta_variante}) ---")
            dati = gioca_una_partita(istruzioni_stile, mostra_mosse=mostra_mosse)
            dati["timestamp"] = datetime.now().isoformat(timespec="seconds")
            dati["etichetta_variante"] = etichetta_variante
            writer.writerow(dati)
            f.flush()  # scrive subito su disco, cosi' non perdi dati se lo script si interrompe
            print(f"Risultato: {dati['risultato']} | Motivo: {dati['motivo']} | Mosse: {dati['numero_mosse']} | Tentativi falliti: {dati['tentativi_falliti_totali']}")


if __name__ == "__main__":

    # Ogni chiave e' un'etichetta per identificare la variante nel CSV,
    # ogni valore e' il testo esatto che va a sostituire {istruzioni_stile} nel prompt.
    VARIANTI_PROMPT = {
        "elo_800": "Sei un giocatore di scacchi debole.",
        "elo_1800": "Sei un giocatore di scacchi normale.",
        "elo_2800": "Sei un giocatore di scacchi fortissimo.",
    }

    N_PARTITE_PER_VARIANTE = 30  # numero di partite per ciascuna variante di prompt

    for etichetta, istruzioni_stile in VARIANTI_PROMPT.items():
        print(f"\n\n=========== VARIANTE: {etichetta} ===========")
        gioca_n_partite(
            istruzioni_stile,
            n_partite=N_PARTITE_PER_VARIANTE,
            file_csv="risultati.csv",
            mostra_mosse=False,
            etichetta_variante=etichetta,
        )



#"elo_800": "Sei un giocatore di scacchi con un rating Elo di circa 800. "
#           "Non ti accorgi bene degli errori dell'avversario e fatichi a "
#           "sfruttare un vantaggio anche quando ce l'hai: giochi senza un "
#           "piano chiaro per arrivare allo scacco matto.",

# "elo_1800": "Sei un giocatore di scacchi con un rating Elo di circa 1800. "
#            "Ti accorgi bene degli errori dell'avversario e sai sfruttare "
#            "un vantaggio quando ce l'hai: giochi con un piano chiaro e "
#            "porti la partita a una conclusione favorevole.",

#"elo_2800": "Sei un giocatore di scacchi con un rating Elo di circa 2800. "
#            "Ti accorgi immediatamente di ogni errore dell'avversario e "
#            "sai sfruttare al meglio qualsiasi vantaggio: giochi con un "
#            "piano chiarissimo e porti la partita a una conclusione "
#            "favorevole nel modo piu' rapido possibile.",

