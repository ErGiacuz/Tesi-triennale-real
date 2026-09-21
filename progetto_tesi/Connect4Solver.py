"""
Solver esaustivo per Connect 4: per una data posizione, determina il
valore teorico del gioco (vittoria/patta/sconfitta per chi deve muovere)
e la mossa ottimale, tramite ricerca alpha-beta con tabella di trasposizione.

A differenza di un minimax euristico, questo esplora l'albero di gioco
fino alla fine (vittoria/patta), quindi il risultato e' garantito corretto,
non una stima.
"""

import time

RIGHE = 6
COLONNE = 7


def colonne_disponibili(griglia):
    return [c for c in range(COLONNE) if griglia[0][c] == 0]


def gioca_mossa(griglia, colonna, simbolo):
    """Ritorna una NUOVA griglia con la mossa applicata (non modifica l'originale)."""
    nuova = [riga[:] for riga in griglia]
    for r in range(RIGHE - 1, -1, -1):
        if nuova[r][colonna] == 0:
            nuova[r][colonna] = simbolo
            return nuova
    return None  # colonna piena, non dovrebbe succedere se controlli prima


def controlla_vittoria(griglia, simbolo):
    g = griglia
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


def griglia_piena(griglia):
    return all(griglia[0][c] != 0 for c in range(COLONNE))


def griglia_to_chiave(griglia):
    """Converte la griglia in una tupla immutabile, usabile come chiave nella
    tabella di trasposizione (dizionario)."""
    return tuple(tuple(riga) for riga in griglia)


# Ordine di esplorazione delle colonne: centro prima, poi via via verso i bordi.
# Questo NON cambia il risultato finale, ma accelera moltissimo l'alpha-beta,
# perche' le mosse centrali sono quasi sempre le piu' forti e permettono
# di "tagliare" prima i rami peggiori.
ORDINE_COLONNE = [3, 2, 4, 1, 5, 0, 6]

tabella_trasposizione = {}


def negamax(griglia, simbolo, avversario, profondita_rimanente, alpha, beta):
    """Ritorna il punteggio della posizione dal punto di vista di 'simbolo'
    (chi deve muovere ora): +1 = vittoria garantita, 0 = patta, -1 = sconfitta.
    'profondita_rimanente' e' il numero di caselle ancora libere."""

    chiave = (griglia_to_chiave(griglia), simbolo)
    if chiave in tabella_trasposizione:
        return tabella_trasposizione[chiave]

    if griglia_piena(griglia):
        return 0  # patta

    mosse = [c for c in ORDINE_COLONNE if griglia[0][c] == 0]

    # Controllo rapido: c'e' una mossa che vince subito?
    for c in mosse:
        prova = gioca_mossa(griglia, c, simbolo)
        if controlla_vittoria(prova, simbolo):
            tabella_trasposizione[chiave] = 1
            return 1

    miglior_punteggio = -2  # peggio di qualsiasi punteggio possibile (-1)
    for c in mosse:
        nuova_griglia = gioca_mossa(griglia, c, simbolo)
        punteggio = -negamax(nuova_griglia, avversario, simbolo, profondita_rimanente - 1, -beta, -alpha)
        if punteggio > miglior_punteggio:
            miglior_punteggio = punteggio
        alpha = max(alpha, miglior_punteggio)
        if alpha >= beta:
            break  # taglio alpha-beta

    tabella_trasposizione[chiave] = miglior_punteggio
    return miglior_punteggio


def trova_mossa_ottimale(griglia, simbolo, avversario):
    """Ritorna (mossa_ottimale_0_indexed, valore_teorico, dettaglio_per_mossa, tempo_impiegato).
    valore_teorico e' dal punto di vista di 'simbolo': 1=vince, 0=patta, -1=perde
    (assumendo l'avversario giochi altrettanto perfettamente)."""

    tabella_trasposizione.clear()  # puliamo tra una chiamata e l'altra per non usare troppa memoria
    inizio = time.time()

    mosse = colonne_disponibili(griglia)
    profondita = sum(riga.count(0) for riga in griglia)

    risultati_per_mossa = {}
    migliore_mossa = None
    miglior_punteggio = -2

    for c in ORDINE_COLONNE:
        if c not in mosse:
            continue
        nuova_griglia = gioca_mossa(griglia, c, simbolo)
        if controlla_vittoria(nuova_griglia, simbolo):
            punteggio = 1
        else:
            punteggio = -negamax(nuova_griglia, avversario, simbolo, profondita - 1, -2, 2)
        risultati_per_mossa[c] = punteggio
        if punteggio > miglior_punteggio:
            miglior_punteggio = punteggio
            migliore_mossa = c

    tempo_impiegato = time.time() - inizio
    return migliore_mossa, miglior_punteggio, risultati_per_mossa, tempo_impiegato


if __name__ == "__main__":
    # TEST DI VALIDAZIONE: scacchiera vuota.
    # Fatto noto e documentato: con gioco perfetto, il PRIMO giocatore vince
    # sempre, e la mossa ottimale di apertura e' la colonna centrale (indice 3,
    # cioe' la "colonna 4" in numerazione 1-7).
    griglia_vuota = [[0] * COLONNE for _ in range(RIGHE)]

    print("Test di validazione: risoluzione della scacchiera vuota...")
    print("(potrebbe richiedere del tempo: e' il caso piu' difficile in assoluto)")
    mossa, valore, dettaglio, tempo = trova_mossa_ottimale(griglia_vuota, simbolo=1, avversario=2)

    print(f"\nMossa ottimale (0-indexed): {mossa} -> colonna {mossa + 1} in numerazione 1-7")
    print(f"Valore teorico per chi muove per primo: {valore} (atteso: 1, cioe' vittoria)")
    print(f"Tempo impiegato: {tempo:.2f} secondi")
    print(f"Dettaglio per ogni mossa (0-indexed -> valore): {dettaglio}")