"""
Solver esaustivo per Othello 6x6: stessa tecnica di connect4_solver.py
(negamax + alpha-beta + tabella di trasposizione), riadattata alle regole
di Othello (cattura nelle 8 direzioni, gestione del pass).
"""

import time

DIMENSIONE = 6
DIREZIONI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def griglia_iniziale():
    g = [[0] * DIMENSIONE for _ in range(DIMENSIONE)]
    m = DIMENSIONE // 2
    g[m - 1][m - 1] = 1
    g[m][m] = 1
    g[m - 1][m] = 2
    g[m][m - 1] = 2
    return g


def _dentro(r, c):
    return 0 <= r < DIMENSIONE and 0 <= c < DIMENSIONE


def _catture_direzione(griglia, r, c, dr, dc, simbolo):
    avversario = 2 if simbolo == 1 else 1
    catture = []
    rr, cc = r + dr, c + dc
    while _dentro(rr, cc) and griglia[rr][cc] == avversario:
        catture.append((rr, cc))
        rr, cc = rr + dr, cc + dc
    if catture and _dentro(rr, cc) and griglia[rr][cc] == simbolo:
        return catture
    return []


def mosse_legali(griglia, simbolo):
    mosse = []
    for r in range(DIMENSIONE):
        for c in range(DIMENSIONE):
            if griglia[r][c] != 0:
                continue
            for dr, dc in DIREZIONI:
                if _catture_direzione(griglia, r, c, dr, dc, simbolo):
                    mosse.append((r, c))
                    break
    return mosse


def gioca_mossa(griglia, r, c, simbolo):
    nuova = [riga[:] for riga in griglia]
    nuova[r][c] = simbolo
    for dr, dc in DIREZIONI:
        for (rr, cc) in _catture_direzione(griglia, r, c, dr, dc, simbolo):
            nuova[rr][cc] = simbolo
    return nuova


def partita_finita(griglia):
    return not mosse_legali(griglia, 1) and not mosse_legali(griglia, 2)


def conta(griglia, simbolo):
    return sum(riga.count(simbolo) for riga in griglia)


def griglia_a_chiave(griglia):
    return tuple(tuple(riga) for riga in griglia)


tabella_trasposizione = {}


def negamax(griglia, simbolo, avversario, alpha, beta):
    chiave = (griglia_a_chiave(griglia), simbolo)
    if chiave in tabella_trasposizione:
        return tabella_trasposizione[chiave]

    if partita_finita(griglia):
        diff = conta(griglia, simbolo) - conta(griglia, avversario)
        risultato = 1 if diff > 0 else (-1 if diff < 0 else 0)
        tabella_trasposizione[chiave] = risultato
        return risultato

    mosse = mosse_legali(griglia, simbolo)
    if not mosse:
        risultato = -negamax(griglia, avversario, simbolo, -beta, -alpha)
        tabella_trasposizione[chiave] = risultato
        return risultato

    migliore = -2
    for (r, c) in mosse:
        nuova = gioca_mossa(griglia, r, c, simbolo)
        punteggio = -negamax(nuova, avversario, simbolo, -beta, -alpha)
        if punteggio > migliore:
            migliore = punteggio
        alpha = max(alpha, migliore)
        if alpha >= beta:
            break

    tabella_trasposizione[chiave] = migliore
    return migliore


def trova_mossa_ottimale(griglia, simbolo, avversario):
    tabella_trasposizione.clear()
    inizio = time.time()

    mosse = mosse_legali(griglia, simbolo)
    risultati = {}
    for (r, c) in mosse:
        nuova = gioca_mossa(griglia, r, c, simbolo)
        risultati[(r, c)] = -negamax(nuova, avversario, simbolo, -2, 2)

    tempo = time.time() - inizio
    if not risultati:
        return None, None, {}, tempo
    migliore = max(risultati, key=risultati.get)
    return migliore, risultati[migliore], risultati, tempo


if __name__ == "__main__":
    print("Test di validazione: risoluzione della scacchiera vuota (posizione iniziale)...")
    print("Fatto noto da verificare: su 6x6, con gioco perfetto vince il SECONDO giocatore.")

    g = griglia_iniziale()
    mossa, valore, dettaglio, tempo = trova_mossa_ottimale(g, 1, 2)
    print(f"\nValore teorico per il PRIMO giocatore (nero): {valore} (atteso: -1, cioe' sconfitta)")
    print(f"Mossa migliore comunque disponibile: {mossa}")
    print(f"Tempo impiegato: {tempo:.2f} secondi")