"""
Solver esaustivo per Qubic (tris 3D, 4x4x4): stessa tecnica di
connect4_solver.py e othello_solver.py (negamax + alpha-beta + tabella
di trasposizione), adattata alle regole di Qubic.

Regole: nessuna gravita', nessuna cattura - si piazza un simbolo in una
cella vuota qualsiasi, si vince allineando 4 simboli lungo una delle 76
linee vincenti (rette, diagonali di faccia, diagonali dello spazio).
"""

import time

DIM = 4
NUM_CELLE = DIM ** 3  # 64


def idx(x, y, z):
    return z * DIM * DIM + y * DIM + x


def _genera_linee_vincenti():
    direzioni_canoniche = []
    visti = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                if (-dx, -dy, -dz) in visti:
                    continue
                visti.add((dx, dy, dz))
                direzioni_canoniche.append((dx, dy, dz))

    linee = []
    for (dx, dy, dz) in direzioni_canoniche:
        for x0 in range(DIM):
            for y0 in range(DIM):
                for z0 in range(DIM):
                    cella_finale = (x0 + 3 * dx, y0 + 3 * dy, z0 + 3 * dz)
                    if all(0 <= c < DIM for c in cella_finale):
                        linea = tuple(idx(x0 + i * dx, y0 + i * dy, z0 + i * dz) for i in range(4))
                        linee.append(linea)
    return linee


LINEE_VINCENTI = _genera_linee_vincenti()
assert len(LINEE_VINCENTI) == 76  # verifica di sanita' ad ogni import


def griglia_vuota():
    return [0] * NUM_CELLE


def celle_disponibili(griglia):
    return [i for i in range(NUM_CELLE) if griglia[i] == 0]


def gioca_mossa(griglia, cella, simbolo):
    nuova = griglia[:]
    nuova[cella] = simbolo
    return nuova


def controlla_vittoria(griglia, simbolo):
    for linea in LINEE_VINCENTI:
        if all(griglia[c] == simbolo for c in linea):
            return True
    return False


def griglia_piena(griglia):
    return all(c != 0 for c in griglia)


def griglia_a_chiave(griglia):
    return tuple(griglia)


tabella_trasposizione = {}


def negamax(griglia, simbolo, avversario, alpha, beta):
    chiave = (griglia_a_chiave(griglia), simbolo)
    if chiave in tabella_trasposizione:
        return tabella_trasposizione[chiave]

    if griglia_piena(griglia):
        return 0

    mosse = celle_disponibili(griglia)

    # Controllo rapido: c'e' una mossa che vince subito?
    for c in mosse:
        prova = gioca_mossa(griglia, c, simbolo)
        if controlla_vittoria(prova, simbolo):
            tabella_trasposizione[chiave] = 1
            return 1

    migliore = -2
    for c in mosse:
        nuova = gioca_mossa(griglia, c, simbolo)
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

    mosse = celle_disponibili(griglia)
    risultati = {}
    for c in mosse:
        nuova = gioca_mossa(griglia, c, simbolo)
        if controlla_vittoria(nuova, simbolo):
            risultati[c] = 1
        else:
            risultati[c] = -negamax(nuova, avversario, simbolo, -2, 2)

    tempo = time.time() - inizio
    if not risultati:
        return None, None, {}, tempo
    migliore = max(risultati, key=risultati.get)
    return migliore, risultati[migliore], risultati, tempo


if __name__ == "__main__":
    print(f"Linee vincenti caricate: {len(LINEE_VINCENTI)}")
    print(f"Celle totali: {NUM_CELLE}")
