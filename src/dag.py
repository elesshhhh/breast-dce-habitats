"""
Bozza del DAG causale (Fase 1, da usare in Fase 6 con DoWhy).

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python src/dag.py

Cosa fa:
  1. definisce nodi e frecce del DAG (unica fonte, da importare in Fase 6);
  2. controlla che il grafo sia aciclico e stampa genitori e figli di ogni nodo;
  3. salva outputs/dag.gml (stringa GML per dowhy.CausalModel)
     e outputs/dag_bozza.png (figura per il relatore).

Scelte (concordate):
  - esposizione: sottotipo recettoriale (HR, HER2); esito: pCR;
  - mediatori in sequenza: volume tumorale -> habitat
    (sottotipo -> volume -> habitat -> pCR, e volume -> pCR);
  - eta' e coorte (I-SPY1 / I-SPY2) come confondenti da aggiustare;
  - trattamento come nodo NON osservato: dipende da sottotipo e coorte e agisce sulla pCR;
    non e' un genitore degli habitat (misurati prima della terapia);
  - frecce aggiuntive: coorte -> sottotipo (criteri di arruolamento),
    coorte -> volume (dimensione minima), coorte -> trattamento, eta' -> volume.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# nodo -> (etichetta, ruolo)
NODES = {
    "age":       ("Età", "confondente"),
    "cohort":    ("Coorte\n(I-SPY1 / I-SPY2)", "confondente"),
    "subtype":   ("Sottotipo\n(HR, HER2)", "esposizione"),
    "tum_vol":   ("Volume\ntumorale", "mediatore"),
    "habitat":   ("Habitat\n(eterogeneità)", "mediatore"),
    "treatment": ("Trattamento\n(non osservato)", "latente"),
    "pCR":       ("pCR", "esito"),
}
UNOBSERVED = {"treatment"}

EDGES = [
    # nucleo della domanda di ricerca
    ("subtype", "tum_vol"), ("subtype", "habitat"), ("subtype", "pCR"),
    ("tum_vol", "habitat"), ("tum_vol", "pCR"),
    ("habitat", "pCR"),
    # trattamento (non osservato)
    ("subtype", "treatment"), ("treatment", "pCR"),
    # confondente eta'
    ("age", "subtype"), ("age", "tum_vol"), ("age", "habitat"), ("age", "pCR"),
    # confondente coorte
    ("cohort", "subtype"), ("cohort", "tum_vol"), ("cohort", "habitat"),
    ("cohort", "treatment"), ("cohort", "pCR"),
]


def build_dag():
    g = nx.DiGraph()
    for n, (label, role) in NODES.items():
        g.add_node(n, label=label, role=role, observed="no" if n in UNOBSERVED else "yes")
    g.add_edges_from(EDGES)
    assert nx.is_directed_acyclic_graph(g), "il grafo contiene un ciclo"
    return g


def gml_string(g):
    """Stringa GML per dowhy.CausalModel(graph=...)."""
    h = nx.DiGraph()
    h.add_nodes_from((n, {"observed": d["observed"]}) for n, d in g.nodes(data=True))
    h.add_edges_from(g.edges())
    return "".join(nx.generate_gml(h))


COLORS = {"esposizione": "#4C78A8", "mediatore": "#F2A541", "esito": "#54A24B",
          "confondente": "#BAB0AC", "latente": "#FFFFFF"}
POS = {"age": (0.0, 2.6), "cohort": (0.0, -2.6), "subtype": (1.4, 0.0),
       "tum_vol": (3.3, 1.0), "habitat": (5.3, 1.0), "pCR": (7.2, 0.0),
       "treatment": (4.3, -1.3)}
# curvatura delle frecce che altrimenti passerebbero sopra altri nodi
RAD = {("age", "pCR"): -0.32, ("cohort", "pCR"): 0.32, ("age", "habitat"): -0.15,
       ("cohort", "habitat"): 0.12, ("subtype", "habitat"): 0.12}


def draw(g, path):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(-1.0, 8.2)
    ax.set_ylim(-3.6, 3.6)
    ax.axis("off")
    boxes = {}
    for n, (x, y) in POS.items():
        label, role = NODES[n]
        boxes[n] = ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=3,
                           bbox=dict(boxstyle="round,pad=0.55", fc=COLORS[role],
                                     ec="#333333", lw=1.2, ls="--" if n in UNOBSERVED else "-"))
    fig.canvas.draw()   # calcola le dimensioni dei riquadri, per tagliare le frecce sul bordo
    for (u, v) in g.edges():
        latent = u in UNOBSERVED or v in UNOBSERVED
        ax.add_patch(FancyArrowPatch(
            POS[u], POS[v], arrowstyle="-|>", mutation_scale=16, lw=1.4,
            color="#555555", linestyle="--" if latent else "-",
            connectionstyle=f"arc3,rad={RAD.get((u, v), 0.0)}",
            patchA=boxes[u].get_bbox_patch(), patchB=boxes[v].get_bbox_patch(),
            shrinkA=2, shrinkB=3, zorder=2))
    handles = [plt.Line2D([], [], marker="s", ls="", ms=12, mfc=COLORS[r], mec="#333333", label=r)
               for r in ["esposizione", "mediatore", "esito", "confondente", "latente"]]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
    ax.set_title("Bozza del DAG: sottotipo → (volume → habitat) → pCR", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    g = build_dag()
    print(f"DAG aciclico: SI  ({g.number_of_nodes()} nodi, {g.number_of_edges()} frecce)")
    print(f"Non osservati: {', '.join(sorted(UNOBSERVED))}\n")
    for n in NODES:
        print(f"{n:10s} genitori: {', '.join(g.predecessors(n)) or '-':40s} "
              f"figli: {', '.join(g.successors(n)) or '-'}")
    (OUT / "dag.gml").write_text(gml_string(g), encoding="utf-8")
    draw(g, OUT / "dag_bozza.png")
    print("\nFile salvati in outputs/: dag.gml, dag_bozza.png")


if __name__ == "__main__":
    main()
