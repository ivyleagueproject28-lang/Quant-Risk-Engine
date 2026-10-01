"""report.py - Report PDF (fpdf2). I font base di FPDF supportano solo latin-1."""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

NAVY = (24, 43, 73)
NEXT = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)   # a capo dopo la cella
RIGHT = dict(new_x=XPos.RIGHT, new_y=YPos.TOP)     # resta sulla stessa riga


def _t(s) -> str:
    """Rende sicuro qualsiasi testo per i font latin-1."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


class InstitutionalPDF(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 3)
        self.cell(0, 8, "QUANTITATIVE RISK & PORTFOLIO ENGINE - REPORT TECNICO",
                  align="L")
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def _section(pdf, title):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, _t(title), **NEXT)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)


def _metrics_table(pdf, metrics, bench_name):
    pct = lambda x: f"{x * 100:.2f}%"
    rows = [
        ("Rendimento annualizzato", pct(metrics["ret"]), "Volatilita annualizzata", pct(metrics["vol"])),
        ("Sharpe Ratio", f"{metrics['sharpe']:.2f}", "Sortino Ratio", f"{metrics['sortino']:.2f}"),
        (f"Alpha di Jensen (vs {bench_name})", pct(metrics["alpha"]), "Beta", f"{metrics['beta']:.2f}"),
        ("Max Drawdown", pct(metrics["max_dd"]), "Calmar Ratio", f"{metrics['calmar']:.2f}"),
        ("VaR 95% (1 giorno)", pct(metrics["var_95"]), "CVaR 95% (1 giorno)", pct(metrics["cvar_95"])),
        ("VaR 99% (1 giorno)", pct(metrics["var_99"]), "CVaR 99% (1 giorno)", pct(metrics["cvar_99"])),
    ]
    pdf.set_text_color(40, 40, 40)
    for a, b, c, d in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(55, 6, _t(a + ":"), **RIGHT)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(35, 6, _t(b), **RIGHT)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 6, _t(c + ":"), **RIGHT)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(40, 6, _t(d), **NEXT)
    pdf.ln(4)


def generate_pdf_report(tickers, weights, metrics_eval, metrics_in, method,
                        bench_name, validated, eval_label):
    pdf = InstitutionalPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "Analisi di Portafoglio e Gestione del Rischio", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, _t(f"Algoritmo: {method}"), **NEXT)
    pdf.cell(0, 6, _t(f"Validazione: {'out-of-sample (train/test)' if validated else 'NESSUNA (tutti i dati in-sample)'}"), **NEXT)
    pdf.ln(4)

    _section(pdf, f"1. Metriche - {eval_label}")
    _metrics_table(pdf, metrics_eval, bench_name)

    if validated:
        _section(pdf, "2. Metriche in-sample (periodo di ottimizzazione)")
        _metrics_table(pdf, metrics_in, bench_name)
        n = 3
    else:
        n = 2

    _section(pdf, f"{n}. Allocazione del capitale")
    pdf.set_fill_color(240, 243, 246)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(90, 7, " Asset", border=1, fill=True, **RIGHT)
    pdf.cell(90, 7, " Peso (%)", border=1, fill=True, **NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for t, w in zip(tickers, weights):
        pdf.cell(90, 6, _t(f" {t}"), border=1, **RIGHT)
        pdf.cell(90, 6, f" {w * 100:.2f}%", border=1, **NEXT)
    pdf.ln(6)

    _section(pdf, f"{n + 1}. Limiti del modello")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    notes = [
        "I rendimenti attesi stimati sul passato sono molto rumorosi: i pesi ottimali sono instabili.",
        "VaR e CVaR sono storici: assumono che il futuro assomigli al campione osservato.",
        "I titoli analizzati sono stati scelti a posteriori (survivorship/selection bias).",
        "I costi di transazione sono un'approssimazione proporzionale al turnover.",
        "Le performance passate non garantiscono risultati futuri.",
    ]
    for line in notes:
        pdf.multi_cell(0, 5, _t("- " + line), **NEXT)

    return bytes(pdf.output())
