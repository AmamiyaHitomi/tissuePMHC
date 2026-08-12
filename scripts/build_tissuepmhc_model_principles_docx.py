from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(__file__).resolve().parents[1] / "results" / "tissuePMHC_model_principles_guide.docx"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "1F2937"
MUTED = "667085"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F4F6F9"
PAGE_WIDTH_DXA = 9360


def set_run_font(run, size=None, bold=None, color=None, italic=None, name="Microsoft YaHei"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for grid_col in list(grid):
        grid.remove(grid_col)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        grid.append(grid_col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_mar = tc_pr.first_child_found_in("w:tcMar")
            if tc_mar is None:
                tc_mar = OxmlElement("w:tcMar")
                tc_pr.append(tc_mar)
            for side, value in (("top", "80"), ("bottom", "80"), ("start", "120"), ("end", "120")):
                node = tc_mar.find(qn(f"w:{side}"))
                if node is None:
                    node = OxmlElement(f"w:{side}")
                    tc_mar.append(node)
                node.set(qn("w:w"), value)
                node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def paragraph(doc, text="", *, size=11, bold=False, color=INK, before=0, after=6, line=1.25, align=None, italic=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold, color=color, italic=italic)
    return p


def labeled_paragraph(doc, label, text):
    p = paragraph(doc, "", after=5)
    r = p.add_run(label)
    set_run_font(r, size=11, bold=True, color=DARK_BLUE)
    r = p.add_run(text)
    set_run_font(r, size=11, color=INK)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if level == 1:
        pf.space_before, pf.space_after = Pt(18), Pt(10)
        size, color = 16, BLUE
    elif level == 2:
        pf.space_before, pf.space_after = Pt(14), Pt(7)
        size, color = 13, BLUE
    else:
        pf.space_before, pf.space_after = Pt(10), Pt(5)
        size, color = 12, DARK_BLUE
    r = p.add_run(text)
    set_run_font(r, size=size, bold=True, color=color)
    return p


def callout(doc, title, body, fill=LIGHT_GRAY):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [PAGE_WIDTH_DXA])
    cell = table.cell(0, 0)
    shade(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    set_run_font(r, size=11, bold=True, color=DARK_BLUE)
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(body)
    set_run_font(r, size=10.5, color=INK)
    paragraph(doc, "", after=3)


def fill_table_cell(cell, text, *, header=False, center=False, size=9.4):
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.1
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    set_run_font(r, size=size, bold=header, color="FFFFFF" if header else INK)
    if header:
        shade(cell, DARK_BLUE)


def configure_document(doc):
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    r = header.add_run("TissuePMHC Modeling Note")
    set_run_font(r, size=8.5, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("From E0 to E29: Shared Structures, Integration Policy and CNN")
    set_run_font(r, size=8.5, color=MUTED)


def add_stage_table(doc):
    heading(doc, "You know, at first glance, where the key uplift comes from.", 1)
    paragraph(doc, "Only the stages of a real change of course or a significant improvement in performance are listed below.", size=10.5, color=MUTED, after=6)
    table = doc.add_table(rows=1, cols=4)
    set_table_geometry(table, [700, 2250, 1250, 5160])
    headers = ["Phase", "Model", "Mean AUROC", "What's it solve?"]
    for cell, text in zip(table.rows[0].cells, headers):
        fill_table_cell(cell, text, header=True, center=(text != "What's it solve?"))
    set_repeat_table_header(table.rows[0])
    rows = [
        ("E0", "Traditional one-hot linear model", "0.7558", "Provides a baseline for sequence location characteristics, but cannot naturally learn local motif, task sharing or HLA hierarchy."),
        ("E2", "Share peptide encoder + task heads", "0.7927", "Let 44 missions to share the basic knowledge of peptide while retaining the output of each task."),
        ("E8", "Global + HLA Dual Branch", "0.8050", "Model global patterns separately from those of the same HLA, and then blend softly."),
        ("E14", "Auxiliary global + plain HLA", "0.8116", "Use the Tissue/HLA subsidiary supervision to enhance global representation while avoiding interference with the specialization of HLA branches."),
        ("E15", "Task-rank fusion", "0.8130", "To re-order the two branches into a bit more even than the probability scale."),
        ("E17", "E14a 5-seed ensemble", "0.8263", "Average independent training models weaken the occasional error of the initialization and training paths."),
        ("E29", "Multi-kernel CNN E14a 3-seed", "0.8341", "A visible extraction of local motifs of 2, 3, 5 amino acid lengths and retention of a strong two-branch structure of E14."),
    ]
    for stage, model, auroc, contribution in rows:
        cells = table.add_row().cells
        fill_table_cell(cells[0], stage, center=True)
        fill_table_cell(cells[1], model)
        fill_table_cell(cells[2], auroc, center=True)
        fill_table_cell(cells[3], contribution)
    paragraph(doc, "Current main results: E29 3-seed ensemble, Mean AUROC 0.8341, Mean AUPPC 0.8228, World-10 Mean AUROC 0.7634.", size=10.5, bold=True, color=DARK_BLUE, before=5, after=5)


def build():
    doc = Document()
    configure_document(doc)

    p = paragraph(doc, "TissuePMHC Key Model Rationale Learning Guide", size=24, bold=True, color=DARK_BLUE, before=10, after=3)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = paragraph(doc, "From traditional linear models to E29 Multi-kernel CNN 3-seed Esmble", size=13, color=MUTED, after=16)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    callout(
        doc,
        "Remember this first:",
        "The main upgrade of the project is not from complicating the optimist, but from three things: a rational sharing of mission knowledge, the retention of HLA-specific information, and a local sequence model that allows models to better recognize peptide.",
        fill=LIGHT_BLUE,
    )
    paragraph(doc, "Suggested reading order: first E2, E8, and E14 to understand the structure; then E15 and E17 to understand rank fusion and ensembling; finally E29 to understand why the strongest model works.", size=10.5, color=MUTED, after=10)

    add_stage_table(doc)

    heading(doc, "What is the mission?", 1)
    paragraph(doc, "Enter a peptide sequence with a length of 9. Each forecast task is defined by \"tassue + HLA allele\": the model will determine whether this peptide is more like a sample under the tessue-HLA conditions. There are 44 tasks.")
    callout(doc, "A visual metaphor.", "Consider each task as a different laboratory criterion: they're looking at peptide, but the focus is not exactly the same. Some patterns are common across all laboratories, some are established only within a particular class of HLA.", fill=LIGHT_GRAY)

    heading(doc, "E0: Traditional linear models - only see what's in each location", 1)
    paragraph(doc, "E0 Expands each location and amino acid of 9-mer into one-hot features, and scores with Logist Repression. It learns independent location effects like \"Big 2 is a better one\".")
    labeled_paragraph(doc, "What's it good at:", "It is fast, understandable and can establish a clear traditional baseline.")
    labeled_paragraph(doc, "What it lacks:", "It's not good at understanding the adjoining amino acid combinations, and it's not able to share experience with different missions. It's more like a position-amino acid score table.")

    heading(doc, "E2: Sharing peptide encoder - 44 tasks to learn together", 1)
    paragraph(doc, "E2 Instead of training complete models for 44 tasks, the task will be shared with all tasks. This encoder will compress the task into a \"indicator\" and then leave it to the mission's own output header to judge.")
    callout(doc, "Structures", "Shares 44 task-special headers. Shares share learns general peptide; how each head learning mission uses these.", fill=LIGHT_BLUE)
    labeled_paragraph(doc, "Why is it working?", "Data for individual tasks are limited, but the peptide mode is common in different tasks. Sharing encoder is equivalent to a set of basics for all tasks to accumulate together.")
    labeled_paragraph(doc, "Limitations:", "E2 Defaults to the same core set of messages for all tasks sharing, without clearly distinguishing between a common pattern across HLA and a particular HLA rule.")

    heading(doc, "3. E8: Global + HLA Dual Branch - Joint Expert Judgement", 1)
    paragraph(doc, "E8 Splits the shared strategy into two feeder paths. Global Branch uses data from all tasks to learn common models across the tissue and across the HLA; HLAbranch groupes with HLA, allowing a special encoder to share a different task under the same allele.")
    callout(doc, "Structure", "The same peptide enters the global expert and the HLA expert. The former asks 'what is the overall pattern?', while the latter asks 'what does this HLA especially favor?'; their outputs are then combined.", fill=LIGHT_BLUE)
    labeled_paragraph(doc, "Why is it working?", "This avoids choosing one. If you use a global model only, HLA motif is easily diluted; if you just group it by hLA, the number and commonality of samples across HLA are wasted. Both branches keep both types of information.")
    labeled_paragraph(doc, "Key insight:", "This upgrade suggests that 'how to share' is more important than simply increasing network width.")

    heading(doc, "4. E13 and E14: Auxiliary Supervision - Making Global Branches more oriented", 1)
    paragraph(doc, "E13 Additional requirements for sharing indications to predict tisue and HLA are added to the main II classification task. The ultimate goal remains positive and negative; the secondary task is to force encoder to form a more organized peptide expression.")
    labeled_paragraph(doc, "Intuitive understanding:", "The general training only tells the model \"Is this correct or wrong?\" The supplementary training also requires it to say \"This peptide is more about which HLA, which Tissue.\" This will give encoder more learning clues.")
    paragraph(doc, "E14 Group this idea with E8: Global Branch uses Tissue/HLA auxiliary training, HLAbranch to keep training. This combination is better than adding auxiliary tasks to both branches.")
    callout(doc, "Why only increase Global Branch?", "Globalbranch is faced with the broadest range of tasks, most likely with the help of a Tissue/HLA information-sharing sign; HLAbranch has been specialized in allele, and additional secondary constraints may limit its freedom.", fill=LIGHT_GRAY)

    heading(doc, "5. E15: Task-rank foundation - comparison of names first, consolidation of comments", 1)
    paragraph(doc, "Both branches can produce probabilities, but the probabilities are not necessarily at the same level. One branch may often give 0.80, while the other, even with a good ranking, only gives 0.55.")
    paragraph(doc, "E15 converts each branch prediction into a within-task percentile rank and then averages the ranks. The key question becomes 'where does this sample rank within the task?' rather than whether a model reports 0.70 or 0.90.")
    labeled_paragraph(doc, "Why does AUROC fit:", "AUROC 's evaluation centred on whether the positive sample was in front of the negative sample. Therefore, the final evaluation target is more closely followed by the track.")

    heading(doc, "6. E17: More seen individual - multiple independent models voting", 1)
    paragraph(doc, "Even if model structures, data and hyper-parameters are identical, random initialization, battling and dropott allow for slightly different models for different training processes. One model may accidentally miss out on some samples, while another may not make the same mistakes.")
    callout(doc, "E17 process", "The E14a  Global branch is trained independently on an average of the seed between the seed and the HLA branch on an average of the seed and lastly on a task-rank foundation.", fill=LIGHT_BLUE)
    labeled_paragraph(doc, "Why is it working?", "The average independent training model retains a common acceptance of multiple models while weakening the occasional deviations caused by individual training processes. E17 5-seed was raised from 0.8116 of E14 to 0.8263.")
    labeled_paragraph(doc, "Why does the checkpoint/SWA not succeed in the same way:", "The different checkpoints on the same training trajectory are too similar and more similar in error; the real independent training saw provided for a more distinct diversity.")

    heading(doc, "E29: Multi-kernel CNN - Why the current strongest model is stronger", 1)
    paragraph(doc, "E14 encoder first directs the amino-acid embedding of nine locations to MLP. It learns position effects but does not have a visible structure to identify continuous local amino acid fragments.")
    paragraph(doc, "E29 Scanning peptide simultaneously with three sets of one-dimensional volume: Volumes with 2 lengths, Volumes with 3 lengths short view motif, Volumes with 5 lengths look at longer local segments. The three scales are re-assemble and pass on to the follow-on network.")
    callout(doc, "Take 9-mer as an example.", "If peptide is SLYNTVATL, the volume of 3 focuses on continuous SLY, LYN, YNT, NTV, etc. The model can thus learn directly whether a local combination is important, rather than looking at amino acids at each location.", fill=LIGHT_BLUE)
    labeled_paragraph(doc, "E29 Key Details:", "The information on the position after the volume is retained, rather than just a global maximum. For HLA-I 9-mer, the presence of the motif near the second or ninth place may have different meanings; the position is kept to avoid losing this anchor position information.")
    labeled_paragraph(doc, "Why is E29 not pushing backwards:", "E29 replaces only the peptide encoder, and continues to keep the global auxiliary branch, HLA plain branch, task-rank foundation and independent seed ensemble. It is an improvement over a strong baseline, not a one-time change in all factors.")
    paragraph(doc, "Eventually, E293-seed Esemble ' s Mean AUROC was 0.8341, Mean AUPPC was 0.8228, World-10 Mean AUROC was 0.7634, which exceeded the results of E175-seed before.")

    heading(doc, "Finally, put the whole logic together.", 1)
    callout(doc, "Core findings of the project", "First, E2 lets the task share knowledge; then E8/E14 combines global patterns, HLA specialities and subsidiary supervision; E15 solves branch scale problems; E17 reduces randomness of training; and finally E29 improves the local motif of the peptide.", fill=LIGHT_BLUE)
    paragraph(doc, "The most effective improvements do not come from complexity for its own sake. Each step addresses a clear bottleneck: sharing scope, representation quality, score comparability, or training variability. The negative E26/E27 result is also informative: if candidate models are essentially similar, a more complex second-level ensemble cannot create new information.")
    heading(doc, "Five keywords you're advised to remember.", 2)
    paragraph(doc, "Share (E2) • Tier sharing (E8) • Additional supervision (E14) • Sort Integration and Independent Integration (E15/E17) • Partial motif CNN (E29)", size=11, bold=True, color=DARK_BLUE, after=8)
    paragraph(doc, "Note: This paper is used to help understand the evolution of the project ' s internal model. All performance figures are derived from the current closed-set standard split; they do not automatically represent generalisation on the peptide-disjoint, protein-disjoint, unseen-HLA or external data.", size=9.5, color=MUTED, italic=True, before=8, after=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
