from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tissuePMHC_model_principles_guide.docx"
FORMULA_DIR = ROOT / "results" / "tissuePMHC_model_principles_formulas"
BLACK = "000000"
DARK_GRAY = "404040"
MID_GRAY = "666666"
LIGHT_GRAY = "F2F2F2"
TABLE_GRAY = "D9E1F2"
PAGE_DXA = 9360


def font(run, size=11, bold=False, color=BLACK, italic=False, name="Microsoft YaHei"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:shd"))
    if node is None:
        node = OxmlElement("w:shd")
        tc_pr.append(node)
    node.set(qn("w:fill"), color)


def width(cell, dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:tcW"))
    if node is None:
        node = OxmlElement("w:tcW")
        tc_pr.append(node)
    node.set(qn("w:w"), str(dxa))
    node.set(qn("w:type"), "dxa")


def geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    for tag, value in (("w:tblW", str(sum(widths))), ("w:tblInd", "120")):
        node = tbl_pr.first_child_found_in(tag)
        if node is None:
            node = OxmlElement(tag)
            tbl_pr.append(node)
        node.set(qn("w:w"), value)
        node.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for dxa in widths:
        child = OxmlElement("w:gridCol")
        child.set(qn("w:w"), str(dxa))
        grid.append(child)
    for row in table.rows:
        for cell, dxa in zip(row.cells, widths):
            width(cell, dxa)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def p(doc, text="", size=11, bold=False, color=BLACK, before=0, after=6, line=1.25, align=None, italic=False):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(before)
    para.paragraph_format.space_after = Pt(after)
    para.paragraph_format.line_spacing = line
    if align is not None:
        para.alignment = align
    if text:
        r = para.add_run(text)
        font(r, size, bold, color, italic)
    return para


def add_heading(doc, text, level=1):
    return p(doc, text, size={1: 16, 2: 13, 3: 12}[level], bold=True, color=BLACK,
             before={1: 18, 2: 14, 3: 10}[level], after={1: 9, 2: 7, 3: 5}[level])


def label_para(doc, label, body):
    para = p(doc, "", after=5)
    r = para.add_run(label)
    font(r, 11, True, DARK_GRAY)
    r = para.add_run(body)
    font(r, 11, False, BLACK)


def formula(doc, name, caption):
    image = FORMULA_DIR / f"{name}.png"
    if not image.is_file():
        raise FileNotFoundError(image)
    para = p(doc, "", before=2, after=1, align=WD_ALIGN_PARAGRAPH.CENTER)
    para.add_run().add_picture(str(image), width=Inches(4.8))
    p(doc, caption, size=9.5, color=MID_GRAY, after=6, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)


def callout(doc, title, body):
    table = doc.add_table(rows=1, cols=1)
    geometry(table, [PAGE_DXA])
    cell = table.cell(0, 0)
    shade(cell, LIGHT_GRAY)
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(2)
    r = para.add_run(title)
    font(r, 11, True, DARK_GRAY)
    para = cell.add_paragraph()
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.line_spacing = 1.2
    r = para.add_run(body)
    font(r, 10.5, False, BLACK)
    p(doc, "", after=3)


def cell_text(cell, text, header=False, center=False):
    para = cell.paragraphs[0]
    para.paragraph_format.space_after = Pt(0)
    para.paragraph_format.line_spacing = 1.08
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    r = para.add_run(text)
    font(r, 9.3, header, BLACK)
    if header:
        shade(cell, TABLE_GRAY)


def stage_table(doc):
    table = doc.add_table(rows=1, cols=4)
    geometry(table, [700, 2450, 1250, 4960])
    for cell, text in zip(table.rows[0].cells, ["Phase", "Model", "Mean AUROC", "Core contributions"]):
        cell_text(cell, text, header=True, center=text != "Core contributions")
    entries = [
        ("E0", "Tradition one-hot Logistic Repression", "About 0.7558", "Establish a position-amino acid linear scoring baseline."),
        ("E2", "Share peptide encoder + task heads", "Roughly 0.7927", "Let 44 tasks learn generic expressions of peptide."),
        ("E8", "Global + HLA Dual Branch", "About 0.8050", "The same is true of the same way that the whole system is taught about the different patterns of the HLA."),
        ("E14", "Auxiliary global + plain HLA", "About 0.8116", "Enhanced sharing with subsidiary oversight and retention of HLA specialization."),
        ("E15", "Task-rank fusion", "About 0.8130", "Eliminate the problem of the inconsistency between probabilities scales of the two branches."),
        ("E17", "E14a 5-seed ensemble", "About 0.8263", "The training gap is reduced on average by the use of stand-alone random seeds."),
        ("E29", "Multi-kernel CNN E14a 3-seed", "0.8341", "A visible extraction of local motif and continued to use strong dual branches with a seed esmble."),
    ]
    for row in entries:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cell_text(cells[i], text, center=i in (0, 2))
    p(doc, "", after=4)


def setup(doc):
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(1)
    sec.left_margin = sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)
    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    font(header.add_run("TissuePMHC Model Explanatory Note"), 8.5, False, MID_GRAY)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run("From E0 to E29: Model structure, integration and integration"), 8.5, False, MID_GRAY)


def build():
    doc = Document()
    setup(doc)

    p(doc, "TissuePMHC Key Model Rationale", size=24, bold=True, color=BLACK, before=12, after=3, align=WD_ALIGN_PARAGRAPH.CENTER)
    p(doc, "From traditional linear models to E29 Multi-kernel CNN 3-seed Esmble", size=13, color=DARK_GRAY, after=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    callout(doc, "Read Focus", "This document does not attempt to list all experiments, but to explain the models that really change the main performance lines: E0, E2, E8, E13/E14, E15, E17 and E29. Please focus on which bottlenecks of the old models are solved at each step.")

    add_heading(doc, "Corporate Principal: How models are built up step by step", 1)
    p(doc, "The research process can be understood as a model that first learns to 'share knowledge across tasks', then learns to 'preserve HLA differences', and is progressively enhanced through auxiliary supervision, rank fusion, multi-seed ensembling, and CNN-based local pattern extraction.")
    p(doc, "The development logic can be summarized as: E0 Traditional Linear Model E2 Multitask Sharing Encoder E8 Double Branch Model E14 Additional Oversight Enhancement Double Branch E15 task-rank foundation  E17 More Seed Integrating E29 Multi-kernel CNN 3-seed.", size=10.5, bold=True, color=DARK_GRAY, after=8)
    stage_table(doc)

    add_heading(doc, "Understanding the task first: what is the model predicting?", 1)
    p(doc, "Enter a peptide sequence with a fixed length of 9. Each forecast task is defined by the tissue and HLA allele: the model needs to determine whether this peptide is more like a sample under the given tissue-HLA conditions. The project sstandsplit contains 44 tasks.")
    callout(doc, "Intuitive metaphors", "Consider each task as a criterion for determining the different laboratories: they are looking at peptide, but the focus is not exactly the same. Some patterns are common across all laboratories, and some are established only within a particular class of HLA.")

    add_heading(doc, "E0: Traditional linear models - only see what's in each location", 1)
    p(doc, "E0 is the most basic sequence scoring machine. Each peptide has a length of 9, for example SLYNTVATL. The model expands the \"first position what amino acid is\" into one-hot features, using the Logistic Repression to give scores.")
    formula(doc, "e0_linear", "E0 linear position - amino acid scoring form.")
    p(doc, "In this case, i. When amino acids a occurs, the effect on the forecast is determined by the corresponding weight. It can learn to do independent location effects such as \"bit 2 is usually better for a amino acid\".")
    label_para(doc, "What's it good at:", "It is fast, understandable and can establish a clear traditional baseline.")
    label_para(doc, "What it lacks:", "It's not good at understanding the adjoining amino acid combinations, and it's not able to share experience with different missions. It's more like a position-amino acid score table.")

    add_heading(doc, "E2: Sharing peptide encoder - 44 tasks to learn together", 1)
    p(doc, "E2 Instead of training complete models for 44 tasks, the task will be shared with all tasks. This encoder will compress the task into a vector and then leave it to the mission 's own output head to judge.")
    formula(doc, "e2_shared", "Shares represent the relationship with the task-specific head.")
    callout(doc, "Structures", "Shares 44 task-special headers. Shares share learns general peptide; how each head learning mission uses these.")
    label_para(doc, "Why is it working?", "Data for individual tasks are limited, but the peptide mode is common in different tasks. Sharing encoder is equivalent to a set of basics for all tasks to accumulate together.")
    label_para(doc, "Limitations:", "E2 Defaults to the same core set of messages for all tasks sharing, without clearly distinguishing between a common pattern across HLA and a particular HLA rule.")

    add_heading(doc, "3. E8: Global + HLA Dual Branch - Joint Expert Judgement", 1)
    p(doc, "E8 Splits the shared strategy into two feeder paths. Global Branch uses data from all tasks to learn common models across the tissue and across the HLA; HLAbranch groupes with HLA, allowing a special encoder to share a different task under the same allele.")
    callout(doc, "Structure", "The same peptide enters the global expert and the HLA expert. The former asks 'what is the overall pattern?', while the latter asks 'what does this HLA especially favor?'; the two outputs are then combined.")
    formula(doc, "e8_fusion", "E8 The basic two branches are on average integrated.")
    label_para(doc, "Why is it working?", "This avoids choosing one. If you use a global model only, HLA motif is easily diluted; if you just group it by hLA, the number and commonality of samples across HLA are wasted. Both branches keep both types of information.")
    label_para(doc, "Key insight:", "This upgrade suggests that 'how to share' is more important than simply increasing network width.")

    add_heading(doc, "4. E13 and E14: Auxiliary Supervision - Making Global Branches more oriented", 1)
    p(doc, "E13 Additional requirements for sharing indications to predict tisue and HLA are added to the main II classification task. The ultimate goal remains positive and negative; the secondary task is to force encoder to form a more organized peptide expression.")
    label_para(doc, "Intuitive understanding:", "The general training only tells the model \"Is this correct or wrong?\" The supplementary training also requires it to say \"This peptide is more about which HLA, which Tissue.\" This will give encoder more learning clues.")
    p(doc, "E14 Group this idea with E8: Global Branch uses Tissue/HLA auxiliary training, HLAbranch to keep training. This combination is better than adding auxiliary tasks to both branches.")
    callout(doc, "Why only increase Global Branch?", "Globalbranch is faced with the broadest range of tasks, most likely with the help of a Tissue/HLA information-sharing sign; HLAbranch has been specialized in allele, and additional secondary constraints may limit its freedom.")

    add_heading(doc, "5. E15: Task-rank foundation - comparison of names first, consolidation of comments", 1)
    p(doc, "Both branches can produce probabilities, but the probabilities are not necessarily at the same level. One branch may often give 0.80, while the other, even with a good ranking, only gives 0.55.")
    p(doc, "E15 converts each branch prediction into a within-task percentile rank and then averages the ranks. The key question becomes 'where does this sample rank within the task?' rather than whether a model reports 0.70 or 0.90.")
    formula(doc, "e15_rank", "E15 tab-rank Fusion: first move to the list in the list and then average.")
    label_para(doc, "Why does AUROC fit:", "AUROC 's evaluation centred on whether the positive sample was in front of the negative sample. Therefore, the final evaluation target is more closely followed by the track.")

    add_heading(doc, "6. E17: More seen individual - multiple independent models voting", 1)
    p(doc, "Even if model structures, data and hyper-parameters are identical, random initialization, battling and dropott allow for slightly different models for different training processes. One model may accidentally miss out on some samples, while another may not make the same mistakes.")
    formula(doc, "e17_seed", "Multiple independents seed forecasts average.")
    callout(doc, "E17 process", "The E14a  Global branch is trained independently on an average of the seed between the seed and the HLA branch on an average of the seed and lastly on a task-rank foundation.")
    label_para(doc, "Why is it working?", "The average independent training model retains a common acceptance of multiple models while weakening the occasional deviations caused by individual training processes. E17 5-seed was raised from 0.8116 of E14 to 0.8263.")
    label_para(doc, "Why does the checkpoint/SWA not succeed in the same way:", "The different checkpoints on the same training trajectory are too similar and more similar in error; the real independent training saw provided for a more distinct diversity.")

    add_heading(doc, "E29: Multi-kernel CNN - Why the current strongest model is stronger", 1)
    p(doc, "E14 peptide encoder first directs the amino-acid embedding of nine locations to MLP. It learns position effects but does not have a visible structure to identify continuous local amino acid fragments.")
    p(doc, "E29 Scanning peptide simultaneously with three sets of one-dimensional volume: Volumes with 2 lengths, Volumes with 3 lengths short view motif, Volumes with 5 lengths look at longer local segments. The three scales are re-assemble and pass on to the follow-on network.")
    callout(doc, "Take 9-mer as an example.", "If peptide is SLYNTVATL, the volume of 3 focuses on continuous SLY, LYN, YNT, NTV, etc. The model can thus learn directly whether a local combination is important, rather than looking at amino acids at each location.")
    label_para(doc, "E29 Key Details:", "The information on the position after the volume is retained, rather than just a global maximum. For HLA-I 9-mer, the presence of the motif near the second or ninth place may have different meanings; the position is kept to avoid losing this anchor position information.")
    label_para(doc, "Why is E29 not pushing backwards:", "E29 replaces only the peptide encoder, and continues to keep the global auxiliary branch, HLA plain branch, task-rank foundation and independent seed ensemble. It is an improvement over a strong baseline, not a one-time change in all factors.")
    formula(doc, "e29_gain", "E29 Core: Better local motif indicates that independence is possible on average.")
    p(doc, "Eventually, E293-seed Esemble ' s Mean AUROC was 0.8341, Mean AUPPC was 0.8228, World-10 Mean AUROC was 0.7634, which exceeded the results of E175-seed before.")

    add_heading(doc, "Core findings", 1)
    p(doc, "The experiments eventually point to a clearer pattern: effective upgrading of the global/HLA branch, which is derived mainly from a rational task sharing structure, complementary, light-weight auxiliary supervision that helps with the shared representation, a task-rank foundation matching AUROC, a truly independent random seed integration, and CNN encoder that meets the 9-mer local motif characteristics.")
    p(doc, "Instead, the direction that did not bring about a significant improvement included complex dynamics, average checkpoints, SWA, MC Droopout, complex stacking on homogenous candidates, and overly complex expert/gate structures.")
    callout(doc, "A word of generality.", "E2 lets different tasks start sharing knowledge, E8 lets sharing become hierarchical, E14 lets shared expressions be guided by help, E15 resolves branch scale differences, E17 reduces training randomity, and E29 really increases the expression power of peptide local motif.")
    p(doc, "Note: This paper is used to understand the evolution of the project ' s internal model. All performance figures are derived from the current closed-set standlist; they do not automatically represent generalisation on the peptide-disjoint, protein-disjoint, unseen-HLA or external data.", size=9.5, color=MID_GRAY, italic=True, before=8, after=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
