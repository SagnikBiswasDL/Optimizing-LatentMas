"""Build docs/loopdelta_stage0.pptx from the Stage-0 results (numbers are from
artifacts/loopdelta_run/artifacts/loopdelta/analysis/summary.json)."""
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x6A, 0x73, 0x7D)
FAINT = RGBColor(0xEE, 0xF0, 0xF3)
LINE = RGBColor(0xD0, 0xD7, 0xDE)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
GOOD = RGBColor(0x1A, 0x7F, 0x37)
BAD = RGBColor(0xCF, 0x22, 0x2E)
WARN = RGBColor(0xBF, 0x87, 0x00)
GREYS = [RGBColor(0xC9, 0xD1, 0xD9), RGBColor(0x8C, 0x95, 0x9F), RGBColor(0x57, 0x60, 0x6A), ACCENT]

SOURCE = "Source: Ouro-1.4B (plain model), MATH train set, run 2026-11-05"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, body, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    paras = body if isinstance(body, list) else [body]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(8)
        run = para.add_run()
        run.text = p
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
    return tb


def slide_base(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    if kicker:
        text(s, 0.6, 0.35, 12, 0.4, kicker, 13, MUTED)
    text(s, 0.6, 0.6, 12.1, 0.9, title, 30, INK, bold=True)
    return s


def footer(s, note=SOURCE):
    text(s, 0.6, 6.85, 12.1, 0.55, note, 10, MUTED)


def box(s, x, y, w, h, title, body, fill=FAINT, title_color=INK):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = LINE
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.15)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size, r.font.bold, r.font.color.rgb = Pt(17), True, title_color
    for line in body if isinstance(body, list) else [body]:
        q = tf.add_paragraph()
        q.space_before = Pt(6)
        r = q.add_run()
        r.text = line
        r.font.size, r.font.color.rgb = Pt(14), INK
    tf.vertical_anchor = 1  # top
    return shp


def big_number(s, x, y, w, value, label, color=INK):
    text(s, x, y, w, 0.8, value, 36, color, bold=True)
    text(s, x, y + 0.75, w, 0.8, label, 13, MUTED)


def bar_chart(s, x, y, w, h, categories, series, colors, y_max=None, y_min=None,
              pct=False, labels=True, legend=True, axis_title=None, fmt=None):
    data = CategoryChartData()
    data.categories = categories
    for name, vals in series:
        data.add_series(name, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    ch = gf.chart
    ch.font.size = Pt(12)
    ch.font.color.rgb = INK
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = FAINT
    va.format.line.fill.background()
    if y_max is not None:
        va.maximum_scale = y_max
    if y_min is not None:
        va.minimum_scale = y_min
    va.tick_labels.number_format = fmt or ('0"%"' if pct else "General")
    va.tick_labels.number_format_is_linked = False
    if axis_title:
        va.has_title = True
        va.axis_title.text_frame.text = axis_title
        va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        va.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    ch.category_axis.format.line.color.rgb = LINE
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    for ser, c in zip(plot.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = c
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.font.size = Pt(11)
        dl.number_format = fmt or ('0.0"%"' if pct else "General")
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    return ch


def table(s, x, y, w, rows, col_w, header_fill=INK, row_colors=None, size=14):
    nrows, ncols = len(rows), len(rows[0])
    t = s.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(0.45 * nrows)).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(size)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
                p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                p.runs[0].font.bold = True
            else:
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i % 2 else FAINT
                p.runs[0].font.color.rgb = INK
                if row_colors and row_colors[i - 1] and j == ncols - 1:
                    p.runs[0].font.color.rgb = row_colors[i - 1]
                    p.runs[0].font.bold = True
    return t


# 1. Title
s = prs.slides.add_slide(BLANK)
text(s, 0.8, 1.2, 11.5, 0.5, "LoopDelta  ·  Stage 0 check  ·  Ouro-1.4B", 16, MUTED)
text(s, 0.8, 1.8, 11.5, 2.0,
     "Does how much the model's memory changes between loops tell us how hard a question is?",
     34, INK, bold=True)
text(s, 0.8, 3.9, 11.0, 1.2,
     "LoopDelta uses that change (call it Δ) to pick how many loops to run per question "
     "and to decide which answers to trust. Stage 0 is the cheap test of whether Δ carries that information.",
     18, MUTED)
box(s, 0.8, 5.25, 11.7, 1.35, "Short answer: no, but we found something useful",
    "Δ mostly tracks question length and is near a coin flip at picking right answers. "
    "But many questions genuinely need fewer loops, so adaptive depth is worth chasing with a different signal.",
    fill=RGBColor(0xFF, 0xF8, 0xE5))
text(s, 0.8, 6.85, 11, 0.4, "November 2026", 12, MUTED)

# 2. Idea
s = slide_base("The idea in one picture", "Background")
text(s, 0.6, 1.45, 12, 1.0,
     "Ouro is a looped model: instead of 96 different layers it has 24 layers and runs the question through "
     "them 4 times. Each pass rewrites the model's memory of the question. Δ = how much that memory changed "
     "since the previous pass.", 16, INK)
deltas = [None, "Δ  big change", "Δ  smaller change", "Δ  tiny change → stop here?"]
for i in range(4):
    y = 2.7 + i * 1.0
    text(s, 0.6, y + 0.12, 1.2, 0.5, f"Pass {i + 1}", 16, INK, bold=True)
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.8), Inches(y), Inches(4.6), Inches(0.62))
    shp.fill.solid()
    shp.fill.fore_color.rgb = FAINT if i < 3 else RGBColor(0xDD, 0xEB, 0xFF)
    shp.line.color.rgb = LINE
    shp.text_frame.text = "same 24 layers read the whole question"
    shp.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    shp.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
    if deltas[i]:
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(6.65), Inches(y - 0.38), Inches(6.65), Inches(y + 0.31))
        c.line.color.rgb = ACCENT
        c.line.width = Pt(2)
        text(s, 6.8, y - 0.45, 3.2, 0.5, deltas[i], 14, ACCENT, bold=True)
box(s, 9.0, 2.7, 3.8, 1.75, "Use 1: pick the depth",
    "If memory stops changing after 2 passes, stop at 2 and save half the compute.")
box(s, 9.0, 4.7, 3.8, 1.75, "Use 2: weight the vote",
    "Four solvers answer. Trust the ones whose memory settled most (small Δ).")
footer(s, "Δ = squared change in keys and values between passes ÷ their size, averaged over 24 layers. "
          "Ouro has 4 passes, so 3 Δ values per question.")

# 3. Setup
s = slide_base("What we ran", "Setup")
box(s, 0.6, 1.6, 5.9, 2.6, "Part A · read the question, don't answer",
    ["1,000 MATH training questions, 200 from each difficulty level 1–5.",
     "Record Δ. Does it line up with difficulty?",
     "Repeated at 8 loops (2× Ouro's training) to give Δ more room."])
box(s, 6.8, 1.6, 5.9, 2.6, "Part B · actually answer",
    ["300 questions, 60 per level, each at 1, 2, 3 and 4 loops.",
     "Per depth: 1 careful answer + 4 varied answers for voting.",
     "6,000 answers in total, graded automatically."])
for k, (v, l) in enumerate([("1,000", "questions, Part A"), ("6,000", "graded answers, Part B"),
                            ("~7 hrs", "on one GPU"), ("~$18", "total compute")]):
    big_number(s, 0.6 + k * 3.1, 4.6, 3.0, v, l)
footer(s, "\"Cost\" later = generated tokens × loops run. That is what the GPU pays for: "
          "2 loops costs half of 4 loops for the same answer length.")

# 4. Loops vs accuracy
s = slide_base("Finding 1: fewer loops is often enough", "Results")
text(s, 0.6, 1.4, 12, 0.6, "Most of the gain comes from going 1 → 2 loops. Hard questions stay hard even at 4.", 17, INK)
bar_chart(s, 0.6, 2.0, 12.1, 4.8, ["Level 1 (easiest)", "Level 2", "Level 3", "Level 4", "Level 5 (hardest)"],
          [("1 loop", (66.7, 38.3, 20, 20, 5)), ("2 loops", (93.3, 80, 58.3, 56.7, 31.7)),
           ("3 loops", (93.3, 85, 66.7, 58.3, 28.3)), ("4 loops", (91.7, 85, 76.7, 56.7, 31.7))],
          GREYS, y_max=100, y_min=0, pct=True, labels=False, axis_title="% answered correctly")
footer(s, "Greedy accuracy, 60 questions per level. Overall: 1 loop 30%, 2 loops 64%, 3 loops 66%, 4 loops 68%.  " + SOURCE)

# 5. Headroom
s = slide_base("How many loops does each question really need?", "Results")
text(s, 0.6, 1.4, 12, 0.6, "70% of questions are solved by 2 loops. Only 43 of 300 need 3 or 4.", 17, INK)
bar_chart(s, 0.6, 2.0, 7.6, 4.7, ["1 loop", "2 loops", "3 loops", "4 loops", "never"],
          [("Questions", (92, 117, 34, 9, 48))], [ACCENT], legend=False,
          axis_title="number of questions (of 300)")
big_number(s, 8.8, 2.3, 4.0, "77.8%", "a perfect depth picker (cheapest correct depth)", GOOD)
big_number(s, 8.8, 3.9, 4.0, "68.0%", "always running 4 loops", INK)
text(s, 8.8, 5.4, 4.0, 1.3, "…and the perfect picker uses about 2/3 the compute. "
     "That gap is the prize for any depth-picking signal.", 15, MUTED)
footer(s, "Fewest loops with a correct greedy answer. Perfect picker on held-out 153 questions: cost 3,318 vs 5,090.  " + SOURCE)

# 6. Test 1
s = slide_base("Test 1: does Δ know which questions are hard?", "Results")
text(s, 0.6, 1.4, 12, 0.8, "A little at first glance. But longer questions are both harder and give bigger Δ. "
     "Account for length and the link disappears.", 17, INK)
bar_chart(s, 0.6, 2.2, 7.8, 4.5, ["Δ, last loop", "Δ, all loops", "Question length"],
          [("Raw link to difficulty", (0.22, 0.06, 0.39)),
           ("After removing question length", (-0.07, -0.07, 0))],
          [ACCENT, GREYS[1]], y_min=-0.2, y_max=0.5, fmt="0.00", axis_title="rank correlation with level")
box(s, 8.9, 2.6, 3.9, 2.4, "In plain words",
    "If you already know how long the question is, Δ tells you nothing extra about how hard it is.",
    fill=RGBColor(0xFF, 0xF8, 0xE5))
text(s, 8.9, 5.2, 3.9, 1.2, "Same at 8 loops: 0.23 raw, −0.06 after removing length.", 14, MUTED)
footer(s, "1,000 questions, 4 loops. 0 = no link, 1 = perfect. \"After removing length\" is 0 by definition for length itself.  " + SOURCE)

# 7. Test 2
s = slide_base("Test 2: does Δ pick depths well?", "Results")
text(s, 0.6, 1.4, 12, 0.8, "Each signal chooses 2, 3 or 4 loops per question (tuned on half the questions, scored on the other half). "
     "Δ beats always-4, but free question length does as well and cheaper.", 16, INK)
table(s, 0.6, 2.35, 12.1, [
    ["How depth is chosen", "Accuracy", "Cost / question", "Verdict"],
    ["Always 4 loops", "68.0%", "5,090", "baseline"],
    ["Random mix, same budget as Δ", "67.5%", "4,290", "luck alone"],
    ["Pick by Δ", "70.6%", "4,260", "beats baseline"],
    ["Pick by question length", "70.6%", "3,370", "same accuracy, 21% cheaper"],
    ["Pick by question length, more budget", "71.9%", "3,990", "best real signal"],
    ["Perfect picker (cheats)", "77.8%", "3,320", "the ceiling"],
], [5.1, 1.7, 2.1, 3.2], row_colors=[None, None, WARN, GOOD, GOOD, ACCENT], size=15)
footer(s, "Greedy, held-out 153 questions (1 question = 0.65 points). Cost = tokens × loops; Δ and length policies include reading "
          "the question at 4 loops first.  " + SOURCE)

# 8. Test 3
s = slide_base("Test 3: does small Δ mean \"trust this answer\"?", "Results")
text(s, 0.6, 1.4, 12, 0.8, "Among the four answers to the same question, does lower Δ point to the correct one? "
     "Near coin flip. The model's own confidence is much better and free.", 16, INK)
ch = bar_chart(s, 0.6, 2.2, 8.2, 4.6, ["2 loops", "3 loops", "4 loops"],
               [("Δ (lower = trust)", (0.40, 0.52, 0.59)), ("Token probability", (0.75, 0.66, 0.84)),
                ("Low entropy", (0.75, 0.68, 0.84))],
               [WARN, GOOD, GREYS[1]], y_min=0.3, y_max=0.9, fmt="0.00",
               axis_title="AUROC within a question (0.5 = coin flip)")
big_number(s, 9.3, 2.4, 3.6, "77.0%", "plain majority vote, 4 loops")
big_number(s, 9.3, 3.8, 3.6, "76.3%", "vote weighted by Δ", WARN)
text(s, 9.3, 5.2, 3.6, 1.5, "With only four voters, weighting barely matters. Δ only looks useful across "
     "questions, because hard questions have big Δ and wrong answers.", 13, MUTED)
footer(s, "254–276 questions per depth where the four answers disagreed. No Δ at 1 loop.  " + SOURCE)

# 9. Side findings
s = slide_base("Two side findings worth keeping", "Results")
text(s, 0.6, 1.45, 6.2, 0.5, "With voting, 3 loops is as good as 4", 18, INK, bold=True)
bar_chart(s, 0.6, 1.95, 6.4, 4.4, ["1 loop", "2 loops", "3 loops", "4 loops"],
          [("One greedy answer", (30, 64, 66.3, 68.3)), ("Majority of 4", (30.7, 67.7, 77.3, 77.0))],
          [GREYS[0], ACCENT], y_min=0, y_max=100, pct=True, axis_title="% correct (300 questions)")
text(s, 0.6, 6.35, 6.4, 0.5, "Same accuracy at 69% of the compute (13,678 vs 19,860).", 13, MUTED)
text(s, 7.5, 1.45, 5.3, 0.5, "The slowest solver adds little wait", 18, INK, bold=True)
text(s, 7.5, 2.0, 5.3, 1.5, "The four solvers run side by side, so the team waits for the longest answer.", 15, INK)
big_number(s, 7.5, 3.2, 2.6, "1.27×", "slowest vs typical, 4 loops", GOOD)
big_number(s, 10.2, 3.2, 2.6, "1.86×", "slowest vs typical, 1 loop", WARN)
text(s, 7.5, 4.9, 5.3, 1.2, "1-loop answers often ramble to the length limit, which is why the wait is longer there.", 13, MUTED)
footer(s)

# 10. Verdict
s = slide_base("Verdict and next steps", "Conclusion")
table(s, 0.6, 1.5, 12.1, [
    ["Stage 0 question", "Evidence", "Answer"],
    ["Does Δ know difficulty?", "Link drops from 0.22 to −0.07 once length is known", "Only via length"],
    ["Does Δ pick depth well?", "Same 70.6% as length, which is 21% cheaper", "No better than length"],
    ["Does Δ pick good answers?", "Near coin flip; token probability far better", "No"],
    ["Is adaptive depth worth it?", "Perfect picker: 77.8% at 2/3 the compute", "Yes"],
], [3.6, 6.0, 2.5], row_colors=[BAD, BAD, BAD, GOOD], size=15)
box(s, 0.6, 4.0, 12.1, 1.5, "Recommendation",
    "Don't build Stage 1 on Δ. Keep the adaptive-depth goal and test cheaper depth signals on the data we already "
    "have: question length, entropy after the first loop, and Ouro's own learned exit gate. No new GPU run needed.",
    fill=RGBColor(0xE8, 0xF1, 0xFF))
text(s, 0.6, 5.75, 12.1, 1.1, [
    "Caveats: one model (Ouro-1.4B) and one dataset (MATH train); depth-policy numbers use 153 held-out questions, so small gaps are noise.",
    "Ouro has only 4 loops (3 Δ values). Running to 8 loops gave the same answer but is outside its training."], 12, MUTED)

out = "docs/loopdelta_stage0.pptx"
prs.save(out)
print("wrote", out, len(prs.slides), "slides")
