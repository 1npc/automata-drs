#!/usr/bin/env python3
"""List possible single-word prose lines in the compiled two-column paper.

This is a layout diagnostic, not a prose rewrite. Equations, figures, tables,
and short headings still require manual inspection of any flagged candidate.
"""
import argparse
import re

import pdfplumber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    args = parser.parse_args()
    with pdfplumber.open(args.pdf) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for column, bounds in [("L", (70, 60, 293, 780)),
                                   ("R", (305, 60, 527, 780))]:
                crop = page.crop(bounds)
                words = crop.extract_words(x_tolerance=2, y_tolerance=2,
                                           extra_attrs=["fontname", "size"])
                rows = []
                for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
                    if not rows or abs(word["top"] - rows[-1][0]) > 3:
                        rows.append((word["top"], [word]))
                    else:
                        rows[-1][1].append(word)
                for j, (top, row) in enumerate(rows):
                    row.sort(key=lambda w: w["x0"])
                    if len(row) != 1:
                        continue
                    word = row[0]
                    if not (9.8 < word["size"] < 11.2):
                        continue
                    if "NimbusRom" not in word["fontname"] or "Medi" in word["fontname"]:
                        continue
                    if not re.search("[A-Za-z]", word["text"]):
                        continue
                    context = rows[max(0, j - 2):j + 2]
                    print(f"page {page_number} {column}, y={top:.1f}: {word['text']}")
                    for _, nearby in context:
                        print("  " + " ".join(w["text"] for w in sorted(nearby, key=lambda w: w["x0"])))


if __name__ == "__main__":
    main()
