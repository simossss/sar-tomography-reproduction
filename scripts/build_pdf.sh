#!/usr/bin/env bash
# Build paper/technical_note_SAR_tomography.pdf from the v2 manuscript.
# Requires pandoc and XeLaTeX or LuaLaTeX (Unicode text and math).
# Does not upload to Zenodo and does not change the Zenodo record title.
# CHANGELOG.md: align that title with the manuscript at upload, not in this draft.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/paper/technical_note_SAR_tomography.md"
OUT="${ROOT}/paper/technical_note_SAR_tomography.pdf"

if [[ ! -f "${SRC}" ]]; then
  echo "missing manuscript: ${SRC}" >&2
  exit 1
fi
if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required" >&2
  exit 1
fi

ENGINE=""
for candidate in xelatex lualatex; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    ENGINE="${candidate}"
    break
  fi
done
if [[ -z "${ENGINE}" ]]; then
  echo "XeLaTeX or LuaLaTeX is required (Unicode manuscript). Tried: xelatex, lualatex." >&2
  exit 1
fi

BUILD="$(mktemp -d "${TMPDIR:-/tmp}/sar-note-pdf.XXXXXX")"
cleanup() { rm -rf "${BUILD}"; }
trap cleanup EXIT

python3 - "${SRC}" "${BUILD}" <<'PY'
import sys
from pathlib import Path

src, build = sys.argv[1:]
lines = Path(src).read_text(encoding="utf-8").splitlines()
if not lines or not lines[0].startswith("# "):
    sys.exit("manuscript must start with an ATX H1 title")
title = lines[0][2:].strip()
rest = lines[1:]
try:
    idx = next(i for i, line in enumerate(rest) if line.startswith("## "))
except StopIteration:
    sys.exit("manuscript has no level-2 section")
front = "\n".join(rest[:idx]).strip() + "\n"
body = "\n".join(rest[idx:]).rstrip() + "\n"
out = Path(build)
(out / "front.md").write_text(front, encoding="utf-8")
(out / "body.md").write_text(body, encoding="utf-8")

def yq(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

(out / "meta.yaml").write_text(
    "\n".join(
        [
            "---",
            f"title: {yq(title)}",
            'author-meta: "Simone Pomposi"',
            'subject: "Version 2 manuscript draft (2026-10-23). Zenodo record title unchanged."',
            "keywords:",
            "  - SAR tomography",
            "  - single-pass Doppler",
            "  - reproducibility",
            "  - Great Pyramid of Giza",
            "lang: en-US",
            "---",
            "",
        ]
    ),
    encoding="utf-8",
)
print(title)
PY

cat > "${BUILD}/header.tex" <<'EOF'
\usepackage{titlesec}
\titleformat{\section}[block]{\Large\bfseries\raggedright}{}{0pt}{}
\titleformat{\subsection}[block]{\large\bfseries\raggedright}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.45em plus 0.15em minus 0.15em}{0.45em}
\titlespacing*{\subsection}{0pt}{1.15em plus 0.1em minus 0.1em}{0.3em}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Single-pass SAR Doppler tomography}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{\footnotesize v2 manuscript draft, 23 October 2026}
\renewcommand{\headrulewidth}{0.3pt}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \fancyfoot[L]{\footnotesize v2 manuscript draft, 23 October 2026}%
  \fancyfoot[R]{\thepage}%
  \renewcommand{\headrulewidth}{0pt}%
}
\usepackage[most]{tcolorbox}
\renewenvironment{quote}{%
  \begin{tcolorbox}[
    enhanced,
    breakable,
    colback=black!2.5,
    colframe=black!55,
    boxrule=0pt,
    leftrule=2.4pt,
    arc=0pt,
    outer arc=0pt,
    left=2.4mm,
    right=1.6mm,
    top=1.1mm,
    bottom=1.1mm,
    before skip=9pt,
    after skip=9pt
  ]
}{%
  \end{tcolorbox}%
}
\makeatletter
\renewcommand{\@maketitle}{%
  \newpage
  \null
  \vskip 0.2em%
  \begin{center}%
    {\LARGE\bfseries\hyphenpenalty=10000\exhyphenpenalty=10000 \@title \par}%
  \end{center}%
  \vskip 0.9em%
}
\makeatother
\frenchspacing
\setstretch{1.06}
\emergencystretch=3em
\tolerance=2000
\hbadness=2500
\widowpenalty=10000
\clubpenalty=10000
\displaywidowpenalty=10000
\hyphenation{Pyramid pyramid Reproduction reproduction Independent independent Giza Khufu Shabramant Vesuvius Copernicus}
\renewcommand{\arraystretch}{1.14}
\setlength{\tabcolsep}{4.5pt}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{8pt}
EOF

cat > "${BUILD}/note.lua" <<'EOF'
local function slen(text)
  local ok, len = pcall(utf8.len, text)
  if ok and len then
    return len
  end
  return #text
end

local function cell_text(cell)
  if not cell or not cell.contents then
    return ""
  end
  local ok, text = pcall(pandoc.utils.stringify, cell.contents)
  if ok and text then
    return text
  end
  return ""
end

function Header(el)
  local text = pandoc.utils.stringify(el)
  if text:find("Publication status notice", 1, true) then
    return pandoc.Para({pandoc.Strong(el.content)})
  end
end

function Table(tbl)
  local n = #tbl.colspecs
  if n < 1 or not tbl.head or not tbl.head.rows or not tbl.head.rows[1] then
    return nil
  end
  local header_cells = tbl.head.rows[1].cells
  local weights = {}
  local sum = 0
  for i = 1, n do
    local header = cell_text(header_cells[i])
    local header_len = slen(header)
    local longest = header_len
    if header_len > 2 then
      for _, body in ipairs(tbl.bodies or {}) do
        for _, row in ipairs(body.body or {}) do
          local len = slen(cell_text(row.cells[i]))
          if len > 64 then
            len = 64
          end
          if len > longest then
            longest = len
          end
        end
      end
    else
      longest = math.max(header_len, 3)
    end
    if longest < 4 then
      longest = 4
    end
    weights[i] = longest
    sum = sum + longest
  end
  local raw = {}
  local raw_sum = 0
  for i = 1, n do
    local share = weights[i] / sum
    local blended = 0.75 * share + 0.25 * (1 / n)
    if blended < 0.07 then
      blended = 0.07
    end
    raw[i] = blended
    raw_sum = raw_sum + blended
  end
  for i = 1, n do
    tbl.colspecs[i] = {tbl.colspecs[i][1], raw[i] / raw_sum}
  end
  if n >= 4 then
    return {
      pandoc.RawBlock("latex", "\\begingroup\\small"),
      tbl,
      pandoc.RawBlock("latex", "\\endgroup"),
    }
  end
  return tbl
end
EOF

FROM="markdown+tex_math_single_backslash+pipe_tables+table_captions+link_attributes+implicit_figures+smart"

pandoc "${BUILD}/front.md" \
  --from "${FROM}" \
  --pdf-engine="${ENGINE}" \
  -t latex \
  -o "${BUILD}/front.tex"

pandoc "${BUILD}/body.md" \
  --from "${FROM}" \
  --metadata-file="${BUILD}/meta.yaml" \
  --shift-heading-level-by=-1 \
  --toc \
  --toc-depth=2 \
  --pdf-engine="${ENGINE}" \
  --include-in-header="${BUILD}/header.tex" \
  --include-before-body="${BUILD}/front.tex" \
  --lua-filter="${BUILD}/note.lua" \
  -V documentclass=article \
  -V fontsize=11pt \
  -V linestretch=1.06 \
  -V papersize=a4 \
  -V geometry=a4paper \
  -V geometry=left=2.45cm \
  -V geometry=right=2.45cm \
  -V geometry=top=1.85cm \
  -V geometry=bottom=2.05cm \
  -V geometry=includeheadfoot \
  -V geometry=headheight=15pt \
  -V geometry=footskip=28pt \
  -V mainfont="TeX Gyre Pagella" \
  -V mainfontoptions="Numbers=Lining" \
  -V mathfont="TeX Gyre Pagella Math" \
  -V monofont="TeX Gyre Cursor" \
  -V monofontoptions="Scale=0.86" \
  -V colorlinks=true \
  -V linkcolor=MidnightBlue \
  -V urlcolor=MidnightBlue \
  -V filecolor=MidnightBlue \
  -V citecolor=MidnightBlue \
  -V toccolor=black \
  -o "${OUT}"

if [[ ! -s "${OUT}" ]]; then
  echo "PDF was not written: ${OUT}" >&2
  exit 1
fi

if command -v pdfinfo >/dev/null 2>&1; then
  pages="$(pdfinfo "${OUT}" | awk '/^Pages:/{print $2}')"
  echo "pages: ${pages}"
  if [[ "${pages:-0}" -lt 8 ]]; then
    echo "PDF is unexpectedly short (${pages} pages)" >&2
    exit 1
  fi
fi

if command -v pdftotext >/dev/null 2>&1; then
  text="$(pdftotext "${OUT}" - | tr '\n' ' ' | tr -s ' ')"
  for needle in \
    "Simone Pomposi" \
    "Independent Assessment" \
    "retracted" \
    "10.3390/rs18162679" \
    "SNR_peak_mean" \
    "285" \
    "No Zenodo upload"
  do
    if ! grep -q -F "${needle}" <<<"${text}"; then
      echo "PDF text is missing: ${needle}" >&2
      exit 1
    fi
  done
fi

echo "wrote ${OUT} with pandoc + ${ENGINE}"
