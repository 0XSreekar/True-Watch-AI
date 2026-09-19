# Fonts

**The font binary is never committed.** It is 400 KB of third-party data and belongs in a
package manager, not in this repository's history.

`gen_plates.py` needs Noto Sans Devanagari, licensed under the SIL Open Font License 1.1,
which permits free use and redistribution of the rendered output.

## Fetch it

```bash
curl -L -o /tmp/noto.zip \
  "https://fonts.google.com/download?family=Noto%20Sans%20Devanagari"
unzip -o /tmp/noto.zip -d datasets/plates/fonts/
```

Or, on a machine with `fonttools` and network access to the Google Fonts repository:

```bash
curl -L -o datasets/plates/fonts/NotoSansDevanagari-Regular.ttf \
  "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
```

## Verify

```bash
python3 datasets/plates/gen_plates.py --check-font
```

That prints the resolved font path and renders one glyph, or exits non-zero with the
reason. `gen_plates.py` searches, in order:

1. `--font <path>`
2. `datasets/plates/fonts/*.ttf`
3. the system font directories for a file whose name contains `NotoSansDevanagari`

If none is found it exits 2 with the fetch command above, rather than falling back to a
font without Devanagari coverage and silently rendering boxes.
