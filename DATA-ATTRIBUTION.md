# Data attribution

HumanArt's code is licensed MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). Some files in the
repository are derived from third-party data and keep that data's licence. This file gives the credit
those licences require; `docs/data-provenance.md` lists every file with its exact source.

## HSRD-100 (CC BY 4.0)

`data/template/face/face-model.hafm`, the default face (embedded in `humanart-model`), is derived from:

> HSRD-100: 100 High-Quality 3D Human Scans Dataset, Digital Reality Lab (2025),
> https://huggingface.co/datasets/digitalrealitylab/HSRD-100, licensed under
> [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Changes made: the head and neck of 8 scans were fitted to HumanArt's template mesh; the file holds the
symmetric mean offset of those 8 fits and their principal components (shape directions) on the
template's vertices. It contains no scan mesh, texture, photograph or participant name. The dataset
card states that all participants gave informed consent. HSRD-100's own licence applies to this file;
HumanArt's licence does not replace it.

`data/template/skin/albedo-detail.png`, the photographed skin colour detail (embedded in
`humanart-agent`), is derived from the same dataset: a 256 × 256 texel patch (about 19 mm) of cheek
skin of scan HSR0015-Body-023's colour texture, high-passed (each channel over its Gaussian blur,
so the capture's light and the person's base colour drop out), made periodic and stored as log
ratios (`humanart-cli skin-detail`). It holds no face, feature or name; HSRD-100's licence applies to
it.

## ISSA (CC BY 4.0)

`data/test-scenes/skin-reference.json` holds summary values computed from: Lu Y. et al. 2025, "The
International Skin Spectra Archive (ISSA): a multicultural human skin phenotype and colour spectra
collection", Scientific Data 12, 487, doi:10.1038/s41597-025-04857-5 (data: doi:10.6084/m9.figshare.28228571.v4),
licensed under CC BY 4.0. Changes made: per-group aggregates only.

## CIE colour tables (CC BY-SA 4.0)

The CIE 1931 2° colour-matching functions and illuminant D65, sampled at 5 nm in
`crates/model/src/skin_appearance/spectra_table.rs`, are from the CIE datasets doi:10.25039/CIE.DS.xvudnb9b
and doi:10.25039/CIE.DS.hjfjmt59, licensed under CC BY-SA 4.0; that subset is shared under the same
licence (ADR-038).

## ANSUR II (public release)

`data/anthropometry/ansur2.json` holds summary statistics of the ANSUR II anthropometric survey (US Army
NSRDEC, 2010-2012; Gordon et al. 2014, NATICK/TR-15/007), cleared for unlimited public release.

## BodyParts3D (CC BY 4.0), reference only

Body proportions and muscle and bone forms were checked against "BodyParts3D, © The Database Center for
Life Science licensed under CC Attribution 4.0 International". No BodyParts3D file or mesh is in the
repository.

## Fonts

Inter and JetBrains Mono (SIL Open Font License 1.1) and Lucide (ISC); their licence texts are in
`data/fonts/`.
