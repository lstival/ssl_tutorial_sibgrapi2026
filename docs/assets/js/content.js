/* ------------------------------------------------------------
   Single source of truth for outbound links and notebook list.
   ------------------------------------------------------------ */

const REPO = "lstival/ssl_tutorial_sibgrapi2026";
const BRANCH = "main";

/* "pending" = code repo not public yet -> links render disabled.
   "live"    = everything resolves normally.
   The repository is public and the weights are published on its "weights-v1" GitHub
   Release, so the notebook/Colab/checkpoint links below all resolve. */
const SITE_STATUS = "live";

const GH = `https://github.com/${REPO}`;
const RAW_NB = (p) => `${GH}/blob/${BRANCH}/notebooks/${p}`;
const COLAB = (p) => `https://colab.research.google.com/github/${REPO}/blob/${BRANCH}/notebooks/${p}`;

const NOTEBOOKS = {
  remote_sensing: {
    label: "Remote sensing",
    sub: "EuroSAT · ViT-S/8 · linear probe",
    items: [
      { n: "00", f: "00_setup_and_data.ipynb", k: null,
        t: "Setup & data",
        d: "EuroSAT loading, RS-specific augmentations, and the shared ViT-S/8 backbone every later notebook reuses." },
      { n: "01", f: "01_contrastive_simclr.ipynb", k: "con",
        t: "Contrastive — SimCLR",
        d: "InfoNCE over augmented view pairs. One fill-in-the-blank: the similarity matrix and temperature." },
      { n: "02", f: "02_masking_mae.ipynb", k: "mask",
        t: "Masking — MAE",
        d: "Masked autoencoding at 75% ratio with an asymmetric decoder. Loss computed on masked patches only." },
      { n: "03", f: "03_distillation_dino.ipynb", k: "dist",
        t: "Distillation — DINO",
        d: "EMA teacher, stop-gradient, centering and sharpening — collapse avoided without any negatives." },
      { n: "04", f: "04_comparative_evaluation.ipynb", k: null,
        t: "Comparative evaluation",
        d: "Frozen-encoder linear probe across all three encoders, plus few-label curves and t-SNE." }
    ]
  },
  time_series: {
    label: "Time series",
    sub: "UCR archive · patch Transformer · linear probe",
    items: [
      { n: "00", f: "00_setup_and_data.ipynb", k: null,
        t: "Setup & data",
        d: "UCR corpus, instance normalization, patching, and the shared 1D Transformer encoder." },
      { n: "01", f: "01_contrastive_simclr.ipynb", k: "con",
        t: "Contrastive — SimCLR",
        d: "The same InfoNCE objective on 1D sequences: jitter, scaling and cropping replace the image augmentations." },
      { n: "02", f: "02_masking_mae.ipynb", k: "mask",
        t: "Masking — MAE",
        d: "Contiguous-segment masking sized against the autocorrelation window, so interpolation is not enough." },
      { n: "03", f: "03_distillation_dino.ipynb", k: "dist",
        t: "Distillation — DINO",
        d: "Teacher–student distillation on temporal views — the strongest encoder on this modality." },
      { n: "04", f: "04_comparative_evaluation.ipynb", k: null,
        t: "Comparative evaluation",
        d: "Same protocol as remote sensing, so the two result tables can be read side by side." }
    ]
  }
};

/* Published as assets on the "weights-v1" GitHub Release -- the notebooks download them
   from there on demand, on Colab or locally. */
const MODELS = [
  { k:"con",  n:"contrastive_vit_s8.pt", p:"Contrastive", m:"Remote sensing", a:"ViT-S/8",  d:"SeCo 100k", s:"42 MB" },
  { k:"mask", n:"mae_vit_s8.pt",         p:"Masking",     m:"Remote sensing", a:"ViT-S/8",  d:"SeCo 100k", s:"42 MB" },
  { k:"dist", n:"dino_vit_s8.pt",        p:"Distillation",m:"Remote sensing", a:"ViT-S/8",  d:"SeCo 100k", s:"42 MB" },
  { k:"con",  n:"contrastive_ts_encoder.pt", p:"Contrastive", m:"Time series", a:"Patch Transformer", d:"UCR (128 sets)", s:"2.4 MB" },
  { k:"mask", n:"mae_ts_encoder.pt",         p:"Masking",     m:"Time series", a:"Patch Transformer", d:"UCR (128 sets)", s:"2.4 MB" },
  { k:"dist", n:"dino_ts_encoder.pt",        p:"Distillation",m:"Time series", a:"Patch Transformer", d:"UCR (128 sets)", s:"2.4 MB" }
];
