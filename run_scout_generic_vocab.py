#!/usr/bin/env python3
"""
SCOUT evaluated with the generic (class-agnostic) finding vocabulary instead
of the oracle vocabulary, on both benchmarks. Reuses the fixed max(z_knn,
z_vocab) fusion rule exactly as defined for the main SCOUT result; only the
vocabulary term changes. The "skin lesion ulceration" term used in the
dermatology generic list has been replaced (it overlapped with the oracle
BCC/AK term "ulceration"); no other change to protocol, splits, or scoring.
"""
import sys
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import open_clip
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, "/workspace/medical_ood/flair_repo")
from flair import FLAIRModel

ROOT = Path("/workspace/medical_ood")
OUT = ROOT / "outputs/scout_generic_vocab_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

OPHTH_DR_NORMAL_TERMS = ["microaneurysm", "hard exudate", "cotton wool spot", "dot and blot hemorrhage",
                          "retinal neovascularization", "macular edema", "venous beading",
                          "diabetic retinal hemorrhage", "normal healthy fundus",
                          "clear optic disc with no pathology"]
OPHTH_GENERIC_UNSEEN_TERMS = ["retinal hemorrhage", "macular degeneration pattern", "optic nerve damage",
                               "retinal detachment", "vascular occlusion", "pigmentary retinal changes",
                               "optic disc swelling", "subretinal fluid", "choroidal lesion",
                               "nerve fiber layer defect"]

DERM_NV_MEL_TERMS = ["pigment network", "atypical pigment network", "blue-white veil",
                      "irregular streaks", "regression structures", "asymmetric pigmentation",
                      "melanocytic nevus pattern", "malignant melanoma pattern",
                      "irregular dots and globules", "multicomponent pattern"]
DERM_GENERIC_UNSEEN_TERMS = ["abnormal vascular pattern", "irregular keratotic scale",
                              "pigment irregularity", "nodular skin growth", "erythematous plaque",
                              "telangiectatic vessels", "atypical scaling pattern",
                              "raised skin nodule", "crusted skin surface", "indurated skin plaque"]

RANK_CANDIDATES = [16, 32, 48, 64, 96, 128, 192, 256]


def preprocess(image_np, image_size=512):
    image = np.float32(image_np)
    if image.max() > 0:
        image = image / 255
    if len(image.shape) > 2:
        image = np.transpose(image, (2, 0, 1))
    else:
        image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    image = torch.tensor(image)
    h, w = image.shape[-2], image.shape[-1]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    image = image[..., top:top + side, left:left + side]
    image = torchvision.transforms.Resize((image_size, image_size), antialias=True)(image)
    return image.to(torch.float32).to(device)


def split_dev_eval(n, frac_dev=0.3, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_dev = int(n * frac_dev)
    return idx[:n_dev], idx[n_dev:]


def fpr_at_95_tpr(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


def evaluate(id_scores, ood_scores):
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    return {
        "AUROC": round(roc_auc_score(y_true, y_score) * 100, 2),
        "AUPR": round(average_precision_score(y_true, y_score) * 100, 2),
        "FPR95": round(fpr_at_95_tpr(y_true, y_score) * 100, 2),
    }


def knn_score(X_query, X_dev, k=20):
    sims = X_query @ X_dev.T
    topk_sims = np.sort(sims, axis=1)[:, -k:]
    return 1 - topk_sims.mean(axis=1)


def run_fusion(per_image, label):
    groups = ["id", "semantic", "covariate", "far_ood"]
    seeds = [42, 1, 7, 123, 2024]
    all_results = {"knn_only": {s: {} for s in seeds}, "vocab_generic_only": {s: {} for s in seeds},
                   "scout_generic": {s: {} for s in seeds}}

    for seed in seeds:
        by_group = {g: [p for p in per_image if p["group"] == g] for g in groups}
        dev, ev = {}, {}
        for g in groups:
            idx = np.arange(len(by_group[g]))
            dev_idx, eval_idx = split_dev_eval(len(idx), seed=seed)
            dev[g] = [by_group[g][i] for i in dev_idx]
            ev[g] = [by_group[g][i] for i in eval_idx]

        X_dev = np.array([p["emb"] for p in dev["id"]])
        knn_dev = knn_score(X_dev, X_dev, k=20)
        vocab_dev = np.array([p["vocab_generic"] for p in dev["id"]])
        knn_mu, knn_sd = knn_dev.mean(), knn_dev.std() + 1e-8
        vocab_mu, vocab_sd = vocab_dev.mean(), vocab_dev.std() + 1e-8

        id_knn = knn_score(np.array([p["emb"] for p in ev["id"]]), X_dev, k=20)
        id_vocab = np.array([p["vocab_generic"] for p in ev["id"]])
        id_z_knn = (id_knn - knn_mu) / knn_sd
        id_z_vocab = (id_vocab - vocab_mu) / vocab_sd
        id_fusion = np.maximum(id_z_knn, id_z_vocab)

        for shift in ["semantic", "covariate", "far_ood"]:
            s_knn = knn_score(np.array([p["emb"] for p in ev[shift]]), X_dev, k=20)
            s_vocab = np.array([p["vocab_generic"] for p in ev[shift]])
            s_z_knn = (s_knn - knn_mu) / knn_sd
            s_z_vocab = (s_vocab - vocab_mu) / vocab_sd
            s_fusion = np.maximum(s_z_knn, s_z_vocab)

            all_results["knn_only"][seed][shift] = evaluate(id_knn, s_knn)
            all_results["vocab_generic_only"][seed][shift] = evaluate(id_vocab, s_vocab)
            all_results["scout_generic"][seed][shift] = evaluate(id_fusion, s_fusion)

        print(f"  [{label}] seed {seed} done", flush=True)

    print(f"\n{'=' * 90}\n[{label}]", flush=True)
    for name, by_seed in all_results.items():
        row = f"{name:22s} | "
        means = {}
        for shift in ["semantic", "covariate", "far_ood"]:
            vals = [by_seed[s][shift]["AUROC"] for s in seeds]
            fprs = [by_seed[s][shift]["FPR95"] for s in seeds]
            means[shift] = np.mean(vals)
            row += f"{shift}={np.mean(vals):5.1f}({np.mean(fprs):5.1f})  "
        row += f" mean={np.mean(list(means.values())):.2f}"
        print(row, flush=True)

    return {"results": {name: {str(s): r for s, r in by_seed.items()} for name, by_seed in all_results.items()}}


def run_ophthalmology():
    manifest = ROOT / "data/manifests/benchmark_manifest.csv"
    model = FLAIRModel(vision_type="resnet_v1", bert_type="emilyalsentzer/Bio_ClinicalBERT",
                        proj_dim=512, from_checkpoint=True, caption="[CLS]")
    model.eval()
    model.to(device)
    resnet = model.vision_model.model

    text_input_ids, text_attention_mask = model.preprocess_text(OPHTH_DR_NORMAL_TERMS + OPHTH_GENERIC_UNSEEN_TERMS)
    with torch.no_grad():
        text_embeds = model.text_model(text_input_ids, text_attention_mask)
    text_embeds = F.normalize(text_embeds, dim=-1).cpu().numpy()
    known_embeds = text_embeds[:len(OPHTH_DR_NORMAL_TERMS)]
    generic_embeds = text_embeds[len(OPHTH_DR_NORMAL_TERMS):]

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Manifest rows: {len(rows)}", flush=True)

    per_image = []
    for i, row in enumerate(rows):
        image_np = np.array(Image.open(row["filepath_relative"]).convert("RGB"))
        pixel_values = preprocess(image_np)
        with torch.no_grad():
            e = resnet(pixel_values)
            e = model.vision_model.projection_head_vision(e)
        emb = F.normalize(e, dim=-1).cpu().numpy()[0]
        sim_known = float(np.mean(known_embeds @ emb))
        sim_generic = float(np.mean(generic_embeds @ emb))
        per_image.append({"group": row["group"], "emb": emb, "vocab_generic": sim_generic - sim_known})
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    return run_fusion(per_image, "ophthalmology")


def run_dermatology():
    manifest = ROOT / "data/manifests/dermatology_manifest.csv"
    bmodel, preprocess_fn = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    bmodel.eval()
    bmodel.to(device)

    def encode_text(prompts, batch_size=256):
        all_embeds = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.no_grad():
                feat = bmodel.encode_text(tokens)
            feat = F.normalize(feat, dim=-1).cpu().numpy()
            all_embeds.append(feat)
        return np.concatenate(all_embeds, axis=0)

    known_embeds = encode_text(DERM_NV_MEL_TERMS)
    generic_embeds = encode_text(DERM_GENERIC_UNSEEN_TERMS)

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Manifest rows: {len(rows)}", flush=True)

    per_image = []
    for i, row in enumerate(rows):
        img = Image.open(row["filepath"]).convert("RGB")
        pixel_values = preprocess_fn(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = bmodel.encode_image(pixel_values)
        emb = F.normalize(img_feat, dim=-1).cpu().numpy()[0]
        sim_known = float(np.mean(known_embeds @ emb))
        sim_generic = float(np.mean(generic_embeds @ emb))
        per_image.append({"group": row["group"], "emb": emb, "vocab_generic": sim_generic - sim_known})
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    return run_fusion(per_image, "dermatology")


def main():
    ophth = run_ophthalmology()
    derm = run_dermatology()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"ophthalmology": ophth, "dermatology": derm}, f, indent=2)
    print(f"\nWritten to {OUT}", flush=True)


if __name__ == "__main__":
    main()
