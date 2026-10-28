#!/usr/bin/env python3
"""
Final, authoritative SCOUT evaluation with a real, saved paired bootstrap.
Re-embeds both benchmarks (FLAIR ophthalmology, BiomedCLIP dermatology),
computes kNN distance, oracle finding-vocabulary score, and their max-z
fusion, and saves per-image scores so the paired bootstrap comparing fusion
against each component is reproducible from this file alone, not from a
transcript or a prior ad-hoc run.
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
OUT = ROOT / "outputs/scout_final_verified_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

OPHTH_DR_NORMAL_TERMS = ["microaneurysm", "hard exudate", "cotton wool spot", "dot and blot hemorrhage",
                          "retinal neovascularization", "macular edema", "venous beading",
                          "diabetic retinal hemorrhage", "normal healthy fundus",
                          "clear optic disc with no pathology"]
OPHTH_ORACLE_UNSEEN_TERMS = ["drusen deposits", "geographic atrophy", "choroidal neovascularization",
                              "retinal pigment epithelium detachment", "optic disc cupping",
                              "retinal nerve fiber layer thinning", "optic nerve pallor",
                              "peripapillary atrophy", "age-related macular degeneration",
                              "glaucomatous angle closure"]

DERM_NV_MEL_TERMS = ["pigment network", "atypical pigment network", "blue-white veil",
                      "irregular streaks", "regression structures", "asymmetric pigmentation",
                      "melanocytic nevus pattern", "malignant melanoma pattern",
                      "irregular dots and globules", "multicomponent pattern"]
DERM_ORACLE_UNSEEN_TERMS = ["arborizing vessels", "ulceration", "blue-gray ovoid nests", "leaf-like structures",
                             "shiny white structures", "actinic keratosis scale", "erythematous scaly patch",
                             "strawberry pattern", "rosette structures", "basal cell carcinoma pattern"]


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


def paired_bootstrap(id_a, ood_a, id_b, ood_b, seed, n_boot=1000):
    rng = np.random.RandomState(seed)
    n_id, n_ood = len(id_a), len(ood_a)
    y = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    auc_a = roc_auc_score(y, np.concatenate([id_a, ood_a]))
    auc_b = roc_auc_score(y, np.concatenate([id_b, ood_b]))
    deltas = []
    for _ in range(n_boot):
        ii = rng.randint(0, n_id, n_id)
        oi = rng.randint(0, n_ood, n_ood)
        a = roc_auc_score(y, np.concatenate([id_a[ii], ood_a[oi]]))
        b = roc_auc_score(y, np.concatenate([id_b[ii], ood_b[oi]]))
        deltas.append((b - a) * 100)
    deltas = np.array(deltas)
    return {"delta_AUROC_pp": round((auc_b - auc_a) * 100, 3),
            "CI95_low": round(np.percentile(deltas, 2.5), 3),
            "CI95_high": round(np.percentile(deltas, 97.5), 3)}


def run_benchmark(per_image, label):
    groups = ["id", "semantic", "covariate", "far_ood"]
    seeds = [42, 1, 7, 123, 2024]
    metric_rows = {"knn_only": {s: {} for s in seeds}, "vocab_only": {s: {} for s in seeds},
                    "fusion": {s: {} for s in seeds}}
    gain_rows = []

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
        vocab_dev = np.array([p["vocab_oracle"] for p in dev["id"]])
        knn_mu, knn_sd = knn_dev.mean(), knn_dev.std() + 1e-8
        vocab_mu, vocab_sd = vocab_dev.mean(), vocab_dev.std() + 1e-8

        id_knn = knn_score(np.array([p["emb"] for p in ev["id"]]), X_dev, k=20)
        id_vocab = np.array([p["vocab_oracle"] for p in ev["id"]])
        id_z_knn = (id_knn - knn_mu) / knn_sd
        id_z_vocab = (id_vocab - vocab_mu) / vocab_sd
        id_fusion = np.maximum(id_z_knn, id_z_vocab)

        for shift in ["semantic", "covariate", "far_ood"]:
            s_knn = knn_score(np.array([p["emb"] for p in ev[shift]]), X_dev, k=20)
            s_vocab = np.array([p["vocab_oracle"] for p in ev[shift]])
            s_z_knn = (s_knn - knn_mu) / knn_sd
            s_z_vocab = (s_vocab - vocab_mu) / vocab_sd
            s_fusion = np.maximum(s_z_knn, s_z_vocab)

            metric_rows["knn_only"][seed][shift] = evaluate(id_knn, s_knn)
            metric_rows["vocab_only"][seed][shift] = evaluate(id_vocab, s_vocab)
            metric_rows["fusion"][seed][shift] = evaluate(id_fusion, s_fusion)

            # real paired bootstrap, fusion vs each raw component score,
            # same per-image arrays used for the AUROC above
            gain_knn = paired_bootstrap(id_knn, s_knn, id_fusion, s_fusion, seed=seed)
            gain_vocab = paired_bootstrap(id_vocab, s_vocab, id_fusion, s_fusion, seed=seed)
            gain_rows.append({"seed": seed, "shift": shift, "comparison": "fusion-vs-knn", **gain_knn})
            gain_rows.append({"seed": seed, "shift": shift, "comparison": "fusion-vs-vocab", **gain_vocab})

        print(f"  [{label}] seed {seed} done", flush=True)

    print(f"\n{'=' * 90}\n[{label}] means across seeds:", flush=True)
    for name, by_seed in metric_rows.items():
        row = f"{name:12s} | "
        for shift in ["semantic", "covariate", "far_ood"]:
            vals = [by_seed[s][shift]["AUROC"] for s in seeds]
            row += f"{shift}={np.mean(vals):5.2f}+/-{np.std(vals):4.2f}  "
        print(row, flush=True)

    print(f"\n[{label}] paired bootstrap, fusion vs components, per shift, per seed:", flush=True)
    for g in gain_rows:
        print(f"  seed={g['seed']} shift={g['shift']} {g['comparison']}: "
              f"delta={g['delta_AUROC_pp']} CI=[{g['CI95_low']},{g['CI95_high']}]", flush=True)

    return {"metrics": {name: {str(s): r for s, r in by_seed.items()} for name, by_seed in metric_rows.items()},
            "gains": gain_rows}


def run_ophthalmology():
    manifest = ROOT / "data/manifests/benchmark_manifest.csv"
    model = FLAIRModel(vision_type="resnet_v1", bert_type="emilyalsentzer/Bio_ClinicalBERT",
                        proj_dim=512, from_checkpoint=True, caption="[CLS]")
    model.eval()
    model.to(device)
    resnet = model.vision_model.model

    text_input_ids, text_attention_mask = model.preprocess_text(OPHTH_DR_NORMAL_TERMS + OPHTH_ORACLE_UNSEEN_TERMS)
    with torch.no_grad():
        text_embeds = model.text_model(text_input_ids, text_attention_mask)
    text_embeds = F.normalize(text_embeds, dim=-1).cpu().numpy()
    known_embeds = text_embeds[:len(OPHTH_DR_NORMAL_TERMS)]
    oracle_embeds = text_embeds[len(OPHTH_DR_NORMAL_TERMS):]

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
        sim_oracle = float(np.mean(oracle_embeds @ emb))
        per_image.append({"group": row["group"], "emb": emb, "vocab_oracle": sim_oracle - sim_known})
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    return run_benchmark(per_image, "ophthalmology")


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
    oracle_embeds = encode_text(DERM_ORACLE_UNSEEN_TERMS)

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
        sim_oracle = float(np.mean(oracle_embeds @ emb))
        per_image.append({"group": row["group"], "emb": emb, "vocab_oracle": sim_oracle - sim_known})
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    return run_benchmark(per_image, "dermatology")


def main():
    ophth = run_ophthalmology()
    derm = run_dermatology()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"ophthalmology": ophth, "dermatology": derm}, f, indent=2)
    print(f"\nWritten to {OUT}", flush=True)


if __name__ == "__main__":
    main()
