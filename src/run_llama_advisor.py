import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import json
import time
from collections import Counter, defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CSV_PATH = "/workspace/batch_out/detections.csv"
MODEL_DIR = "/workspace/tinyllama_pcb_merged"
OUT_TXT = "/workspace/batch_out/repair_guidance.txt"
OUT_JSON = "/workspace/batch_out/repair_guidance.json"

torch.set_num_threads(4)

def parse_detections(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing YOLO CSV: {csv_path}")
    per_image = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        img_col = next((c for c in fields if any(k in c.lower() for k in ["image", "file", "img"])), None)
        cls_col = next((c for c in fields if any(k in c.lower() for k in ["class", "label", "defect", "name"])), None)
        for row in reader:
            img = row[img_col] if img_col else "pcb_frame"
            lbl = row[cls_col] if cls_col else str(row)
            per_image[img].append(lbl.strip())
    return per_image

def main():
    per_image = parse_detections(CSV_PATH)
    if not per_image:
        print("[Stage 2] No defects detected in CSV — PCB PASS (LLM bypassed).")
        return

    t0 = time.time()
    print(f"[Stage 2 - OFFLINE] Loading local merged model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()
    print(f"[Stage 2 - OFFLINE] Ready in {time.time() - t0:.1f}s")

    results, report_lines = [], []
    for img_name, labels in per_image.items():
        counts = Counter(labels)
        summary = ", ".join(f"{cnt}x {lbl}" for lbl, cnt in counts.items())
        prompt = (
            "<|system|>\n"
            "You are an expert industrial PCB quality-control engineer. "
            "Provide concise severity, manufacturing root cause, and step-by-step repair instructions.</s>\n"
            "<|user|>\n"
            f"Inspection image: {img_name}\n"
            f"Detected defects ({len(labels)} total): {summary}\n"
            "Give: 1) Severity, 2) Likely root cause, 3) Operator repair steps.</s>\n"
            "<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        t_gen = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=False,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_sec = time.time() - t_gen
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        advice = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tok_s = len(new_tokens) / max(gen_sec, 1e-6)

        print(f"\n=== OFFLINE PCB ADVISORY: {img_name} ({len(new_tokens)} tok | {gen_sec:.1f}s | {tok_s:.2f} tok/s) ===")
        print(f"Defects: {summary}\n{advice}")

        results.append({
            "image": img_name,
            "total_detections": len(labels),
            "defect_counts": dict(counts),
            "generation_seconds": round(gen_sec, 2),
            "tokens_per_sec": round(tok_s, 2),
            "repair_guidance": advice,
        })
        report_lines.append(f"Image: {img_name}\nDefects: {summary}\n{advice}\n{'-'*60}\n")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Stage 2 - OFFLINE] Saved: {OUT_TXT} and {OUT_JSON}")

if __name__ == "__main__":
    main()
