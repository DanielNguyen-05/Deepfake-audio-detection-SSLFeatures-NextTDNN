import os
from pathlib import Path

input_file = "/Users/dangnguyen/Desktop/RADAR26/Deepfake-audio-detection-SSLFeatures-NextTDNN/models/amf_hubert_wavlm_nextdnn_eca_L8_Light_ASVSpoof5/eval_scores.txt"
output_dir = "submissions/RADAR2026-dev"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "score.tsv")

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write("filename\tscore\n")
    
    lines = f_in.readlines()
    # Sắp xếp theo uttid
    lines.sort(key=lambda x: x.split()[0])
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            uid = parts[0]
            fake_score = parts[1]          # <-- fake score (cột 2)
            f_out.write(f"{uid}\t{fake_score}\n")

print(f"✅ Submission file đã sẵn sàng: {output_file}")