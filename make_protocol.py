# Đọc danh sách file từ flac.scp
with open("RADAR2026-dev/flac.scp", "r", encoding="utf-8") as f_in:
    lines = f_in.readlines()

# Tạo file protocol.txt (Cột 1: ID, Cột 2: Nhãn "spoof")
with open("RADAR2026-dev/protocol.txt", "w", encoding="utf-8") as f_out:
    for line in lines:
        if line.strip():
            utt_id = line.strip().split()[0] 
            f_out.write(f"{utt_id} spoof\n")

print("Đã tạo xong file protocol.txt với 100% nhãn spoof!")