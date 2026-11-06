# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202604026*
- **Họ và tên:*Võ Trần Công*
- **SEED** (in ở ô đầu notebook): 1016

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
  F1 test thường (SUB)  F1 50 case khó  \
1. BoW + LogReg                                0.919           0.316   
2. BoW + MLP                                   0.919           0.372   
3a. Emb + kNN                                  0.976           0.742   
3b. Emb + LogReg                               1.000           0.632   
3c. Emb + MLP head                             1.000           0.630   
4a. LLM nhỏ zero-shot                          0.974           0.862   
4d. LLM nhỏ + prompt phòng thủ                 0.974           0.889   

                                acc 50 case khó  
1. BoW + LogReg                           0.527  
2. BoW + MLP                              0.509  
3a. Emb + kNN                             0.709  
3b. Emb + LogReg                          0.618  
3c. Emb + MLP head                        0.636  
4a. LLM nhỏ zero-shot                     0.836  
4d. LLM nhỏ + prompt phòng thủ            0.873  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Dự đoán: F1 khoảng 0.4–0.5, precision cao hơn recall vì dữ liệu lệch.
Kết quả: F1 test 0.885 (precision 0.976, recall 0.81).
Vì: tôi nghĩ dữ liệu lệch thì chỉ số phải thấp. Thực ra dữ liệu lệch chỉ làm accuracy dễ gây hiểu lầm, còn spam SMS lại gần như tách tuyến tính bằng từ khoá ("free", "txt", "claim"...), nên một mô hình tuyến tính đơn giản đã làm tốt. Phần đúng là precision cao hơn recall, do bias âm làm mô hình thận trọng.

Dự đoán: đường gấp khúc, dao động liên tục.
Kết quả: chỉ một đỉnh nhọn (~2.5) ở vài epoch đầu, sau đó hội tụ nhanh về 0.005, thấp hơn cả lr = 0.5.
Vì: tôi chưa tính tới việc sigmoid bão hoà. Khi p gần 0 hoặc 1, error của tin đoán đúng gần bằng 0 nên gradient tự nhỏ lại. Thêm vào đó loss là hàm lồi và dữ liệu gần tách tuyến tính, nên sau một bước quá đà mô hình ổn định lại thay vì dao động mãi.

Dự đoán: BoW sai nhiều ở nhóm "Tưởng spam nhưng không phải" và "Nói về spam", vì các tin này chứa toàn từ khoá spam.
Kết quả: BoW đúng 100% ở cả hai nhóm, nhưng chỉ đúng 0–17% ở "Né từ khoá" và 10% ở "Spam giọng tự nhiên".
Vì: bias của BoW âm mạnh (khoảng −2.96), và các tin hiện đại này ít khớp với từ khoá spam cũ của bộ dữ liệu, nên BoW đoán gần như mọi tin là ham. Kết quả là nó đúng hết nhóm ham và sai gần hết nhóm spam. Accuracy 52% che mất điều đó, chỉ F1 0.33 mới cho thấy.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề 3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?
Cách làm: dùng lại build_mlp, train_mlp và predict_mlp. Mỗi cấu hình đặt lại torch.manual_seed(SEED), train 30 epoch, đo thời gian bằng time.perf_counter() và tính F1 trên test.
Giải thích: số tham số tăng khoảng 128 lần từ 4 lên 512 nơ-ron, nhưng F1 gần như không đổi, vì feature bag-of-words đã gần tách tuyến tính. Mạng lớn chỉ train lâu hơn và overfit sớm hơn

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Nên chọn: hệ thống lai, với BoW + MLP làm lớp chính.
Minh chứng: BoW + MLP có p95 2.3 ms (đạt SLA 50 ms), thông lượng khoảng 11.6k tin/s (dư cho 116 tin/s), F1 test 0.938. Encoder (p95 132–180 ms) và LLM (p95 2.6–3.2 s, khoảng 3,300 USD/tháng với model nhỏ) không đạt độ trễ.
Thiết kế: chuẩn hoá đầu vào và luật chặn trước, rồi BoW + MLP xử lý mọi tin. Tin lưỡng lự (p trong khoảng 0.3–0.7) chuyển sang Encoder + LogReg hoặc LLM xử lý bất đồng bộ.
Trade off: độ bền thấp trước spam kiểu mới (F1 trên case khó chỉ 0.40), và một phần nhỏ tin được kiểm tra muộn hơn. Bù lại bằng cách lấy mẫu cho người kiểm tra và train lại định kỳ.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
Phần	Hỏi AI	AI trả lời  Sửa / kiểm chứng gì
Phần 1	Gợi ý từng bước vì sao BoW đoán sai một tin spam	Câu hỏi gợi mở về trọng số, z và bias	Tự in trọng số từng token và document frequency để kiểm chứng
Phần 3.B	Dự đoán BoW và Encoder trên 6 tin khó	AI đoán BoW chỉ đúng 1–2/6	Kết quả thật BoW đúng 5/6, tôi phân tích lại nguyên nhân
