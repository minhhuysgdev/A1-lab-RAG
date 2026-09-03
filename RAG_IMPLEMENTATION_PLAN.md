# Kế hoạch triển khai hệ thống RAG trên dataset Tiki Thời trang nữ

> Mục tiêu: xây dựng H1 làm baseline, sau đó đo lường mức cải thiện của H2 (Hybrid retrieval + metadata filtering) và H3 (Multi-modal RAG) **so với cùng một baseline, trên cùng một bộ đánh giá**.

---

## 0. Hiện trạng dữ liệu (đã khảo sát thực tế)

| Chỉ số | Giá trị | Ảnh hưởng tới thiết kế |
|---|---|---|
| Số bản ghi thô | 1.463 | |
| **Bản ghi trùng lặp hoàn toàn** | **151** (trùng `product_id`, mọi trường giống hệt) | Corpus thật chỉ còn **1.312** sản phẩm — bắt buộc dedup trước khi index, nếu không kết quả đánh giá sẽ bị thổi phồng |
| Tên trùng | 161 | Cần kiểm tra riêng (khác `product_id` nhưng cùng tên) |
| Danh mục | 15 danh mục con, ~100 sp/danh mục | Tập cân bằng, thuận lợi cho đánh giá theo danh mục |
| Độ dài `description` | TB 1.108, max 3.924 ký tự | **Không cần chunking** — 1 sản phẩm = 1 document |
| `short_description` | Là tiền tố của `description` ở 1.097/1.463 (75%) | Không index cả hai, gây trùng lặp vô ích |
| `configurable_options` | **110+ biến thể tên** cho ~3 khái niệm | Đây là mỏ vàng của H2 nhưng **bắt buộc phải chuẩn hoá** |
| Sản phẩm không có option | 113 | Cần fallback khi lọc theo thuộc tính |
| `origin` | 23 giá trị, 46 sp rỗng | Chuẩn hoá được về danh sách quốc gia |
| `image_urls` | 56.630 URL thô → **15.227 ảnh unique** | Trùng 73% do nhiều biến thể kích thước cache của cùng 1 ảnh — quyết định chi phí của H3 |
| Trường bị thiếu | **Không có giá, rating, lượt bán, tồn kho** | Mọi truy vấn về giá/độ phổ biến đều bất khả thi — phải loại khỏi phạm vi hoặc crawl bổ sung |

### Đặc thù ngôn ngữ
Toàn bộ text **đã qua tách từ** (`Đầm sơ_mi xanh mint`). Đây vừa là ràng buộc vừa là cơ hội:
- BM25 hoạt động **tốt hơn** trên text đã tách từ (token là đơn vị ngữ nghĩa thật).
- Embedding model như `bge-m3` được train trên text tự nhiên → cần **bỏ underscore**.
- Nhưng `dangvantuan/vietnamese-embedding` (nền PhoBERT) lại **yêu cầu input đã tách từ** → giữ nguyên underscore.

→ Phải sinh **hai phiên bản text** cho mỗi sản phẩm và chọn đúng phiên bản cho từng model.

### Môi trường hiện có
Python 3.13.11, `torch` 2.10, `numpy`, `pandas`, **`streamlit` 1.51.0 (đã cài sẵn)**.

Đã kiểm tra khả năng cài trên Python 3.13 — tất cả đều có wheel:

| Gói | Phiên bản | Vai trò |
|---|---|---|
| `chromadb` | 1.5.9 | Vector DB chính |
| `faiss-cpu` | 1.15.0 | Vector DB đối chiếu |
| `sentence-transformers` | 6.0.1 | Embedding + reranker |
| `rank-bm25` | 0.2.2 | BM25 cho H2 |
| `streamlit` | 1.51.0 ✅ | Frontend |

---

## Phase 0 — Chuẩn bị dữ liệu (dùng chung cho cả H1/H2/H3)

**Thời lượng: 1–1,5 ngày. Đây là phần quyết định chất lượng của cả ba hướng.**

### 0.1 Dedup
```
Loại 151 bản ghi trùng product_id (giữ bản đầu tiên) → 1.312 sản phẩm
Kiểm tra 161 title trùng: nếu khác product_id nhưng cùng shop + cùng mô tả → gộp
```

### 0.2 Làm sạch `description`
Vấn đề: mô tả chứa đoạn boilerplate lặp lại (chính sách thuế, phí vận chuyển, hàng cồng kềnh) không mang thông tin sản phẩm nhưng chiếm tỉ trọng lớn trong embedding.

Cách làm — **tự động, không hard-code**:
1. Tách `description` thành câu.
2. Đếm tần suất mỗi câu (đã chuẩn hoá) trên toàn corpus.
3. Câu xuất hiện ở **> 5% số sản phẩm** → xếp vào boilerplate, loại bỏ.
4. Lưu danh sách câu bị loại ra file riêng để kiểm tra thủ công (tránh loại nhầm câu mô tả hợp lệ như "Chất liệu cotton").

### 0.3 Chuẩn hoá `configurable_options` ⭐
Đây là bước có giá trị cao nhất cho H2. 110+ tên thô cần gom về **3 thuộc tính chuẩn**:

| Canonical | Gom từ các biến thể |
|---|---|
| `color` | Màu, Màu sắc, MÀU SẮC, Bảng màu, Chọn màu, Màu Áo, Nhóm màu, Màu chủ đạo, Màu săc, Máu sắc… |
| `size` | Size, SIZE, Kích cỡ, Kích thước, Bảng Size, Chọn size, Size Áo, Lựa size, KÍCH CỠ… |
| `variant` | Phân loại, Loại, Kiểu dáng, Họa tiết, Set, Combo, Bộ… (phần còn lại) |

Quy tắc chuẩn hoá: `strip()` → `lower()` → bỏ dấu câu → bỏ dấu tiếng Việt → khớp regex (`mau|color` / `size|kich\s*(co|thuoc)` / còn lại là `variant`).

Sau đó chuẩn hoá **giá trị**: gom `S/M/L/XL/XXL/Freesize`, và nhóm màu về bảng màu cơ bản (đen/trắng/đỏ/xanh…). Giữ cả `raw_value` để không mất thông tin.

### 0.4 Sinh hai phiên bản text
```python
text_segmented = title + breadcrumbs + brand + description   # giữ underscore → cho BM25 & PhoBERT
text_natural   = text_segmented.replace('_', ' ')            # bỏ underscore → cho bge-m3
```

### 0.5 Chuẩn hoá `origin` và dedup `image_urls`
- `origin`: gom 23 giá trị về danh sách quốc gia chuẩn; 46 giá trị rỗng → `unknown`.
- `image_urls`: bỏ tiền tố kích thước cache (`/cache/w1200/`, `/cache/200x280/` → `/cache/`) rồi lấy `set()`, ưu tiên giữ bản `w1200`. **56.630 → 15.227 URL.**

**Đầu ra Phase 0:** `products_clean.jsonl` — 1.312 dòng, mỗi dòng có đủ: `product_id`, `title`, `text_segmented`, `text_natural`, `category_l2`, `brand`, `origin_norm`, `attrs.{color,size,variant}`, `image_urls_unique`.

---

## Phase 0.5 — Xây bộ đánh giá (BẮT BUỘC làm trước H1)

**Thời lượng: 1 ngày.** Không có bộ này thì không thể chứng minh H2/H3 tốt hơn H1 — mọi so sánh sẽ chỉ là cảm tính.

Xây **~120 truy vấn** chia 3 loại:

| Loại | Số lượng | Cách sinh | Gold label |
|---|---|---|---|
| **A. Thuộc tính** | 40 | Tổ hợp từ metadata đã chuẩn hoá: *"áo thun nữ màu đen size L"*, *"chân váy thương hiệu Azuno"* | Toàn bộ sản phẩm khớp bộ lọc (đánh giá dạng set) |
| **B. Ngữ nghĩa (known-item)** | 60 | LLM đọc `description` của 1 sản phẩm → sinh câu hỏi tự nhiên của người mua | Chính sản phẩm đó |
| **C. Hình ảnh** | 20 | Lấy ảnh sản phẩm làm truy vấn, hoặc mô tả thị giác (*"váy hoa nhí nền trắng tay bồng"*) | Sản phẩm tương ứng (chỉ dùng cho H3) |

### Chống rò rỉ ở loại B — điểm dễ sai nhất
Nếu để LLM sinh câu hỏi tự do, nó sẽ **copy nguyên cụm từ hiếm** trong mô tả (VD tên brand `Haint_Boutique`). Khi đó BM25 đạt Recall@1 ≈ 100% một cách giả tạo và kết luận "H2 tốt hơn" sẽ vô nghĩa.

**Ràng buộc bắt buộc khi sinh:**
- Cấm nhắc tên thương hiệu và `product_id`.
- Diễn đạt lại theo giọng người mua thật ("mình cần váy đi tiệc mùa hè, vải mát").
- Không dùng lại cụm >3 từ liên tiếp có trong mô tả gốc.
- **Rà thủ công 100% loại B** — 60 câu là làm được, và đây là phần đáng đầu tư nhất.

### Metric thống nhất cho cả 3 hướng
- **Retrieval:** Recall@1, Recall@5, Recall@10, MRR@10, nDCG@10
- **Generation:** Faithfulness (câu trả lời có bịa không), Answer Relevance — chấm bằng LLM-as-judge trên ~30 mẫu, cộng rà tay
- **Vận hành:** latency p50/p95, chi phí index, chi phí mỗi truy vấn

> Toàn bộ ba hướng dùng **cùng bộ query, cùng metric, cùng LLM sinh câu trả lời, cùng top-k**. Chỉ thay đổi khối retrieval. Đây là điều kiện tiên quyết để so sánh có giá trị.

---

## Phase 1 — Kiến trúc hệ thống và giao diện

**Thời lượng: 1 ngày (dựng khung) + tích hợp dần theo từng hướng.**

### Chọn vector DB: Chroma làm chính, FAISS để đối chiếu

Đây là quyết định quan trọng vì nó ảnh hưởng trực tiếp tới độ khả thi của H2:

| | Chroma | FAISS |
|---|---|---|
| **Lọc metadata** | **Có sẵn** (`where={"color": "đen"}`) | **Không có** — phải tự lọc thủ công ngoài index |
| Lưu trữ | Bền vững, tự động (DuckDB/SQLite) | Phải tự quản lý serialize `.index` + mapping id |
| Lưu document + metadata | Có sẵn | Chỉ lưu vector, phải tự giữ store song song |
| Tốc độ ở quy mô lớn | Tốt | Tốt hơn (>1M vector) |
| Phù hợp corpus 1.312 sp | ✅ | Thừa năng lực |

> **Khuyến nghị: dùng Chroma làm hệ chính.** H2 cần lọc theo `color` / `size` / `brand` / `category` — Chroma hỗ trợ native qua mệnh đề `where`, còn với FAISS bạn sẽ phải tự viết lớp lọc bên ngoài rồi xử lý bài toán "lọc trước hay tìm trước", vốn là một nguồn lỗi không cần thiết ở quy mô này.
>
> Vẫn nên **triển khai FAISS như một backend thay thế** (cùng interface) để báo cáo có phần so sánh hai vector DB về tốc độ index, latency truy vấn và mức tiêu thụ RAM. Đây là nội dung so sánh có giá trị mà không tốn nhiều công, vì lớp trừu tượng dưới đây cho phép hoán đổi bằng một dòng cấu hình.

### Cấu trúc dự án

```
RAG/
├── data/
│   ├── product_tiki_data.json          # gốc
│   ├── products_clean.jsonl            # sau Phase 0 (1.312 sp)
│   ├── eval_set.json                   # 120 query
│   └── images/                         # ảnh tải về cho H3
├── backend/
│   ├── config.py                       # chọn model, vector DB, top-k
│   ├── preprocess.py                   # Phase 0: dedup, làm sạch, chuẩn hoá
│   ├── embedder.py                     # nạp model, encode text/ảnh (có cache)
│   ├── vectorstore/
│   │   ├── base.py                     # interface trừu tượng
│   │   ├── chroma_store.py             # backend Chroma
│   │   └── faiss_store.py              # backend FAISS
│   ├── retrievers/
│   │   ├── dense.py                    # H1
│   │   ├── bm25.py                     # H2
│   │   ├── hybrid.py                   # H2: RRF + filter + rerank
│   │   └── multimodal.py               # H3
│   ├── filters.py                      # H2: bóc filter từ query bằng LLM
│   ├── generator.py                    # prompt + gọi LLM sinh câu trả lời
│   └── pipeline.py                     # điều phối: query → retrieve → generate
├── frontend/
│   └── app.py                          # Streamlit
├── eval/
│   ├── build_eval_set.py
│   ├── run_eval.py                     # chạy 1 cấu hình → metrics
│   └── results/                        # kết quả từng lần chạy
└── requirements.txt
```

**Nguyên tắc thiết kế quan trọng:** `pipeline.py` nhận một object cấu hình chọn retriever (`dense` / `hybrid` / `multimodal`) và vector store (`chroma` / `faiss`). Cả **Streamlit và script đánh giá đều gọi chung một `pipeline`** — nhờ vậy con số hiển thị trên giao diện luôn khớp với con số trong báo cáo. Nếu để UI và eval đi hai đường code khác nhau, bạn sẽ không bao giờ chắc chúng đo cùng một thứ.

### Giao diện Streamlit

Xây **một app duy nhất**, mở rộng dần qua ba hướng thay vì viết ba app riêng.

**Bố cục:**
```
┌─ Sidebar ────────────┐┌─ Main ─────────────────────────────┐
│ Chế độ:              ││  [ Ô nhập truy vấn        ] [Tìm]  │
│  ○ H1 Dense          ││                                     │
│  ○ H2 Hybrid         ││  💬 Câu trả lời của LLM             │
│  ○ H3 Multi-modal    ││     (kèm trích dẫn product_id)      │
│  ○ So sánh cạnh nhau ││  ─────────────────────────────────  │
│                      ││  📦 Kết quả truy xuất               │
│ Vector DB:           ││  ┌────┬────────────────────────┐   │
│  ○ Chroma  ○ FAISS   ││  │ảnh │ Tên sản phẩm           │   │
│                      ││  │    │ Brand · Danh mục       │   │
│ Bộ lọc:              ││  │    │ Màu · Size · Xuất xứ   │   │
│  Danh mục [15 mục ▾] ││  │    │ điểm: 0.82  [Chi tiết] │   │
│  Thương hiệu   [▾]   ││  └────┴────────────────────────┘   │
│  Màu · Size    [▾]   ││                                     │
│  Xuất xứ       [▾]   ││  ⏱ 340ms · 5 kết quả · hybrid+rerank│
│                      ││                                     │
│ top-k: ──●──── 5     ││                                     │
│ ☑ Bật reranker       ││                                     │
│ 🖼 Tải ảnh lên (H3)  ││                                     │
└──────────────────────┘└─────────────────────────────────────┘
```

**Yêu cầu chức năng:**

| # | Tính năng | Thuộc hướng | Ghi chú |
|---|---|---|---|
| U1 | Ô nhập truy vấn + hiển thị câu trả lời có trích dẫn | H1 | Trích dẫn phải bấm được, cuộn tới sản phẩm tương ứng |
| U2 | Thẻ kết quả kèm ảnh, metadata, điểm số | H1 | Ảnh lấy từ `image_urls` (bản `w1200`) |
| U3 | Hiện latency + cấu hình đang chạy | H1 | Minh bạch để đối chiếu với báo cáo |
| U4 | Bộ lọc metadata ở sidebar | H2 | Đổ từ `products_clean.jsonl`, đã chuẩn hoá |
| U5 | Hiện **filter mà LLM bóc được** từ truy vấn | H2 | Cho người dùng sửa lại nếu LLM bóc sai — vừa hữu ích vừa là công cụ debug |
| U6 | **Chế độ so sánh cạnh nhau** | H2 | Hai cột H1 vs H2 cho cùng truy vấn ⭐ |
| U7 | Bật/tắt từng thành phần (BM25, filter, reranker) | H2 | Cho phép demo trực quan bảng ablation |
| U8 | Tải ảnh lên để tìm sản phẩm tương tự | H3 | `st.file_uploader` |
| U9 | Lưới ảnh sản phẩm tương tự | H3 | `st.columns` + `st.image` |
| U10 | Nút gửi phản hồi 👍/👎 cho mỗi kết quả | Tất cả | Ghi ra file → làm dữ liệu phân tích lỗi |

**U6 (so sánh cạnh nhau) là tính năng đáng đầu tư nhất** — nó biến bảng số liệu khô khan thành thứ nhìn thấy được khi bảo vệ, và trong lúc phát triển nó chính là công cụ giúp bạn nhanh chóng tìm ra những truy vấn mà H2 thua H1.

**Lưu ý kỹ thuật Streamlit:**
- Bọc `@st.cache_resource` cho việc nạp embedding model và kết nối vector DB — nếu không, mỗi lần bấm nút Streamlit sẽ nạp lại model từ đầu và app sẽ chậm không dùng được.
- `@st.cache_data` cho kết quả truy vấn, khoá theo `(query, config)`.
- Dùng `st.session_state` giữ lịch sử truy vấn để so sánh nhiều lượt.
- Nạp model **một lần lúc khởi động**, không nạp trong hàm xử lý sự kiện.

---

## H1 — Baseline: Dense RAG thuần

**Thời lượng: 1–2 ngày.** Mục tiêu là **một baseline trung thực, không tối ưu quá tay** — vì đây là mốc để H2/H3 vượt qua.

### Kiến trúc
```
[Streamlit] Query → [Backend] Embed → Chroma similarity_search(k=5)
          → Prompt + context → LLM → Câu trả lời + trích nguồn product_id → [Streamlit] Hiển thị
```

### Các bước

**B1. Chọn embedding model** — chạy thử 2 ứng viên trên bộ eval, chọn cái tốt hơn:

| Model | Input | Dim | Ghi chú |
|---|---|---|---|
| `AITeamVN/Vietnamese_Embedding` | `text_natural` | 1024 | Nền bge-m3, fine-tune tiếng Việt |
| `dangvantuan/vietnamese-embedding` | `text_segmented` | 768 | Nền PhoBERT, **cần input đã tách từ** |

Đây cũng là dịp kiểm chứng giả thuyết: dữ liệu đã tách từ có lợi cho model PhoBERT hay không.

**B2. Ablation chọn trường để embed** — chạy 4 cấu hình, báo cáo Recall@5:
1. Chỉ `title`
2. `title + breadcrumbs`
3. `title + breadcrumbs + brand + description` (mặc định)
4. Như (3) nhưng **chưa** loại boilerplate → dùng để **định lượng giá trị của Phase 0.2**

**B3. Index vào Chroma** — tạo collection `products_h1` với `cosine` làm hàm khoảng cách. Mỗi document nạp kèm **đầy đủ metadata đã chuẩn hoá** (`category_l2`, `brand`, `color`, `size`, `origin_norm`) — dù H1 chưa dùng tới, việc nạp sẵn giúp H2 tái sử dụng ngay collection này mà không phải index lại.

> Chroma lưu embedding bền vững trên đĩa (`PersistentClient`), nên chỉ encode một lần. Với 1.312 sản phẩm, quá trình index mất khoảng 1–2 phút trên CPU.

**B4. Cài `faiss_store.py`** — cùng interface `base.py`, dùng `IndexFlatIP` trên vector đã chuẩn hoá L2 (tương đương cosine). Lưu `.index` + file mapping `id ↔ product_id`. Chỉ để đối chiếu hiệu năng, không phải hệ chính.

**B5. Sinh câu trả lời** — prompt buộc trả lời **chỉ dựa trên context**, trả về `product_id` làm trích dẫn, và **được phép nói "không tìm thấy"**. Nếu bỏ điều kiện cuối, model sẽ bịa và làm hỏng metric faithfulness.

**B6. Streamlit v1** — dựng U1, U2, U3. Đây đã là một demo chạy được đầu-cuối, và có nó sớm giúp phát hiện lỗi dữ liệu bằng mắt nhanh hơn nhiều so với đọc log.

### Deliverable H1
- `baseline_results.json` (điểm từng query) + bảng ablation + phân tích ~20 ca lỗi phân loại theo nguyên nhân. **Bảng lỗi này chính là đầu vào thiết kế cho H2.**
- **App Streamlit chạy được** với chế độ H1.
- **Bảng so sánh Chroma vs FAISS:** thời gian index, latency p50/p95, RAM tiêu thụ, dung lượng đĩa.

### Dự đoán điểm yếu (cần xác nhận bằng số liệu)
- Truy vấn có ràng buộc cứng (size/màu/brand) → dense không "hiểu" ràng buộc, chỉ đo tương đồng chung.
- Tên brand hiếm → embedding làm nhoè token hiếm.
- 15 danh mục có mô tả từ vựng giống nhau → nhầm lẫn chéo danh mục.

---

## H2 — Hybrid retrieval + Metadata filtering

**Thời lượng: 4–6 ngày.** Xây **cộng dồn** trên H1, mỗi thành phần bật/tắt độc lập được để đo đóng góp riêng.

### Kiến trúc
```
Query ─┬→ [LLM] bóc filter ──────→ Lọc cứng trên metadata (thu hẹp candidate)
       │                                        ↓
       ├→ BM25 (text_segmented) ──┐      candidate pool
       └→ Dense (H1)  ────────────┴→ RRF → Top-30 → Cross-encoder rerank → Top-5 → LLM
```

### Các thành phần

**C1. BM25** — dùng `rank_bm25` (BM25Okapi) trên `text_segmented`, tokenize bằng `split()` vì text đã tách từ sẵn. Tune `k1`, `b`.

**C2. Hợp nhất RRF** — `score = Σ 1/(60 + rank_i)`. Chọn RRF thay vì cộng điểm có trọng số vì **không cần chuẩn hoá thang điểm** giữa BM25 (không chặn trên) và cosine (`[-1,1]`) — tránh một nguồn lỗi phổ biến.

**C3. Bóc filter từ truy vấn** — LLM structured output trả về:
```json
{"category": "Áo thun nữ", "color": "đen", "size": "L", "brand": null, "origin": null}
```
Ánh xạ thẳng sang mệnh đề `where` của Chroma:
```python
where = {"$and": [{"category_l2": "Áo thun nữ"}, {"color": "đen"}, {"size": "L"}]}
collection.query(query_embeddings=[emb], n_results=30, where=where)
```

**Nguyên tắc an toàn:** áp filter dạng *soft* — nếu sau khi lọc còn < 5 kết quả thì **nới dần** (bỏ ràng buộc yếu nhất trước) và **báo cho người dùng biết ràng buộc nào đã bị nới**. Nhớ 113 sản phẩm không có option nào; lọc cứng theo size sẽ loại oan chúng.

> Đây chính là chỗ Chroma trả công: cùng chức năng này với FAISS đòi hỏi tự lấy dư kết quả rồi lọc sau (`over-fetch then filter`), và phải tự đoán cần lấy dư bao nhiêu mới đủ — với filter chặt thì có khi lấy 500 vẫn không còn kết quả nào.

**C4. Reranking** — cross-encoder `AITeamVN/Vietnamese_Reranker` hoặc `BAAI/bge-reranker-v2-m3` trên top-30. Đây thường là thành phần cho lợi ích lớn nhất trên mỗi đơn vị công sức.

**C5. Streamlit v2** — bổ sung U4 → U7. Trong đó:
- **U5** hiện bộ filter LLM bóc được dưới dạng các chip sửa được — vừa minh bạch với người dùng, vừa là công cụ debug tốt nhất cho C3.
- **U7** cho phép bật/tắt từng thành phần ngay trên UI, tức là **demo trực tiếp bảng ablation** thay vì chỉ chiếu bảng số.
- **U6** hiển thị hai cột H1 | H2 cho cùng một truy vấn, tô màu những sản phẩm chỉ xuất hiện ở một bên.

### Thiết kế thí nghiệm so sánh với baseline
Chạy **ablation cộng dồn** để biết cải thiện đến từ đâu, không chỉ biết "H2 tốt hơn":

| # | Cấu hình | Recall@5 | MRR@10 | Δ vs H1 | Latency |
|---|---|---|---|---|---|
| 1 | H1 Dense thuần (baseline) | — | — | — | — |
| 2 | BM25 thuần | | | | |
| 3 | Dense + BM25 (RRF) | | | | |
| 4 | (3) + metadata filter | | | | |
| 5 | (4) + reranker | | | | |

Đồng thời **tách kết quả theo loại query A và B**. Giả thuyết cần kiểm chứng: H2 thắng đậm ở loại A (thuộc tính) nhưng chỉ hơn nhẹ ở loại B (ngữ nghĩa). Nếu số liệu cho thấy điều đó, đây chính là kết luận có giá trị nhất của phần H2.

### Đề xuất cải tiến (làm sau khi có số liệu ablation)
1. **Query expansion** — sinh 2–3 biến thể truy vấn rồi hợp nhất kết quả; xử lý tốt trường hợp người dùng viết không dấu hoặc dùng từ địa phương.
2. **Trọng số theo trường** — BM25 riêng cho `title` và cho `description`, `title` nhân trọng số cao hơn (khớp ở tiêu đề đáng tin hơn ở mô tả).
3. **Định tuyến truy vấn** — phân loại truy vấn thành *thuộc tính* / *ngữ nghĩa* / *hỗn hợp*, chỉ chạy nhánh cần thiết → giảm latency mà gần như không mất chất lượng.
4. **Sinh câu hỏi giả để làm giàu index (HyDE ngược)** — với mỗi sản phẩm, sinh sẵn 3 câu hỏi mà nó trả lời được, embed và index thêm; thu hẹp khoảng cách giữa giọng văn truy vấn và giọng văn mô tả.
5. **Nhóm biến thể** — 161 title trùng cho thấy có sản phẩm gần trùng; gom nhóm trước khi trả kết quả để top-5 không bị lấp đầy bởi các biến thể của cùng một mẫu.

---

## H3 — Multi-modal RAG

**Thời lượng: 1,5–2,5 tuần.** Rủi ro cao nhất — **nên chốt xong H1 + H2 trước khi bắt đầu.**

### Vì sao đáng làm với dataset này
Thời trang là lĩnh vực mà mô tả text diễn đạt rất kém về kiểu dáng, hoạ tiết, phom. Truy vấn *"váy hoa nhí tay bồng dáng xoè"* gần như không thể khớp bằng text nếu người bán không viết đúng những từ đó — mà mô tả trên Tiki thì rất tuỳ tiện.

### Các bước

**D1. Thu thập ảnh** *(rủi ro chính)*
- Dedup theo Phase 0.5: **56.630 → 15.227 URL** (tiết kiệm 73% băng thông và thời gian).
- Giới hạn thêm: **tối đa 5 ảnh/sản phẩm** (hiện TB 10,4, max 121 ảnh — cái đuôi này vô ích) → còn **~6.000 ảnh**.
- Tải bất đồng bộ có giới hạn tốc độ + retry + cache đĩa. Ghi log ảnh lỗi/404.
- **Bắt buộc có phương án dự phòng:** nếu tỉ lệ tải thành công < 80%, thu hẹp H3 xuống một subset 300–500 sản phẩm có ảnh đầy đủ và nêu rõ giới hạn này trong báo cáo. Kết quả trên subset vẫn hợp lệ nếu được trình bày trung thực.

**D2. Chọn model** — `jinaai/jina-clip-v2`: đa ngôn ngữ (89 thứ tiếng, **có tiếng Việt**), chung không gian nhúng cho ảnh và text, hỗ trợ Matryoshka nên cắt chiều được.

> Cảnh báo: phần lớn CLIP/SigLIP là **model tiếng Anh**. Nếu dùng chúng, phải dịch truy vấn Việt → Anh, và bước dịch này sẽ thành một nguồn lỗi mới, làm nhiễu phép so sánh với baseline. Chọn model đa ngữ ngay từ đầu để tránh.

**D3. Chiến lược index** — chạy **2 phương án**, so sánh:
- **(a) Late fusion:** giữ nguyên index text của H1, thêm index ảnh riêng, hợp nhất bằng RRF ở tầng kết quả. *Ưu điểm: không phá H1, dễ bật/tắt.*
- **(b) Unified embedding:** với mỗi sản phẩm, tính `emb = α·emb_text + (1−α)·mean(emb_images)`, quét `α ∈ {0.3, 0.5, 0.7}`. *Ưu điểm: một index duy nhất, truy vấn nhanh.*

**D3b. Lưu vào Chroma** — tạo collection riêng `products_h3_image`. Vì Chroma cho phép nạp embedding tính sẵn (`add(embeddings=...)`), có thể dùng chung hạ tầng với H1/H2 mà không cần thư viện mới. Phương án (a) dùng 2 collection rồi RRF ở tầng ứng dụng; phương án (b) dùng 1 collection với vector đã trộn.

**D4. Bổ sung mô tả thị giác (tuỳ chọn, giá trị cao)** — dùng vision LLM sinh mô tả thị giác chuẩn hoá cho ảnh chính (*"váy midi, hoạ tiết hoa nhí, nền trắng, tay bồng, cổ vuông"*), rồi **đưa mô tả đó vào index text của H2**. Cách này biến thông tin hình ảnh thành thứ mà BM25 và dense đều dùng được, thường hiệu quả hơn CLIP thuần cho truy vấn text. Chi phí: ~1.312 lần gọi vision model.

### Thiết kế so sánh với baseline
H3 **không** thay thế H1 mà mở rộng năng lực. Cần báo cáo tách bạch hai phần:

| Nhóm truy vấn | H1 | H3 (a) late fusion | H3 (b) unified | H3 + mô tả thị giác |
|---|---|---|---|---|
| Loại B — ngữ nghĩa text | baseline | | | |
| Loại C — thị giác/kiểu dáng | *(H1 không làm được)* | | | |
| Ảnh → sản phẩm | *(không hỗ trợ)* | | | |

**Điểm phải trung thực trong báo cáo:** rất có khả năng H3 **kém hơn H1** ở nhóm truy vấn text thuần (vì vector ảnh làm loãng tín hiệu text), nhưng **mở ra năng lực hoàn toàn mới** ở nhóm C. Đây là một kết luận hợp lệ và thú vị — đừng cố ép H3 phải thắng ở mọi chỉ số.

**D5. Streamlit v3** — bổ sung U8, U9: `st.file_uploader` nhận ảnh tải lên, encode bằng cùng model `jina-clip-v2`, truy vấn collection ảnh, hiển thị kết quả dạng lưới bằng `st.columns`. Đây là tính năng gây ấn tượng nhất khi demo vì nó làm được điều mà baseline hoàn toàn không làm được.

---

## Lộ trình tổng thể

| Giai đoạn | Thời lượng | Đầu ra | Chặn bởi |
|---|---|---|---|
| Phase 0 — Làm sạch dữ liệu | 1–1,5 ngày | `products_clean.jsonl` (1.312 sp) | — |
| Phase 0.5 — Bộ đánh giá | 1 ngày | `eval_set.json` (120 query) | Phase 0 |
| **Phase 1 — Khung kiến trúc** | 1 ngày | `vectorstore/` + `pipeline.py` + khung Streamlit | Phase 0 |
| **H1 — Baseline** | 2–3 ngày | Điểm baseline + ablation + **Streamlit v1** + so sánh Chroma/FAISS | Phase 0.5, Phase 1 |
| **H2 — Hybrid + filter** | 5–7 ngày | Bảng ablation 5 dòng + đề xuất cải tiến + **Streamlit v2 (so sánh cạnh nhau)** | H1 |
| **H3 — Multi-modal** | 2–3 tuần | So sánh theo nhóm truy vấn + **Streamlit v3 (tìm bằng ảnh)** | H1 (H2 tuỳ chọn) |
| Tổng hợp báo cáo | 2 ngày | Báo cáo so sánh + demo trực tiếp | Tất cả |

**Đường găng:** Phase 0 → Phase 0.5 → Phase 1 → H1. Bốn bước này chặn mọi thứ phía sau, nên làm tuần tự và làm kỹ. H2 và H3 sau đó có thể chạy song song nếu có nhiều người.

> **Lưu ý về ước lượng:** thời lượng mỗi hướng đã cộng thêm ~1 ngày so với phiên bản trước để tính phần giao diện. Phase 1 tuy tốn 1 ngày nhưng **tiết kiệm lại nhiều hơn thế** ở H2 và H3, vì lớp trừu tượng vector store và pipeline dùng chung khiến việc thêm một hướng mới chỉ là viết một retriever rồi khai báo, không phải sửa xuyên suốt code.

---

## Rủi ro và phương án xử lý

| Rủi ro | Khả năng | Tác động | Xử lý |
|---|---|---|---|
| Query loại B bị rò rỉ từ vựng → BM25 thắng giả tạo | **Cao** | **Cao** — làm sai lệch toàn bộ kết luận H2 | Ràng buộc khi sinh + rà tay 100% (60 câu) |
| Tải ảnh thất bại / bị chặn tốc độ | Trung bình | Cao — chặn H3 | Dedup trước, giới hạn 5 ảnh/sp, có phương án subset |
| Không có trường giá → demo thiếu thuyết phục | Cao | Trung bình | Nêu rõ giới hạn ngay từ đầu báo cáo; hoặc crawl bổ sung sớm |
| Chuẩn hoá `configurable_options` gom nhầm | Trung bình | Trung bình | Giữ `raw_value`; rà tay bảng ánh xạ (chỉ ~110 dòng) |
| Loại nhầm câu hợp lệ khi lọc boilerplate | Trung bình | Trung bình | Xuất danh sách câu bị loại để duyệt trước khi áp dụng |
| Corpus quá nhỏ (1.312) → chênh lệch không có ý nghĩa thống kê | Trung bình | Cao | Báo cáo khoảng tin cậy bootstrap; dùng paired test trên cùng bộ query |
| Python 3.13 thiếu wheel cho một số thư viện | Thấp | Thấp | **Đã kiểm tra: chromadb 1.5.9, faiss-cpu 1.15.0, sentence-transformers 6.0.1, streamlit 1.51 đều có wheel cho 3.13** |
| Streamlit nạp lại model mỗi lần tương tác → app không dùng được | **Cao** | Trung bình | Bắt buộc `@st.cache_resource` cho model và kết nối DB ngay từ đầu — đây là lỗi phổ biến nhất khi làm RAG trên Streamlit |
| UI và script eval cho kết quả khác nhau | Trung bình | Cao — mất tin cậy khi bảo vệ | Cả hai gọi chung `pipeline.py`, tuyệt đối không nhân bản logic truy xuất |
| Chroma đổi API giữa các phiên bản (1.x có breaking change so với 0.4/0.5) | Trung bình | Trung bình | Ghim phiên bản trong `requirements.txt`; cô lập mọi lệnh gọi trong `chroma_store.py` |

---

## Phụ lục — `requirements.txt` dự kiến

```
# Đã có sẵn
streamlit==1.51.0
torch==2.10.0
numpy
pandas

# Vector DB
chromadb==1.5.9
faiss-cpu==1.15.0          # backend đối chiếu

# Retrieval
sentence-transformers==6.0.1
rank-bm25==0.2.2
transformers

# H3 - multimodal
pillow
httpx                       # tải ảnh bất đồng bộ

# Tiện ích
python-dotenv
tqdm
```

Khởi chạy: `streamlit run frontend/app.py`

---

## Nguyên tắc xuyên suốt

1. **Một biến tại một thời điểm.** Mọi so sánh giữ nguyên bộ query, LLM sinh, top-k; chỉ đổi khối retrieval.
2. **Cố định seed và lưu lại toàn bộ output từng lần chạy**, để kết quả tái lập được.
3. **Ghi nhận cả kết quả âm.** "Metadata filter không cải thiện gì ở truy vấn ngữ nghĩa" là một phát hiện có giá trị, không phải thất bại.
4. **Corpus nhỏ nên chênh lệch vài phần trăm có thể chỉ là nhiễu.** Chênh 1–2 sản phẩm trên 120 query đã đổi được ~1% Recall — luôn kèm khoảng tin cậy trước khi kết luận.
5. **Giao diện và script đánh giá dùng chung một pipeline.** Con số demo trên màn hình phải là chính con số trong báo cáo.
