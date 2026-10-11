# Nova — Uçtan Uca Sesli Yapay Zeka Ajanı

Kullanıcıyı mikrofondan dinleyen, konuşulan dili (Türkçe / İngilizce) algılayıp **aynı dilde**, **yalnızca bilgi
kaynaklarına (doküman, SQLite veritabanı, canlı API) dayanarak** cevap veren ve cevabı doğal sesle okuyan,
tamamen açık kaynak bileşenlerle çalışan bir sesli asistan.

> Kodun ve tasarım kararlarının dosya dosya ayrıntılı anlatımı: [PROJE_REHBERI.md](PROJE_REHBERI.md).

> Bilgi tabanındaki **"Nova Robotik"** şirketi, ürünleri, siparişleri ve servis merkezleri **kurgusaldır**; sistemin
> kaynaklara bağlı cevap üretip üretmediğini ölçebilmek için hazırlanmış örnek veridir.

```
Mikrofon ─► VAD (Silero) ─► STT (Whisper) ─► Ajan: RAG + araçlar + LLM (Ollama) ─► cümle cümle TTS (Piper) ─► Hoparlör
                 ▲                                                                        │
                 └──────────── kullanıcı araya girerse (barge-in) üretim anında iptal ◄───┘
```

---

## 1. Hızlı başlangıç (Docker)

İki kurulum var; ikisinde de ajan (STT, TTS, arama, araçlar) Docker'da çalışır, fark LLM'in nerede çalıştığıdır.
Docker Desktop (Windows/WSL2) AMD GPU'yu konteynere geçirmediği için GPU'yu yalnızca **bilgisayarda kurulu Ollama**
kullanabilir.

```bash
git clone https://github.com/AbdullahKarakoc/akinrobotics-nova-voice-agent.git
cd akinrobotics-nova-voice-agent
```

`.env` gerekmez: API anahtarı yoktur, tüm ayarların varsayılanı vardır ([.env.example](.env.example) yalnızca
değiştirmek için şablondur). Modeller ve SQLite veritabanı (`data/seed.sql`'den) ilk açılışta otomatik oluşturulur.

| Kurulum | LLM | Nerede | İlk ses (bu dizüstü, p50) |
|---|---|---|---|
| **A. GPU (önerilen)** | Qwen3-4B-Instruct-2507, 4-bit IQ4_XS (2.1 GB) | Bilgisayardaki Ollama, ekran kartı (%90 GPU) | **~6 sn** |
| B. Tamamen Docker | Qwen3-4B-Instruct-2507, 4-bit Q4_K_M (2.5 GB) | Konteynerdeki Ollama, CPU | ~16 sn |
| B + küçük model | Qwen3 1.7B, 4-bit Q4_K_M (1.3 GB) | Konteynerdeki Ollama, CPU | ~7.6 sn (daha çok uydurma, §4.3) |

### A. GPU ile (bilgisayarda Ollama kurulu)

Gereksinim: Docker (Compose v2) + [Ollama](https://ollama.com) (AMD kartlarda Vulkan için güncel sürücü). Model elle
indirilmez: ajan ilk açılışta bilgisayardaki Ollama'ya indirtir (~2.1 GB).

```bash
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build agent
```

Bu override ([docker-compose.host-ollama.yml](docker-compose.host-ollama.yml)) ajanı `host.docker.internal:11434`'e
bağlar ve GPU ayarlarını yapar: model `LLM_MODEL_GPU` (varsayılan 4B IQ4_XS), bağlam 3072 token, LLM'in CPU'da kalan
katmanları için 4 iş parçacığı (§2.2). Modelin GPU'da olduğu `ollama ps` ile görülür ("10%/90% CPU/GPU").
Hız öncelikliyse: `LLM_MODEL_GPU=qwen3:1.7b-q4_K_M` (ilk ses ~3.4 sn).

### B. Tamamen Docker (GPU'suz)

```bash
docker compose up --build
```

`ollama` (LLM sunucusu) + `ollama-pull` (modeli bir kez çeker) + `agent` konteynerleri açılır. İlk açılış uzun sürer:
Ollama imajı (~9.4 GB), LLM (~2.5 GB) ve ses/embedding modelleri (~0.8 GB) bir kez indirilir; 4B modelin CPU'da
ilk ısınması ~85 sn. İndirme sırasında terminal bir süre çıktı vermeyebilir: `docker compose logs -f`. Daha hızlı ama
daha az doğru: `LLM_MODEL=qwen3:1.7b-q4_K_M docker compose up`.

### Her iki kurulumda

Tarayıcıda **http://localhost:8000** adresini açın, mikrofon düğmesine bir kez basın ve konuşun (sonraki turlarda
tekrar basmaya gerek yok; "Dinlemeyi durdur" mikrofonu duraklatır). Hazır olduğunu `http://localhost:8000/health` →
`"llm_ready": true` ile görebilirsiniz. Ayarlar `.env` dosyasıyla değiştirilir (`cp .env.example .env`); tüm seçenekler
[app/config.py](app/config.py) içindedir.

> Mikrofon erişimi tarayıcıda yalnızca `localhost` veya HTTPS üzerinden açılır. Hoparlörle (kulaklıksız) kullanımda
> tarayıcının yankı giderme özelliği devrededir; yine de ajanın kendi sesini duyması sorun çıkarırsa `BARGE_IN=false`.

### Docker'sız (geliştirme)

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                # ayar şablonu (gizli bilgi içermez; isteğe bağlı)
# Ollama kurulu ve açık olmalı (https://ollama.com); LLM'i ajan ilk açılışta kendisi indirtir.
# Model .env'deki LLM_MODEL: GPU'da hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:IQ4_XS önerilir.
python -m scripts.download_models                   # Whisper, Piper sesleri, embedding modeli
python -m app                                       # http://localhost:8000
pytest                                              # 31 birim testi, model gerektirmez
```

### Diğer giriş noktaları

| Komut / uç nokta | Ne yapar |
|---|---|
| `GET /` | Web arayüzü (konuşma, kaynaklar, tur başına gecikme dökümü) |
| `WS /ws` | Sesli konuşma protokolü (16 kHz PCM giriş, PCM + JSON olay çıkışı) |
| `POST /api/chat` `{"text": "..."}` | Aynı ajanla metin tabanlı konuşma (test/hata ayıklama) |
| `GET /api/metrics` | Tüm turların gecikme istatistikleri (ortalama, p50, p90) |
| `python -m scripts.evaluate` | Doğruluk / halüsinasyon değerlendirmesi ([eval/questions.json](eval/questions.json)) |
| `python -m scripts.benchmark_voice` | Gerçek zamanlı ses akışıyla uçtan uca gecikme ölçümü |

---

## 2. Mimari

```mermaid
flowchart LR
    subgraph Tarayıcı
        MIC[Mikrofon<br/>AudioWorklet → 16 kHz PCM] --> WS
        WS -->|PCM parçaları| SPK[Oynatma kuyruğu]
    end
    WS((WebSocket)) --> VAD[Silero VAD<br/>konuşma sonu tespiti]
    VAD -->|ifade| STT[faster-whisper small<br/>dil tespiti TR/EN]
    STT --> AG{Ajan}
    AG -->|paralel| RT[Araç yönlendirici<br/>SQLite · Open-Meteo · saat]
    AG -->|paralel| RAG[Hibrit arama<br/>BM25 + çok dilli embedding]
    RT --> LLM[LLM · Ollama · 4-bit<br/>Qwen3-4B-Instruct · araç çağırma]
    RAG --> LLM
    LLM -->|token akışı| SEG[Cümle bölücü]
    SEG --> TTS[Piper TTS<br/>TR / EN ses]
    TTS -->|cümle cümle| WS
```

| Katman | Seçim | Neden |
|---|---|---|
| Konuşma algılama | **Silero VAD v6** (ONNX, 2 MB) | 32 ms'lik pencere başına < 1 ms CPU; enerji tabanlı VAD'lere göre gürültüye dayanıklı. 500 ms sessizlik = tur sonu; kısa duraklamalar turu bölmez. |
| Konuşma tanıma | **faster-whisper `small` int8** (CTranslate2) | TR+EN'i ve **dil tespitini tek modelde** yapar; int8 ile CPU'da hızlı. Varsayılan çözümleme yolu yerine özel tek geçişli kısa pencere çözümlemesi ile, gerçek seste aynı doğrulukta ~4 kat (kısa komutlarda daha fazla) hızlandırıldı (§4.4). |
| Dil modeli | **Ollama** + **Qwen3-4B-Instruct-2507, 4-bit** — GPU'da IQ4_XS (2.1 GB), CPU'da Q4_K_M (2.5 GB). Hızlı seçenek: Qwen3 1.7B Q4_K_M (düşünme kapalı) | Açık ağırlıklı (Apache-2.0), Türkçesi ve araç çağırması iyi. 4B Instruct: düşünme modu olmayan, talimat takibi için eğitilmiş sürüm; 1.7B'ye göre alan dışı ve STT'nin bozduğu sorularda belirgin biçimde daha az uydurma (ret 3/5 → 4/5, kaynaksız sayı 1 → 0, §4.3). IQ4_XS, 4 GB kartın kullanılabilir ~2.9 GB'ına %90 sığıyor; CPU'da k-quant (Q4_K_M) daha hızlı. 4-bit kuantizasyon fp16'ya göre ~3.5 kat az bellek; bu boyutta kalite kaybı küçük. Ollama: tek komutla model yönetimi (Hugging Face GGUF dahil), CPU/GPU (Vulkan), prompt (KV) önbelleği. Seçim 5 model ölçülerek yapıldı (§4.3). |
| Bilgi tabanı | **Hibrit arama**: BM25 (Türkçe için 5 harf kök kesme) + `paraphrase-multilingual-MiniLM-L12-v2` (fastembed / ONNX) | Türkçe soru İngilizce dokümanı bulabilir (diller arası); BM25 "T5", "E02" gibi kodları yakalar. PyTorch gerektirmez (imaj küçük). Küçük korpus için NumPy yeterli. |
| Araçlar | **SQLite** (sipariş, fiyat/stok, servis merkezi), **Open-Meteo** (hava, anahtarsız), saat/tarih | Ödevdeki "doküman + veritabanı + API" kaynak türlerinin üçü de var. Değişken veriler (fiyat, stok) dokümanda değil veritabanında. |
| Ses sentezi | **Piper** (VITS, ONNX) `tr_TR-dfki-medium`, `en_US-ryan-medium` | CPU'da gerçek zamanın ~10 katı hızlı (RTF ≈ 0.1), her iki dilde ses var, tamamen yerel. |
| Taşıma / arayüz | **FastAPI + WebSocket**, tarayıcıda AudioWorklet | Tam çift yönlü akış: kullanıcı ajan konuşurken araya girebilir. Kurulum gerektirmeyen istemci. |

### 2.1 Tasarım ilkeleri

* **Her bileşen değiştirilebilir.** STT, TTS ve LLM birer küçük arayüzün (`transcribe`, `synthesize`, `stream_chat`)
  arkasında; seçim yapılandırma ile yapılır ([app/stt](app/stt), [app/tts](app/tts), [app/llm](app/llm)).
  LLM için iki arka uç var: **Ollama** yerel API'si ve **OpenAI uyumlu** herhangi bir sunucu (vLLM, llama.cpp
  `llama-server`, LM Studio, OpenAI, Groq…): `LLM_BACKEND=openai LLM_BASE_URL=http://host:8080/v1 LLM_MODEL=...`.
  Tek bir kapalı servise bağımlılık yok; varsayılan kurulum tamamen yerel ve internetsiz çalışır (hava durumu aracı hariç).
* **Yeni bilgi kaynağı eklemek:** `data/docs/` klasörüne `.md`/`.txt` dosyası bırakmak yeterli; açılışta parçalanıp
  indekslenir (embedding'ler içerik özetine göre önbelleklenir).
* **Yeni araç eklemek:** bir fonksiyon + dekoratör:
  ```python
  @registry.tool("get_invoice", "Fatura tutarını ve son ödeme tarihini getirir.",
                 {"invoice_no": {"type": "string"}},
                 router=lambda text: ...)   # isteğe bağlı: niyet kesinse argümanları döndür
  def get_invoice(invoice_no: str) -> dict: ...
  ```
  Şema LLM'e otomatik verilir; hata/zaman aşımı yakalanıp modele "bilgi şu an yok" olarak iletilir
  ([app/tools/registry.py](app/tools/registry.py)).
* **Gözlemlenebilirlik:** her log satırı `oturum/tur` kimliği taşır; her tur için tek satırlık metrik logu
  (STT, arama, LLM ilk token, TTS, uçtan uca) ve kullanılan kaynaklar yazılır. `LOG_JSON=true` ile JSON log.

### 2.2 Gecikmeyi düşürmek için yapılanlar

Bu donanımda (§4.1) en pahalı adım LLM'in **prompt işleme (prefill)** süresidir: yalnızca CPU ile `qwen2.5:3b`
saniyede ~25-30, `qwen2.5:1.5b` ~50-65 token işleyebiliyor; GPU'da bile 4B model ilk tokene ~2 sn harcıyor. Bu yüzden
optimizasyonların çoğu "LLM'e daha az token göstermek" ve "beklemeyi paralelleştirmek" üzerine:

1. **Akışlı ve boru hatlı üretim:** LLM tokenları cümlelere bölünür; ilk cümle biter bitmez sentezlenip gönderilir,
   LLM bu sırada sonraki cümleyi üretir. Kullanıcı cevabın tamamını beklemez.
2. **Önbellek dostu prompt düzeni:** sistem promptu ve araç şemaları her turda birebir aynıdır → Ollama'nın KV önbelleği
   bunları yeniden işlemez. Tura özgü her şey (bağlam, araç verisi, dil talimatı) en sondaki kullanıcı mesajında.
3. **Az ama isabetli bağlam:** en iyi 2 parça + en iyi skorun 0.15 altındakileri at; alaka eşiğinin altında bağlam hiç yok.
4. **Deterministik araç yönlendirme:** "1003 numaralı sipariş", "T5'in fiyatı", "Ankara'da hava" gibi niyeti kesin
   sorular, LLM'e sorulmadan regex ile tanınır; araç **arama ile paralel** çalışır ve sonuç bağlama eklenir. Bu, ikinci bir
   LLM turunu (prefill + üretim) tamamen ortadan kaldırır ve küçük modellerin araç çağırmayı "unutması" sorununu çözer.
   Yönlendiricinin kaçırdığı durumlarda LLM'in kendi araç çağırması yedek olarak çalışır; o durumda araç beklenirken
   "Bir saniye, kontrol ediyorum." dolgu cümlesi söylenir.
5. **Whisper için tek geçiş + kısa pencere:** standart yol sesi 30 sn'ye tamamlar ve dil bilinmiyorsa encoder'ı iki kez
   çalıştırır. Encoder çıktısını hem dil tespiti hem çözümleme için kullanan ve yalnızca "ifade + 2 sn" uzunluğunda
   pencere kodlayan özel yol, 2-3 sn'lik bir cümlede STT süresini **~6.5-11 sn → ~1 sn**'ye indirdi; gerçek insan
   sesinde (FLEURS) Türkçe WER değişmedi (%21.3 → %20.9). Kısa pencerenin bilinen yan etkisi olan tekrar döngüleri,
   süreyle orantılı token sınırı + tekrar kırpma ile engellendi ([app/stt/faster_whisper_stt.py](app/stt/faster_whisper_stt.py)).
6. **Araya girme (barge-in):** kullanıcı konuşmaya başladığında oynatma anında durur ve LLM'e giden HTTP akışı kapatılır
   (sunucu üretimi keser). Kullanıcı cevap gelmeden önce yalnızca duraksayıp devam ettiyse, iki parça birleştirilip tek
   ifade olarak işlenir — cümlenin ilk yarısı kaybolmaz.
7. **Modeller bir kez yüklenir ve ısıtılır;** LLM açılışta sistem promptuyla çağrılarak önbelleğe alınır (model
   Ollama'da yoksa önce indirtilir). Ollama iş parçacığı sayısı (`LLM_NUM_THREAD=8`) bu işlemcide CPU'da prefill'i
   ~%35 hızlandırdı; GPU kurulumunda 4 (CPU'da kalan birkaç katman TTS ile yarışmasın, §4.2).
8. **GPU belleğine sığdırma:** 4 GB'lık kartta model + bağlam belleği sığmazsa katmanlar CPU'ya taşar ve üretim
   yavaşlar. Bağlam penceresi gerçek prompt uzunluğuna göre 4096 → 3072'ye indirildi (en uzun tek tur prompt ~1000
   token), GPU için daha küçük bir 4-bit kuantizasyon (IQ4_XS) seçildi (§4.3).

### 2.3 Doğruluk ve halüsinasyon kontrolü

| Katman | Önlem |
|---|---|
| Kaynak zorunluluğu | Sistem promptu: şirket/ürün/sipariş/fiyat bilgisi **yalnızca** verilen bağlam, araç sonucu veya canlı veriden; yoksa "bu bilgiye sahip değilim" + neye yardım edebileceği. Alan dışı sorular (genel kültür vb.) bilerek reddedilir. |
| Açık "bilgi yok" sinyali | Arama, kalibre edilmiş bir eşiğin (0.45) altındaysa modele bağlam yerine **"(no relevant information found)"** verilir. Ölçümlerde en etkili önlem bu oldu (§4.3: bu sinyalin olmadığı "full" modunda model genel müdür adı uydurdu). |
| Yapılandırılmış veri araçtan | Fiyat, stok, sipariş durumu, adres, hava durumu modelin hafızasından değil araçtan gelir; araçlar bulunamayan kayıt için açıkça `found: false` döner. |
| Yanlış biçimli araç çağrısı | Küçük modeller bazen araç çağrısını düz metin JSON olarak yazar; bu metin **seslendirilmez**, ayrıştırılıp gerçek araç çağrısına çevrilir. |
| Seslendirme öncesi sayı kontrolü | Her cümle **okunmadan önce** kontrol edilir: içindeki bir sayı soru, bağlam ve araç çıktılarının hiçbirinde yoksa cümle okunmaz, yerine "Bu konuda kaynaklarımda kesin bir bilgi yok." denir (gerçek kullanımda "pil ömrü fiyatı 499 lira" gibi bir uydurma böyle durduruldu). Ayrıca loglanır ve arayüzde ⚠ ile gösterilir. |
| Şeffaflık | Her cevabın altında kullanılan doküman parçaları ve araç çağrıları gösterilir ve loglanır. |
| Örnekleme | Düşük sıcaklık (0.2), kısa cevap (en fazla 200 token). |
| STT kaynaklı hatalar | Whisper'ın sessizlikte uydurduğu bilinen ifadeler ("Altyazı M.K.", "Thanks for watching"), konuşma-yok olasılığı yüksek çıktılar ve tekrar döngüleri atılır. Tarayıcıda 48 → 16 kHz dönüşümü 63 katsayılı FIR alçak geçiren filtreyle yapılır (eski tek kutuplu filtre 12 kHz'i yalnızca 5 dB bastırıp s/z/r seslerini konuşma bandına katlıyordu; yeni filtre 96 dB); tarayıcının gürültü bastırması kapalı. Yanlış duyulan ürün adları ("Zehbe Z2", "Vehbar ve iki" → "Rehber R2") regex ile düzeltilir, araç yönlendirici bitişik/bozuk yazımları da tanır ("havanası" → hava durumu), LLM'e de konuşmanın tanıma hatası içerebileceği söylenir. |
| Tekrar döngüsü | Küçük modellerin aynı cümleyi tekrar tekrar üretmesi: aynı cümle ikinci kez seslendirilmez. |
| Model boyutu | GPU'da 4B model: alan dışı ve bozuk tanınmış sorularda 1.7B'ye göre belirgin biçimde daha az uydurma (§4.3). |
| Ölçüm | [scripts/evaluate.py](scripts/evaluate.py): 26 soruluk sabit set (TR+EN) ile olgu doğruluğu, kaynakta olmayan soruda doğru ret (cevapta yasak ifade, ör. "Ankara", geçmemeli), araç seçimi, cevap dili ve kaynaksız sayı oranı ölçülür. |

### 2.4 Çok dillilik

Dil, Whisper'ın dil tespitiyle **ses üzerinden** belirlenir ve desteklenen dillerle (TR, EN) sınırlandırılır (kısa
Türkçe ifadeler aksi halde Azerice vb. sanılabiliyor). Cevap dili LLM'e hedef dilde verilen bir talimatla bildirilir
("Yalnızca Türkçe cevap ver."), TTS sesi de dile göre seçilir. Yazılı girişte basit bir işlev-kelimesi sayacı kullanılır.
Bilgi tabanı karışık dillidir (bazı dokümanlar İngilizce); çok dilli embedding ile Türkçe soru İngilizce dokümandan
cevaplanabilir. Yeni dil eklemek: `LANGUAGES`, bir Piper sesi ve iki kısa metin (dolgu/hata cümlesi).

---

## 3. Proje yapısı

```
app/
  audio/vad.py              Silero VAD akışı + konuşma sonu (endpointing) durum makinesi
  stt/                      STT arayüzü + faster-whisper (tek geçiş, kısa pencere)
  llm/                      LLM arayüzü + Ollama ve OpenAI-uyumlu akışlı istemciler
  tts/                      TTS arayüzü + Piper
  knowledge/                doküman parçalama, Türkçe BM25, embedding, hibrit arama
  tools/                    araç kayıt defteri, yönlendiriciler, SQLite ve web API araçları
  agent/                    promptlar, ajan döngüsü (arama → LLM → araçlar), metin yardımcıları
  pipeline/session.py       WebSocket oturumu: VAD → STT → ajan → TTS, barge-in, metrikler
  main.py                   FastAPI: arayüz, /ws, /api/chat, /api/metrics, /health
web/                        tarayıcı istemcisi (AudioWorklet yakalama, oynatma kuyruğu, arayüz)
data/docs/                  bilgi tabanı dokümanları (TR + EN), data/seed.sql örnek veritabanı
eval/questions.json         değerlendirme seti
scripts/                    model indirme, değerlendirme, sesli benchmark, rapor tablosu
reports/                    ölçüm çıktıları (JSON)
tests/                      birim testleri (sahte LLM/VAD ile, model gerektirmez)
Dockerfile                  ajan imajı (modeller imaja gömülmez, ilk açılışta /models birimine iner)
docker-compose.yml          B kurulumu: ollama + ollama-pull + agent (CPU)
docker-compose.host-ollama.yml  A kurulumu (override): bilgisayardaki Ollama + GPU ayarları
.env.example                ayar şablonu
```

---

## 4. Ölçüm sonuçları

**Özet (son hal, bu dizüstü):**

| | A. GPU (önerilen) | B. Tamamen Docker (CPU) |
|---|---|---|
| LLM | Qwen3-4B-Instruct-2507 IQ4_XS | Qwen3-4B-Instruct-2507 Q4_K_M |
| Kullanıcı sustu → ilk ses (p50) | **6.0 sn** | 16.0 sn |
| Doğruluk seti (26 soru, GPU'da ölçüldü) | 24/26 geçti, kaynaksız sayı 0, ret 4/5 | aynı model ailesi |
| STT (Whisper small, CPU) | ~1.3 sn, Türkçe WER %20.9 (gerçek ses) | aynı |

Ayrıntılar ve ara adımlar aşağıda; tüm ham çıktılar [reports/](reports/) klasöründe.

### 4.1 Donanım

| | |
|---|---|
| İşlemci | AMD Ryzen 7 3750H (4 çekirdek / 8 iş parçacığı, 2.3-4.0 GHz, 2019 dizüstü) |
| Bellek | 16 GB DDR4 (yaklaşık 14 GB kullanılabilir) |
| GPU | Radeon RX 560X 4 GB (+ dahili Vega 10). Eski sürücüyle (2020) Ollama kartı görmedi → ilk ölçümler **yalnızca CPU**. Sürücü güncellendikten sonra (31.0, 2026) Ollama kartı **Vulkan** ile kullanıyor; 1.7B modelin 29/29 katmanı, 4B Instruct modelin ~%90'ı GPU'da (kullanılabilir VRAM ~2.9 GB). |
| Yazılım | Windows 11, Python 3.11, Ollama 0.40.2 (yerel), Docker Desktop 29.8 |

§4.4, §4.3'ün ilk tablosu ve §4.2'deki CPU ölçümü yalnızca CPU ile; "GPU" etiketli ölçümlerde yalnızca LLM GPU'da,
STT/TTS yine CPU'da.
Docker Desktop (WSL2) AMD GPU'yu konteynere geçirmediği için GPU kullanımı **ana makinedeki Ollama** ile
(`docker-compose.host-ollama.yml`) mümkün.

### 4.2 Uçtan uca sesli gecikme

`scripts/benchmark_voice.py`: 8 soru (5 TR, 3 EN) × 2 tekrar. Sorular Piper ile sese çevrilip WebSocket üzerinden
**gerçek zamanlı** (32 ms paketlerle, tarayıcının yaptığı gibi) gönderilir; süre, istemcinin son konuşma örneğini
gönderdiği andan ilk ajan sesini aldığı ana kadar ölçülür. Yalnızca CPU (`qwen3:1.7b`, RAG,
`LLM_NUM_THREAD=8`), 16 tur ([reports/voice_qwen3-1-7b_rag.json](reports/voice_qwen3-1-7b_rag.json)):

| Adım | p50 | p90 | Not |
|---|---|---|---|
| Konuşma sonu tespiti (VAD) | 0.65 sn | 0.68 sn | 500 ms sessizlik eşiği + çerçeve payı |
| STT (faster-whisper small) | 0.88 sn | 1.18 sn | **RTF 0.42** (2-3 sn'lik ifadeler) |
| Bilgi tabanında arama | 11 ms | 14 ms | embedding + BM25 |
| LLM ilk token | 3.0 sn | 4.2 sn | üretim hızı ~9 token/sn |
| TTS ilk cümle | 1.0 sn | 2.2 sn | **RTF 0.21** (LLM ile aynı anda çalışırken; tek başına RTF ≈ 0.10) |
| **Kullanıcı sustu → ilk ajan sesi** | **7.6 sn** | **10.7 sn** | ortalama 7.3 sn |

Yorum: CPU'da gecikmenin ~%40'ı LLM, ~%15'i STT, ~%10'u VAD bekleme süresi. Ses gelmeye başladıktan sonra
TTS gerçek zamandan hızlı olduğu için konuşma kesintisiz akar.

**Tüm yapılandırmalar karşılaştırmalı** — ajan Docker'da; GPU sütunlarında Ollama ana makinede (RX 560X, Vulkan),
8'er tur. **"GPU, 4B Instruct" sütunu önerilen A kurulumu, son sütun B kurulumudur.** GPU + 1.7B:
[reports/voice_qwen3-1-7b-q4_rag_gpu.json](reports/voice_qwen3-1-7b-q4_rag_gpu.json).

| Adım (p50) | CPU, 1.7B | GPU, 1.7B | GPU, 4B Instruct IQ4_XS | Tam Docker (CPU), 4B Instruct Q4_K_M |
|---|---|---|---|---|
| Konuşma sonu tespiti (VAD) | 0.65 sn | 0.70 sn | 0.75 sn | 0.68 sn |
| STT (CPU, konteynerde) | 0.88 sn | 1.24 sn (RTF 0.55) | 1.28 sn | 1.19 sn |
| LLM ilk token | 3.0 sn | **0.72 sn** | 1.97 sn | 10.4 sn (p90 15.5) |
| LLM üretim hızı | ~9 token/sn | **~26 token/sn** | ~8.6 token/sn | ~4.2 token/sn |
| TTS ilk cümle | 1.0 sn | **0.31 sn** (RTF 0.10) | 0.56 sn (RTF 0.15) | 0.57 sn (RTF 0.17) |
| **Kullanıcı sustu → ilk ajan sesi** | 7.6 sn | **3.4 sn** | 6.0 sn | 16.0 sn (p90 24.2) |

Son sütun `docker compose up`'ın varsayılanı (B) ([reports/voice_qwen3-4b-instruct-q4km_rag_cpu_docker.json](reports/voice_qwen3-4b-instruct-q4km_rag_cpu_docker.json)):
tüm modeller konteynerde, GPU yok. Cevaplar doğru (8/8 kaynaklı, ret ve araç doğru), ama 4B modelin CPU'da prefill'i
(~1000 token'lık ilk prompt dahil) ilk sesi ~16 sn'ye çıkarıyor; açılıştaki ısınma 85 sn sürüyor. Bu bilinçli bir
doğruluk tercihi. Aynı makinede GPU yolu (`docker-compose.host-ollama.yml`) 6.0 sn, 1.7B ile tam Docker ~7.6 sn.

4 GB'lık giriş seviyesi bir GPU bile 1.7B modelde LLM süresini ~4 kat kısalttı. GPU'da varsayılan yine de 4B: ilk ses
~2.6 sn gecikiyor, karşılığında alan dışı ve bozuk tanınmış sorularda belirgin biçimde daha az uydurma var (§4.3).
Hız öncelikliyse `LLM_MODEL_GPU=qwen3:1.7b-q4_K_M`. 4B'nin GPU'ya sığmayan birkaç katmanı CPU'da çalıştığı için
GPU kurulumunda `LLM_NUM_THREAD=4`: 8 iş parçacığında bu katmanlar, aynı anda çalışan TTS ile yarışıyordu (ilk ses 7.4 sn,
TTS ilk cümle 1.1 sn). 4B ölçümü: [reports/voice_qwen3-4b-instruct-iq4_rag_gpu.json](reports/voice_qwen3-4b-instruct-iq4_rag_gpu.json).
8 turluk küçük örneklemler; mertebe olarak değerlendirilmeli.

> Not: Piper'ın Türkçe sesi robotik olduğu için bu testte Türkçe tanıma hataları gerçek kullanımdan kötüdür
> ("Rehber R2" → "Vehber ve iki"); gecikme ölçümü için geçerli, doğruluk için §4.4'teki gerçek ses ölçümü esas alınmalı.

### 4.3 Doğruluk / halüsinasyon ve LLM seçimi

**Yalnızca CPU, ilk set:** [eval/questions.json](eval/questions.json) setinin ilk hali, 24 soru (13 TR + 11 EN; sonradan
26'ya çıktı, aşağıdaki GPU tablosu). 18'inde beklenen bilgi kaynaklarda var,
4'ünde **yok** (genel kültür, kaynakta olmayan ürün ağırlığı, kaynakta olmayan genel müdür adı) — bunlarda doğru davranış
"bilmiyorum" demektir; 7'si araç gerektirir. Her soru yeni bir konuşmada, `/api/chat` üzerinden sorulur
(`python -m scripts.evaluate`, tüm yapılandırmalar için `scripts/eval_matrix.ps1`). Çıktılar: [reports/](reports/).

| LLM | Bilgi modu | Geçen | Olgu doğruluğu | Doğru ret | Araç seçimi | Cevap dili | Kaynaksız sayı içeren cevap | LLM ilk token p50 | Toplam cevap p50 |
|---|---|---|---|---|---|---|---|---|---|
| **qwen3:1.7b** | **rag** | **23/24** | **18/18** | **3/4** | 7/7 | 24/24 | **0** | 3.0 sn | 6.9 sn |
| qwen3:1.7b | full | 22/24 | 18/18 | 2/4 | 7/7 | 24/24 | 0 | 3.7 sn | 8.2 sn |
| qwen2.5:3b | rag | 22/24 | 18/18 | 3/4 | 7/7 | 23/24 | 1 | 4.6 sn | 10.6 sn |
| qwen2.5:3b | full | 20/24 | 17/18 | 1/4 | 7/7 | 24/24 | 1 | 3.3 sn | 8.9 sn |
| qwen2.5:1.5b | rag | 20/24 | 17/18 | 1/4 | 7/7 | 24/24 | 0 | 2.4 sn | 5.3 sn |
| qwen2.5:1.5b | full | 20/24 | 17/18 | 1/4 | 7/7 | 24/24 | 0 | 1.9 sn | 5.4 sn |

Tablodaki tüm LLM'ler 4-bit (Q4_K_M) çalıştı: Ollama'nın `qwen3:1.7b`, `qwen2.5:3b`, `qwen2.5:1.5b` etiketleri
Q4_K_M dosyasını gösterir (`qwen3:1.7b` ile `qwen3:1.7b-q4_K_M` aynı model katmanı: `3d0b790534fe`).
"Toplam cevap" metin cevabının tamamının üretilme süresidir; sesli kullanımda kullanıcı ilk cümleyi çok daha önce duyar (§4.2).

**Çıkarımlar ve seçimler**

* **Bu tablodaki seçim: `qwen3:1.7b` + `rag`.** CPU'da en doğru ve 3B modelden daha hızlı yapılandırma; kaynaksız tek
  bir sayı bile üretmedi. Sonradan varsayılan 4B oldu (aşağıda); 1.7B hızlı seçenek olarak duruyor.
* **"full" (tüm bilgi tabanı önbellekli sistem promptunda)** prefill'i ortadan kaldırdığı için 3B modelde ilk token süresini
  ~%30 kısalttı, ama her modelde **ret oranı düştü**: modeller kaynakta olmayan bilgiyi uydurdu (ilk denemede
  "genel müdür Ahmet Yılmaz", "Temizlik C3 68 kg" — başka ürünün ağırlığı). Her turda arama sonucunun kimliklerini
  ("ilgili: K3, K7" / "ilgili bilgi yok") eklemek bunu kısmen düzeltti ama RAG'ın seviyesine çıkaramadı. Korpus küçük
  olduğu için seçenek olarak duruyor (`KNOWLEDGE_MODE=full`); doğruluğun öncelikli olduğu varsayılan RAG.
* **Model boyutu halüsinasyon için belirleyici:** `qwen2.5:1.5b`, "ilgili bilgi bulunamadı" sinyaline rağmen genel müdür
  için isim uydurdu ("Kadir Tanrıverdi"). Daha hızlı olsa da varsayılan yapılmadı.
* **Hiçbir küçük model "Türkiye'nin başkenti" sorusunu reddetmedi** (cevap doğru ama kapsam dışı; 1.5B bir denemede
  "İstanbul" dedi). Bu, prompt ile çözülemeyen bir model sınırı; kritik kullanımda alan dışı soru sınıflandırıcısı eklenebilir.

**Son seçim — GPU ile daha büyük model (RX 560X 4 GB, Vulkan, RAG, 26 soru).** Sette sonradan eklenenler: canlı kullanımda görülen
iki hata — "Akınsoft şirketi nerede?" (1.7B "Ankara'da" diye uydurdu) ve STT'nin bozduğu "Zehbe Z2'nin piyomru ne
kadar?" (1.7B: "1999 yılında üretildi"). Ret sorularına `forbid` eklendi: cevapta "Ankara" geçen ret başarısız sayılır.

| LLM (hepsi 4-bit) | VRAM'e sığan | Geçen | Olgu | Doğru ret | Kaynaksız sayı | İlk token p50 | Üretim hızı |
|---|---|---|---|---|---|---|---|
| qwen3:1.7b Q4_K_M | %100 | 23/26 | 18/19 | 3/5 | 1 | 1.2 sn | ~26 token/sn |
| qwen3:4b Q4_K_M (2.5 GB) | %75 | 24/26¹ | 19/19 | 3/5¹ | 0 | 2.5 sn | ~8 token/sn |
| **Qwen3-4B-Instruct-2507 IQ4_XS (2.1 GB)** | %90-95 | **24/26** | 18/19² | **4/5** | **0** | 1.9 sn | ~9-10 token/sn |

¹ Dünya Kupası sorusunu doğru reddetti ama o sırada betik "cannot provide" kalıbını tanımıyordu; sonradan eklendi.
² Kalan hata "Zehbe Z2": bu test yazılı yoldan gider; sesli yolda STT düzeltme tablosu onu zaten "Rehber R2'nin pil
ömrü"ne çevirir. Kalan ret hatası her modelde aynı: "Türkiye'nin başkenti" sorusuna "Ankara" diyorlar.

Kartta kullanılabilir alan ~2.9 GB (Ollama günlüğü). `qwen3:4b` sığmadığı için CPU'ya taşıyor ve yavaşlıyor. Instruct-2507,
aynı 4B boyutunda düşünme modu olmayan, talimat takibi için eğitilmiş sürüm; daha küçük 4-bit kuantizasyonu (IQ4_XS) ve
3072 bağlamla neredeyse tamamı GPU'ya sığıyor. GPU kurulumunun varsayılanı bu (`docker-compose.host-ollama.yml`).
Bedeli gecikme: üretim 1.7B'nin ~1/3'ü hızında (§4.2). Doğruluk gecikmeden önemli görüldüğü için tam Docker
kurulumunda (CPU) da varsayılan 4B: aynı modelin CPU'da daha hızlı çalışan Q4_K_M sürümü (`docker-compose.yml`).

**Geliştirme sürecinde aynı set ile ölçülen etkiler** (qwen2.5:3b, RAG):

| Aşama | Geçen | Araç seçimi | LLM ilk token p50 |
|---|---|---|---|
| İlk sürüm: araçları yalnızca LLM çağırıyor, top-3 bağlam, 4 iş parçacığı | 18/24 | 5/7 | 10.2 sn |
| + deterministik araç yönlendirme, top-2 + bağıl kesme, 8 iş parçacığı | 21/24 | 7/7 | 5.2 sn |
| + prompt düzeltmeleri, metin olarak yazılan araç çağrısının onarımı, eşik kalibrasyonu | 22/24 | 7/7 | 4.6 sn |

Alaka eşiği (0.45) veriyle seçildi: değerlendirme setindeki bilgi sorularının en iyi arama skoru ≥ 0.62, set dışından
eklenen 9 alakasız soru dahil alan dışı soruların skoru ≤ 0.43. Eşik bu setle kalibre edildiği için yeni sorularda
biraz iyimser olabilir.

### 4.4 STT: gerçek insan sesiyle doğruluk ve hız

`scripts/benchmark_stt.py`: Google FLEURS test setinden (gerçek konuşmacılar, CC-BY 4.0) dil başına 15 kayıt,
ortalama 10.5 sn. Kelime hata oranı (WER) ve RTF, `small` int8, 4 iş parçacığı:

| Çözümleme yolu | TR WER | TR süre / ifade | TR RTF | EN WER | EN RTF |
|---|---|---|---|---|---|
| Standart faster-whisper (otomatik dil tespiti, 30 sn pencere) | 21.3% | 12.8 sn | 1.07 | 7.7% | 1.15 |
| Tek geçiş, 30 sn pencere + domain promptu | 35.9% | 6.8 sn | 0.57 | 9.8% | 0.74 |
| Tek geçiş, kısa pencere + domain promptu | 35.5% | 2.9 sn | 0.24 | 9.8% | 0.23 |
| **Tek geçiş, kısa pencere, promptsuz (varsayılan)** | **20.9%** | **3.0 sn** | **0.28** | – | – |

* Tek geçiş + kısa pencere, standart yolla **aynı Türkçe doğrulukta ~4 kat hızlı**; 2-3 sn'lik sesli komutlarda kazanç
  daha büyük (~7-11 sn → ~1 sn). Dil tespiti tüm kayıtlarda doğru (30/30).
* Ölçüm önemli bir hatayı ortaya çıkardı: ürün adlarını Whisper'a "önceki metin" olarak vermek (hotwords) Türkçe WER'i
  **%21 → %36**'ya çıkarıyordu. Bu yüzden kaldırıldı; ürün adları artık çözümlemeden sonra küçük bir düzeltme tablosuyla
  normalleştiriliyor ([app/stt/domain_terms.py](app/stt/domain_terms.py): "Rehber ve iki" → "Rehber R2").
* Son satır yalnızca TR için ölçüldü (EN satırları domain promptlu sürümle alındı; prompt EN'de etkisi küçüktü: 7.7 → 9.8).

**Daha büyük model denendi mi? (`medium`, aynı 15 TR kaydı, ortalama 11.9 sn):**

| Model / pencere | TR WER | 12 sn'lik ifade | 2.5 sn'lik komut |
|---|---|---|---|
| **small, kısa pencere (varsayılan)** | 20.9% | 3.0 sn | **1.1 sn** |
| medium, kısa pencere | 23.7% | 11.8 sn | 4.0 sn |
| medium, 30 sn pencere | **12.2%** | 26.1 sn | — (çok yavaş) |

`medium` Türkçe hatayı neredeyse yarıya indiriyor, ama yalnızca tam 30 sn pencereyle — kısa pencere `small`'ın
doğruluğunu korurken `medium`'unkini bozuyor. Bu CPU'da 30 sn pencereli `medium` canlı konuşma için fazla yavaş, bu
yüzden varsayılan `small`. `medium` kullanılacaksa `STT_MODEL=medium STT_DYNAMIC_WINDOW=false` birlikte ayarlanmalı
(GPU'lu bir makinede mantıklı). Bu makinede doğruluğu artırmanın yolu Whisper'ı da GPU'ya almak: CTranslate2'nin AMD
desteği yok, whisper.cpp'nin Vulkan arka ucu ise RX 560X'te çalışabilir (yeni bir STT arka ucu; denenmedi — whisper.cpp
Windows için hazır Vulkan derlemesi yayınlamıyor, derlemek için Visual Studio + Vulkan SDK gerekiyor).

**Diğer STT ayarları (small, aynı TR kayıtları):** beam search (`STT_BEAM_SIZE=5`) WER'i 20.9% → 19.5%'e indiriyor
ama 3 sn'lik komuta ~0.3 sn ekliyor; kapalı. İş parçacığı sayısı ortama bağlı: Windows'ta doğrudan çalışırken 8 iş
parçacığı daha hızlı (1.25 → 1.10 sn), Docker (WSL2) içinde ise daha yavaş (uçtan uca ölçümde STT 1.27 → 1.91 sn);
varsayılan 4.

### 4.5 Ölçümleri tekrar üretmek

```bash
python -m scripts.benchmark_stt --n 15            # STT: gerçek insan sesiyle WER / RTF (FLEURS, ~15 dk)
python -m app &                                    # sunucu
python -m scripts.benchmark_voice --repeat 2      # uçtan uca sesli gecikme
python -m scripts.evaluate                        # doğruluk / halüsinasyon
powershell -File scripts/eval_matrix.ps1          # tüm LLM × bilgi modu kombinasyonları (Windows)
powershell -File scripts/eval_matrix.ps1 -Configs "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:IQ4_XS|rag" -Port 8001 -Suffix _gpu
                                                  # tek model, Docker'daki ajan 8000'de açıkken
python -m scripts.report_table                    # README tablosu
```

---

## 5. Sınırlamalar ve sonraki adımlar

* **Donanım darboğazı.** Yalnızca CPU ile ilk ses ~7.6 sn, bunun büyük kısmı LLM. LLM giriş seviyesi bir GPU'ya (RX 560X)
  alınınca ~3.4 sn'ye indi; mimari hiç değişmedi, yalnızca Ollama'nın çalıştığı yer değişti. GPU'da daha doğru 4B
  modeli seçmek ilk sesi ~6 sn'ye çıkarıyor (4 GB kart modelin tamamını alamıyor). 8 GB'lık bir kart 4B'yi tamamen
  GPU'ya alır ve bu farkı kapatır; `LLM_BACKEND=openai` ile vLLM de bir seçenek.
* **GPU yalnızca ana makinedeki Ollama ile:** Docker Desktop (WSL2) AMD GPU'yu konteynere geçirmiyor; GPU için
  `docker-compose.host-ollama.yml` kullanılmalı. Tam konteyner kurulumu (`docker compose up`) CPU'da çalışır ve
  4B model orada belirgin biçimde yavaştır (ilk ses ~16 sn, §4.2); hız gerekiyorsa `LLM_MODEL=qwen3:1.7b-q4_K_M`.
* **STT akışlı değil:** ifade bittikten sonra çözümlenir (~1 sn). Akışlı/artımlı tanıma (ör. konuşurken kısmi çözümleme)
  bu süreyi neredeyse sıfırlar; bir sonraki en büyük kazanç bu.
* **Sentetik test sesi:** uçtan uca benchmark Piper sesleriyle yapıldı; Piper'ın tek Türkçe sesi oldukça robotik olduğundan
  bu testlerdeki Türkçe tanıma hataları gerçek kullanımdan kötüdür. STT doğruluğu bu yüzden ayrıca gerçek insan sesiyle
  (FLEURS) ölçüldü (§4.4).
* **Küçük modellerin sınırları:** alan dışı genel kültür sorusunu cevaplayabiliyor (4B dahil hiçbir model
  "Türkiye'nin başkenti" sorusunu reddetmedi), araç çıktısını nadiren yanlış okuyabiliyor
  (bir denemede araçtan gelen "Cuma"yı "Cumartesi" dedi). Sayısal uydurmalar sayı kontrolüyle yakalanıyor; sözel olanlar
  yakalanmıyor. Daha güçlü bir model ya da cevap sonrası LLM tabanlı doğrulama (ek gecikme pahasına) bunu azaltır.
* **Yankı:** hoparlörle kullanımda araya girme, tarayıcının yankı gidermesine güvenir; kulaklık önerilir ya da `BARGE_IN=false`.
* **Kimlik doğrulama yok:** sipariş sorgusu numarayla yapılır; gerçek sistemde müşteri doğrulaması gerekir.
