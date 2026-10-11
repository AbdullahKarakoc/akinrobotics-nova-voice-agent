# Proje Rehberi — Nova Sesli Yapay Zeka Ajanı, Baştan Sona

Bu belge projeyi **öğrenmek** için yazıldı. README "ne yapıyor, nasıl çalıştırılır, ne ölçüldü" sorularını cevaplar;
bu rehber ise "**nasıl** ve **neden** böyle yapıldı" sorularını, her kod dosyasını tek tek açarak anlatır.

Önerilen okuma sırası: §1 (büyük resim) → §2 (kavramlar) → §5 (bir turun yolculuğu) → §6 (dosya dosya kod) →
§7-9 (tasarım kararları) → §11 (mülakat soruları).

---

## İçindekiler

1. [Büyük resim](#1-büyük-resim)
2. [Temel kavramlar sözlüğü](#2-temel-kavramlar-sözlüğü)
3. [Kullanılan teknolojiler ve seçilme nedenleri](#3-kullanılan-teknolojiler-ve-seçilme-nedenleri)
4. [Klasör yapısı](#4-klasör-yapısı)
5. [Bir konuşma turunun yolculuğu (kod üzerinden)](#5-bir-konuşma-turunun-yolculuğu-kod-üzerinden)
6. [Dosya dosya kod açıklaması](#6-dosya-dosya-kod-açıklaması)
7. [Gecikme mühendisliği](#7-gecikme-mühendisliği)
8. [Doğruluk ve halüsinasyon kontrolü](#8-doğruluk-ve-halüsinasyon-kontrolü)
9. [Ölçüm metodolojisi ve sonuçlar](#9-ölçüm-metodolojisi-ve-sonuçlar)
10. [Geliştirme sırasında karşılaşılan sorunlar ve çözümleri](#10-geliştirme-sırasında-karşılaşılan-sorunlar-ve-çözümleri)
11. [Mülakatta sorulabilecek sorular ve cevapları](#11-mülakatta-sorulabilecek-sorular-ve-cevapları)
12. [Projeyi genişletmek: alıştırmalar](#12-projeyi-genişletmek-alıştırmalar)

---

## 1. Büyük resim

### 1.1 Ödev ne istiyordu?

AKINROBOTICS mülakat ödevi: kullanıcıyla **sesli** konuşan, **sesten sese** çalışan bir ajan. Beklentiler:

| Beklenti | Projede karşılığı |
|---|---|
| Çok dillilik (TR + EN, aynı dilde cevap) | Whisper'ın dil tespiti + dile göre LLM talimatı + dile göre TTS sesi |
| Doğruluk (cevap bilgi kaynağından) | RAG (doküman), SQLite (veritabanı), Open-Meteo (API), halüsinasyon önlemleri |
| Açık platform | Tüm modeller açık kaynak/açık ağırlıklı ve yerel; LLM sunucusu değiştirilebilir |
| Düşük gecikme | Akışlı üretim, cümle cümle TTS, hızlandırılmış Whisper, önbellek dostu prompt, araç yönlendirme |
| Özelleştirilebilirlik | Her bileşen bir arayüzün arkasında; araç = dekoratörlü fonksiyon; doküman = klasöre dosya |
| Sürdürülebilirlik | Modüler paketler, bağlamlı loglama, Docker, birim testleri |

### 1.2 Sistem tek cümlede

> Tarayıcı mikrofon sesini WebSocket ile sunucuya akıtır; sunucu konuşmanın bittiğini **VAD** ile anlar, **Whisper** ile
> yazıya çevirir, **ajan** ilgili bilgiyi dokümanlardan ve araçlardan toplayıp **LLM**'e cevap ürettirir, LLM'in
> ürettiği metin **cümle cümle Piper** ile sese çevrilip tarayıcıya geri akıtılır.

```
 TARAYICI                                SUNUCU (FastAPI, Python)
 ┌──────────────┐  16 kHz PCM   ┌───────────────────────────────────────────────────────────────┐
 │ Mikrofon     │ ────────────► │ VAD (Silero) ──► STT (Whisper) ──► Ajan                        │
 │ AudioWorklet │               │                                    ├─ Araç yönlendirici ─┐     │
 │              │               │                                    ├─ Hibrit arama (RAG) ┤     │
 │ Hoparlör     │ ◄──────────── │ TTS (Piper) ◄── Cümle bölücü ◄──── LLM (Ollama, akışlı) ◄┘     │
 │ çalma kuyruğu│  PCM + JSON   └───────────────────────────────────────────────────────────────┘
 └──────────────┘
```

### 1.3 Son durum: iki kurulum

| | A. GPU (önerilen) | B. Tamamen Docker |
|---|---|---|
| Komut | `docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build agent` | `docker compose up --build` |
| Ajan (VAD, STT, TTS, arama, araçlar) | Docker, CPU | Docker, CPU |
| LLM sunucusu | Bilgisayardaki Ollama (Vulkan ile RX 560X) | Konteynerdeki Ollama (CPU) |
| LLM | Qwen3-4B-Instruct-2507, 4-bit **IQ4_XS** (%90 GPU'da) | Aynı model, 4-bit **Q4_K_M** |
| Kullanıcı sustu → ilk ses | **~6 sn** | ~16 sn |

Neden iki kurulum? Docker Desktop (Windows/WSL2) AMD ekran kartını konteynere geçiremiyor; GPU'yu yalnızca Windows'ta
çalışan Ollama kullanabiliyor. Ajan her iki durumda da aynı kod, yalnızca `LLM_BASE_URL` ve model ayarı değişiyor —
bileşenlerin arayüz arkasında olmasının pratik faydası. Hız öncelikli olursa her iki kurulumda da `qwen3:1.7b-q4_K_M`
(GPU'da ~3.4 sn, CPU'da ~7.6 sn) seçilebilir; bedeli daha çok uydurma (§9.3).

### 1.4 Zaman çizelgesi: kullanıcı sustuktan sonra ne olur?

Önerilen kurulumda (A) tipik bir tur (ölçülmüş değerler, medyan):

```
t=0.00 sn  kullanıcının son hecesi
t=0.75     VAD: "500 ms sessizlik oldu, tur bitti"            (endpoint)
t=2.0      STT bitti: "Rehber R2'nin pil ömrü ne kadar?" [tr]  (~1.3 sn, Whisper CPU'da)
t=2.0      arama bitti (~15 ms) — araç gerekmiyor
t=~4.0     LLM ilk token                                      (~2 sn: prompt işleme, GPU)
t=~5.4     ilk cümle (ya da ilk virgüle kadar) tamamlandı → TTS (~9 token/sn üretim)
t=6.0      kullanıcı ilk sesi duyar                            ← "gecikme" dediğimiz sayı
           ... LLM sonraki cümleleri üretirken önceki cümleler çalınıyor ...
```

Yalnızca CPU'da ve 1.7B modelle ilk ölçümde bu çizelge 7.6 sn'ydi (LLM ilk token ~3 sn); en hızlı yapılandırma GPU +
1.7B ile 3.4 sn. Bu çizelgeyi aklınızda tutun: §7'deki her optimizasyon bu çizgideki bir aralığı kısaltmaya yöneliktir.

---

## 2. Temel kavramlar sözlüğü

### Ses

| Kavram | Açıklama |
|---|---|
| **PCM** | Sıkıştırılmamış ham ses. Projede 16-bit tamsayı (int16, -32768..32767), tek kanal. |
| **Örnekleme hızı** | Saniyedeki örnek sayısı. Mikrofon genelde 48 kHz; Whisper ve VAD **16 kHz** ister; Piper **22.05 kHz** üretir. |
| **Yeniden örnekleme (resampling)** | Bir örnekleme hızından diğerine çevirme. Tarayıcıda 48k → 16k yapılır (pcm-worklet.js). |
| **Aliasing** | Örnekleme hızını düşürürken yüksek frekansların yanlış frekanslara "katlanması". Önlemek için önce alçak geçiren filtre uygulanır. |
| **Çerçeve (frame)** | Sabit uzunlukta ses parçası. Silero VAD 512 örnek = **32 ms** çerçeve kullanır. |
| **Mel spektrogram** | Sesin zaman-frekans gösterimi; Whisper'ın girdisi. 10 ms'de bir çerçeve → 30 sn = 3000 çerçeve. |

### Konuşma işleme

| Kavram | Açıklama |
|---|---|
| **VAD** (Voice Activity Detection) | Bir ses çerçevesinde konuşma olup olmadığını söyleyen model. |
| **Endpointing / tur sonu tespiti** | VAD olasılıklarından "kullanıcı konuşmayı bitirdi" kararı çıkarmak (sessizlik süresi eşiği). |
| **Barge-in** | Kullanıcının, ajan konuşurken sözünü kesebilmesi. Doğal konuşmanın temel özelliği. |
| **STT / ASR** | Konuşmayı yazıya çevirme (Speech-to-Text / Automatic Speech Recognition). |
| **TTS** | Yazıyı sese çevirme (Text-to-Speech). |
| **WER** (Word Error Rate) | STT doğruluk ölçüsü: (eklenen + silinen + değişen kelime) / referans kelime sayısı. Düşük = iyi. |
| **RTF** (Real-Time Factor) | İşlem süresi / ses süresi. RTF 0.4 → 10 sn'lik ses 4 sn'de işlenir. **< 1 ise gerçek zamandan hızlı.** |
| **Dil tespiti (LID)** | Konuşulan dili belirleme. Whisper bunu kodlayıcı çıktısından yapar. |

### Büyük dil modeli (LLM)

| Kavram | Açıklama |
|---|---|
| **Token** | Modelin işlediği metin parçası (kelime parçası). Türkçe, eklemeli yapısı yüzünden İngilizceden fazla token'a bölünür. |
| **Prefill (prompt işleme)** | Modelin tüm girdi metnini işlediği ilk aşama. CPU'da pahalı: bu makinede qwen2.5:3b ~25-32 token/sn. |
| **Decode (üretim)** | Token token cevap üretme. Bu makinede qwen3:1.7b CPU'da ~9, GPU'da ~26 token/sn; 4B model GPU'da ~9 token/sn (konuşma hızı ~4-5 token/sn olduğu için ses yine kesintisiz akar). |
| **TTFT** (Time To First Token) | İstek gönderildikten ilk token gelene kadar geçen süre ≈ prefill süresi. Sesli ajan için en kritik LLM metriği. |
| **KV cache / prompt önbelleği** | Modelin işlediği token'ların ara durumları. Sonraki istek **aynı önekle** başlıyorsa bu kısım yeniden işlenmez. |
| **Sıcaklık (temperature)** | Rastgelelik. Düşük (0.2) = daha tutarlı, daha az "yaratıcı" (halüsinasyon için iyi). |
| **Araç çağırma (tool/function calling)** | LLM'in metin yerine "şu fonksiyonu şu argümanlarla çağır" çıktısı vermesi. |
| **Halüsinasyon** | Modelin kaynakta olmayan bilgiyi kendinden emin şekilde uydurması. |
| **Quantization (int8, Q4)** | Ağırlıkları daha az bitle saklamak: daha az bellek, daha hızlı CPU çıkarımı, küçük doğruluk kaybı. Projede LLM **4-bit Q4_K_M** (`qwen3:1.7b-q4_K_M`, 1.27 GB; fp16 olsa ~3.4 GB), Whisper **int8**. "K_M": llama.cpp'nin blok bazlı "k-quant" yöntemi; önemli katmanlarda biraz daha fazla bit kullanan orta (M) varyant. CPU'da token üretimi bellek bant genişliğiyle sınırlı olduğu için daha küçük ağırlık = daha hızlı üretim. GPU modeli **IQ4_XS** (`Qwen3-4B-Instruct-2507`, 2.1 GB): "importance matrix" ile hangi ağırlıkların hassas olduğu örnek metinle ölçülüp bitler ona göre dağıtılır; aynı kalitede Q4_K_M'den biraz küçük — 4 GB karta sığdırmak için seçildi. |
| **VRAM'e sığmak / katman taşması** | Ollama modelin katmanlarını GPU belleğine yükler; model + bağlam belleği (KV cache, `num_ctx` ile büyür) sığmazsa kalan katmanlar CPU'da çalışır ve hız düşer (`ollama ps` → "10%/90% CPU/GPU"). |

### Bilgi getirme

| Kavram | Açıklama |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Soruya ilgili doküman parçalarını bulup LLM'e bağlam olarak vermek. |
| **CAG** (Cache-Augmented Generation) | Tüm bilgi tabanını prompta koyup KV önbelleğinde tutmak (projede `KNOWLEDGE_MODE=full`). |
| **Chunk (parça)** | Dokümanın aranabilir küçük bölümü. Projede her markdown başlığı altı bir parça. |
| **Embedding** | Metnin anlamını temsil eden sayı vektörü (burada 384 boyutlu). Benzer anlam → yakın vektörler. |
| **Kosinüs benzerliği** | İki vektör arasındaki açının kosinüsü; normalize vektörlerde iç çarpım. 1 = aynı yön. |
| **BM25** | Klasik kelime eşleşmesine dayalı arama skoru (TF-IDF'in gelişmiş hâli). |
| **Hibrit arama** | Anlamsal (embedding) + kelimesel (BM25) skorları birleştirmek. |
| **Kök kesme (stemming)** | Kelimeyi köküne indirmek. Projede basit "ilk 5 harf" yöntemi (Türkçe için bilinen etkili bir hile). |

### Yazılım

| Kavram | Açıklama |
|---|---|
| **WebSocket** | Tarayıcı ile sunucu arasında sürekli açık, iki yönlü bağlantı. Ses akışı için HTTP'den uygun. |
| **AudioWorklet** | Tarayıcıda ses işleme kodunu ayrı, gerçek zamanlı bir iş parçacığında çalıştıran API. |
| **asyncio** | Python'un eşzamansız (async/await) çalışma modeli; çok sayıda G/Ç işini tek iş parçacığında yürütür. |
| **ThreadPoolExecutor** | CPU yoğun işleri (STT, TTS) asyncio döngüsünü bloklamadan ayrı iş parçacığında çalıştırmak için. |
| **ONNX / ONNX Runtime** | Modelleri çerçeveden bağımsız formatta saklayıp hızlı çalıştıran altyapı (VAD, Piper, embedding). |
| **CTranslate2** | Transformer modelleri için optimize C++ çıkarım motoru; faster-whisper bunun üstüne kurulu. |

---

## 3. Kullanılan teknolojiler ve seçilme nedenleri

| Katman | Teknoloji | Neden bu? | Alternatifler (neden değil) |
|---|---|---|---|
| Web sunucusu | **FastAPI + Uvicorn** | async, WebSocket desteği, az kod | Flask (async/WebSocket zayıf) |
| Ayarlar | **pydantic-settings** | Tip kontrollü ayarlar, ortam değişkeni/.env okuma | elle `os.environ` (hataya açık) |
| VAD | **Silero VAD v6** (ONNX) | 2 MB, çerçeve başına < 1 ms, gürültüye dayanıklı; faster-whisper ile birlikte geliyor | WebRTC VAD (enerji tabanlı, gürültüde zayıf) |
| STT | **faster-whisper `small` int8** | TR+EN + dil tespiti tek modelde; CTranslate2 ile CPU'da hızlı | Vosk (TR doğruluğu düşük), büyük Whisper (CPU'da yavaş) |
| LLM sunucusu | **Ollama** | Tek komutla model indirme, CPU/GPU, KV önbelleği, araç çağırma | llama.cpp sunucusu, vLLM (OpenAI arka ucuyla destekleniyor) |
| LLM (Docker, CPU) | **Qwen3-4B-Instruct-2507, 4-bit Q4_K_M** | Kullanıcı kararı: hız yerine doğruluk; GPU'da ölçülen 4B kalitesi, CPU'da hızlı çalışan k-quant ile | 1.7B (hızlı seçenek, aşağıda) |
| LLM (hızlı seçenek) | **Qwen3 1.7B, 4-bit Q4_K_M** (düşünme kapalı) | Küçük modeller içinde Türkçe ve araç çağırma iyi; ölçümle seçildi (§9); 4-bit ile 1.27 GB | qwen2.5:3b (daha yavaş), qwen2.5:1.5b (daha çok halüsinasyon), 8-bit q8_0 (2.07 GB, CPU'da daha yavaş) |
| LLM (GPU) | **Qwen3-4B-Instruct-2507, 4-bit IQ4_XS** | Alan dışı / bozuk tanınmış sorularda daha az uydurma; 4 GB karta ~%90 sığıyor (§9) | qwen3:4b Q4_K_M (karta sığmadı, daha yavaş), 1.7B (hızlı ama daha çok uydurma) |
| Embedding | **paraphrase-multilingual-MiniLM-L12-v2** (fastembed/ONNX) | Çok dilli (TR soru ↔ EN doküman), 220 MB, PyTorch gerektirmez | multilingual-e5-large (2.2 GB, yavaş) |
| Vektör deposu | **NumPy matrisi** | 31 parça için veritabanı gereksiz; arama ~10 ms | Qdrant/pgvector (büyük korpusta) |
| Veritabanı | **SQLite** | Kurulumsuz, dosya tabanlı, örnek "operasyonel veri" için ideal | PostgreSQL (gereksiz ağırlık) |
| Canlı API | **Open-Meteo** | Ücretsiz, API anahtarı gerektirmiyor | OpenWeatherMap (anahtar gerekir) |
| TTS | **Piper** (VITS, ONNX) | CPU'da RTF ≈ 0.1, Türkçe ses var, yerel | XTTS (CPU'da çok yavaş), Kokoro (Türkçe yok) |
| İstemci | **Saf HTML/JS + AudioWorklet** | Kurulum/derleme gerektirmez | React (bu boyutta gereksiz) |
| Paketleme | **Docker + Compose** | Ödevin teslim şartı; tek komutla kurulum | — |
| Test | **pytest + pytest-asyncio** | async kodu test edebilmek | — |

---

## 4. Klasör yapısı

```
akınRobotics/
├── app/                      ← uygulamanın kendisi (Python paketi)
│   ├── __main__.py           ← `python -m app` giriş noktası (uvicorn'u başlatır)
│   ├── main.py               ← FastAPI uygulaması: HTTP/WebSocket uç noktaları
│   ├── config.py             ← tüm ayarlar (tek yer)
│   ├── logging_setup.py      ← oturum/tur bağlamlı loglama
│   ├── services.py           ← modelleri bir kez yükleyip bir arada tutan kap
│   ├── audio/vad.py          ← Silero VAD + tur sonu durum makinesi
│   ├── stt/                  ← konuşma tanıma (arayüz + Whisper + terim düzeltme)
│   ├── llm/                  ← LLM istemcileri (arayüz + Ollama + OpenAI uyumlu)
│   ├── tts/                  ← ses sentezi (arayüz + Piper)
│   ├── knowledge/            ← doküman parçalama, BM25, embedding, hibrit arama
│   ├── tools/                ← araç kayıt defteri, yönlendiriciler, SQLite/API araçları, MCP istemcisi
│   ├── agent/                ← promptlar, ajan döngüsü, metin yardımcıları
│   └── pipeline/             ← WebSocket oturumu ve metrikler
├── mcp_servers/nova_service/ ← bağımsız MCP sunucusu (garanti, teşhis, servis talebi)
├── web/                      ← tarayıcı istemcisi (index.html, app.js, pcm-worklet.js, style.css)
├── data/
│   ├── docs/                 ← bilgi tabanı (6 markdown, TR + EN karışık)
│   └── seed.sql              ← örnek veritabanı (ürün, sipariş, servis merkezi)
├── eval/questions.json       ← 26 soruluk değerlendirme seti
├── scripts/                  ← model indirme, değerlendirme, benchmark'lar
├── tests/                    ← 40 birim testi
├── reports/                  ← ölçüm sonuçları (JSON)
├── models/                   ← indirilen modeller (git'e girmez)
├── Dockerfile, docker-compose.yml, docker-compose.host-ollama.yml
├── requirements.txt, requirements-dev.txt, pytest.ini, .env.example
└── README.md, PROJE_REHBERI.md
```

**Mimari desen:** her dış bileşen (STT, TTS, LLM) için `base.py` içinde küçük bir **arayüz** (Python `Protocol`),
yanında somut uygulama(lar) ve `__init__.py` içinde ayara göre doğru uygulamayı seçen bir **fabrika fonksiyonu**
(`create_stt`, `create_tts`, `create_llm`) var. Uygulamanın geri kalanı yalnızca arayüzü bilir. Bu sayede Piper yerine
başka bir TTS koymak, sadece yeni bir sınıf yazıp fabrikaya bir satır eklemektir.

---

## 5. Bir konuşma turunun yolculuğu (kod üzerinden)

Kullanıcı "1003 numaralı siparişim ne durumda?" diyor. Kodu takip edelim:

1. **Mikrofon → 16 kHz PCM** — `web/pcm-worklet.js`, `PcmCapture.process()`: her 128 örneklik tarayıcı bloğunu
   alçak geçiren filtreden geçirir, 16 kHz'e indirir, 1024 örnek (64 ms) dolunca ana iş parçacığına yollar.
2. **WebSocket'e gönderim** — `web/app.js`, `micNode.port.onmessage`: `ws.send(data.pcm)` (ikili mesaj).
3. **Sunucuda karşılama** — `app/main.py`, `voice()` → `VoiceSession.run()` (`app/pipeline/session.py`):
   ikili mesaj gelince `on_audio(pcm16)`.
4. **VAD** — `Endpointer.process()` (`app/audio/vad.py`): sesi 512'lik çerçevelere böler, her çerçeve için
   `VADStream.__call__` konuşma olasılığı döner. 200 ms üst üste konuşma → `SPEECH_START` olayı; konuşma sırasında
   500 ms sessizlik → `SPEECH_END` olayı (içinde ifadenin tüm sesi).
5. **Tur başlatma** — `VoiceSession._start_turn()` bir `asyncio.Task` olarak `_run_turn()`'ü başlatır.
6. **STT** — `_run_turn()`, `FasterWhisperSTT.transcribe()`'ı ayrı iş parçacığında çalıştırır →
   `Transcript(text="1003 numaralı siparişim ne durumda?", language="tr")`. Ardından `fix_domain_terms()`.
   İstemciye `{"type": "transcript", ...}` gider (ekranda kullanıcı balonu belirir).
7. **Ajan** — `Agent.respond()` (`app/agent/agent.py`):
   * `self.tools.route(question)` → `route_order` "sipariş" kelimesini ve "1003"ü görür →
     `ToolCall(get_order_status, {"order_number": "1003"})`. Araç **hemen** çalıştırılır (`ToolRegistry.execute`).
   * Paralelde `_retrieve()` → hibrit arama (bu soru için alakalı doküman çıkmayabilir).
   * `build_user_message()` → `CONTEXT: ...`, `LIVE DATA: get_order_status(...): {"status_tr": "kargoya verildi, yolda", ...}`,
     `USER: ...`, `(Yalnızca Türkçe cevap ver.)`.
   * `OllamaLLM.stream_chat()` → Ollama'ya akışlı istek; token'lar `TextDelta` olarak gelir.
8. **Cümle bölme** — `_run_turn()` her token'ı `SentenceChunker.push()`'a verir; tamamlanan cümle bir kuyruğa
   (`asyncio.Queue`) konur.
9. **TTS** — Paralel çalışan `_speak()` görevi kuyruktan cümleyi alır, `clean_for_speech()` ile temizler,
   `PiperTTS.synthesize()` ile sese çevirir, 8 baytlık başlıkla (`turn_id`, `sample_rate`) birlikte ikili mesaj olarak yollar.
10. **Çalma** — `app.js`, `onAudio()`: başlığı okur, `AudioBuffer` oluşturur, `playhead` zamanına göre kuyruğa planlar.
11. **Kapanış** — LLM bitince `answer` olayı (tam metin + kaynaklar + kaynaksız sayı uyarısı) ve `metrics` olayı
    (gecikme dökümü) gider; sunucu tek satırlık "turn metrics" logu yazar; konuşma geçmişine eklenir.

**Araya girme senaryosu:** 9. adımda kullanıcı konuşmaya başlarsa → `on_audio()` içinde `SPEECH_START` →
`interrupt("barge_in")` → tur görevi iptal edilir (`task.cancel()`), bu da LLM'e giden HTTP akışını kapatır (Ollama
üretimi durdurur); istemciye `interrupt` olayı gider, `stopPlayback()` çalan sesi anında keser.

---

## 6. Dosya dosya kod açıklaması

### 6.1 `app/config.py` — tüm ayarlar

* `Settings(BaseSettings)`: her alan bir ayar. Pydantic, aynı isimli **büyük harfli ortam değişkenini** veya `.env`
  dosyasındaki satırı otomatik okur ve tipine çevirir (`LLM_NUM_THREAD=8` → `llm_num_thread: int = 8`).
* Gruplar: sunucu, VAD/tur yönetimi, STT, LLM, bilgi tabanı, TTS, araçlar.
* Önemli varsayılanlar ve nedenleri yorumlarda: `rag_min_score = 0.45` (kalibrasyonla seçildi), `stt_hotwords = ""`
  (ölçümde WER'i bozduğu için kapalı), `llm_think = False` (Qwen3'ün düşünme aşamasını atlar → gecikme düşer),
  `llm_num_ctx = 3072` (en uzun prompt ~1000 token + geçmiş; küçük bağlam = GPU'ya daha çok katman),
  `stt_cpu_threads = 4` (Docker/WSL2'de 8'den hızlı).
* `@lru_cache get_settings()`: ayarlar süreç boyunca bir kez oluşturulur (singleton).
* `voice_for(language)`: dile göre Piper sesi.

### 6.2 `app/logging_setup.py` — bağlamlı loglama

* **Problem:** aynı anda birden çok oturum ve tur varken log satırının hangisine ait olduğunu bilmek.
* **Çözüm:** `contextvars.ContextVar` ile `session_id_var` ve `turn_id_var`. asyncio her görev için bağlamı kopyaladığı
  için, bir turun içinde set edilen değer o görevin tüm loglarına otomatik eklenir.
* `_ContextFilter` her kayda `session_id`/`turn_id` ekler; `_TextFormatter` / `_JsonFormatter` iki format sunar
  (`LOG_JSON=true` ile makinece okunabilir JSON — log toplama sistemleri için).
* Yapısal veri `extra={"data": {...}}` ile verilir ve satırın sonuna JSON olarak eklenir. Örnek:
  `08:41:08 INFO [d48c3e/1] app.pipeline.session: turn metrics {"stt_ms": 2045, ...}`

### 6.3 `app/services.py` — model kabı

* `build_services(settings)`: VAD, STT, TTS, arama ve araçları **bir kez** yükler, her birinin yükleme süresini loglar.
* `Services` dataclass'ı bunları bir arada tutar; tüm oturumlar aynı modelleri paylaşır.
* `stt_executor` ve `tts_executor`: **tek iş parçacıklı** havuzlar. Neden tek? Whisper ve Piper zaten içeride çok
  çekirdek kullanıyor; aynı anda iki STT çalıştırmak çekirdekleri boğar ve ikisini de yavaşlatır.

### 6.4 `app/main.py` ve `app/__main__.py` — HTTP/WebSocket uygulaması

* `lifespan`: FastAPI açılırken `build_services`'i **ayrı iş parçacığında** çalıştırır (modeller yüklenirken olay
  döngüsü donmasın), ardından `_warmup()` görevini başlatır.
* `_warmup()`: önce `ensure_model()` ile model Ollama'da var mı bakar (`/api/show`), yoksa indirtir (`/api/pull`) —
  ana makinedeki Ollama'yı kullanan GPU kurulumunda elle `ollama pull` gerekmez. Sonra LLM'e sistem promptuyla küçük
  bir istek atar → model belleğe yüklenir ve **sistem promptu KV önbelleğine girer**. LLM sunucusu henüz hazır değilse (Docker'da model iniyor olabilir) 10 sn arayla 60 kez dener.
  Başarınca `state["llm_ready"] = True`.
* Uç noktalar: `GET /` (arayüz), `GET /health` (sağlık + `llm_ready`), `GET /api/metrics` (tüm turların istatistiği),
  `WS /ws` (sesli oturum), `POST /api/chat` (metinle aynı ajan — değerlendirme betiği bunu kullanır).
* `/api/chat` konuşma geçmişini `session_id` ile `OrderedDict` içinde tutar; en fazla 100 oturum (LRU mantığı:
  `move_to_end` + en eskiyi at).
* `__main__.py`: `python -m app` → uvicorn'u ayarlardaki host/port ile başlatır.

### 6.5 `app/audio/vad.py` — VAD ve tur sonu tespiti

**`SileroVAD` / `VADStream`**
* Model: faster-whisper paketinin içinden gelen `silero_vad_v6.onnx`. Girdi: `input` (1×576 = 64 örnek bağlam + 512
  yeni örnek), `h` ve `c` (1×1×128 — LSTM'in gizli durumları). Çıktı: konuşma olasılığı + yeni durumlar.
* `SileroVAD` ONNX oturumunu bir kez açar (paylaşılır); her ses akışı için `stream()` ayrı bir `VADStream` verir —
  çünkü `h`, `c` ve 64 örneklik bağlam **akışa özgü** durumdur.

**`Endpointer`** — olasılıklardan olaylara giden durum makinesi:
```
BOŞTA ──(art arda ≥ min_speech_ms konuşma)──► KONUŞUYOR ──(≥ min_silence_ms sessizlik)──► BOŞTA
           SPEECH_START olayı                                   SPEECH_END olayı (+ ses)
```
* **Histerezis:** konuşma başlangıcı için eşik 0.5, sessizlik sayımı için 0.35 (`neg_threshold`). İki eşik, sınırdaki
  olasılıkların durumu sürekli değiştirmesini (titreşim) önler.
* **min_speech_ms (200):** öksürük, tık sesi gibi kısa patlamalar tur başlatmasın.
* **min_silence_ms (500):** cümle içindeki doğal duraklamalar turu bölmesin. Büyük değer → daha geç cevap;
  küçük değer → kullanıcının sözü kesilir. 500 ms yaygın bir denge noktası.
* **Ön tampon (pre-roll, `speech_pad_ms`=300):** VAD konuşmayı birkaç çerçeve geç fark eder; kelimenin ilk hecesi
  kaybolmasın diye son 300 ms hep saklanır ve ifadenin başına eklenir.
* **Sondaki sessizliği kırpma:** ifade sesi, son "sesli" çerçeveden 300 ms sonrasına kadar kesilir — Whisper'a
  gereksiz sessizlik gitmez.
* `speech_end_time`: son sesli çerçevenin geldiği an — gecikme ölçümünün sıfır noktası.
* `min_speech_ms` dışarıdan değiştirilebilir: ajan konuşurken daha yüksek tutulur (barge-in için daha katı başlangıç).
* `process()` gelen sesi `_pending` tamponunda biriktirir ve 512'lik çerçevelere böler — ağdan gelen paket boyutundan
  bağımsız çalışır.

### 6.6 `app/stt/` — konuşma tanıma

**`base.py`:** `Transcript` (metin, dil, dil olasılığı, ses süresi, işlem süresi, `rtf` özelliği) ve `SpeechToText`
protokolü: `transcribe(audio, languages) -> Transcript`.

**`faster_whisper_stt.py`** — projenin en "mühendislik" yoğun dosyası.

*Standart yolun sorunu:* Whisper 30 saniyelik pencerelerle eğitilmiştir; faster-whisper her sesi 30 sn'ye **dolgu ile
tamamlar** ve dil bilinmiyorsa kodlayıcıyı (encoder) **iki kez** çalıştırır (önce dil tespiti, sonra çözümleme).
2-3 sn'lik bir komut için bu makinede `small` model ~7-11 sn sürdü.

*Bizim yolumuz (`transcribe`):*
1. `feature_extractor(audio)` → mel spektrogram.
2. **Kısa pencere:** 3000 çerçeve yerine `ifade + 200 çerçeve (2 sn)`. Bu 2 sn'lik sessizlik payı, çözümleyicinin
   konuşmanın bittiğini görüp metni sonlandırmasını sağlar (payı küçük tutunca model kendini tekrar etmeye başladı).
3. `model.encode(...)` **bir kez** çalışır; aynı `encoder_output` hem `detect_language` hem `generate` için kullanılır.
4. **Kısıtlı dil tespiti:** olasılıklar arasından yalnızca `["tr", "en"]` içinden en olası seçilir (kısa Türkçe ifadeler
   Azerice/Özbekçe sanılabiliyordu).
5. `generate(...)`: açgözlü çözümleme (`beam_size=1`), **token bütçesi** `10 + 12 × saniye` (döngüye girerse sonsuza
   gitmesin).
6. Güvenlik kontrolleri:
   * `trim_repetitions()`: regex `(\S.{2,}?)(?:[\s,.!?]+\1){2,}` → "Merhaba Merhaba Merhaba" → "Merhaba".
   * `no_speech_prob > 0.6` ve düşük skor → metni at (aslında konuşma yoktu).
   * `_HALLUCINATIONS`: Whisper'ın sessizlikte uydurduğu bilinen ifadeler ("Altyazı M.K.", "Thanks for watching",
     "Abone ol") — altyazılı videolarla eğitildiği için öğrendiği kalıplar.
* Sonuç: gerçek insan sesinde **aynı Türkçe WER (%21) ile ~4 kat hız** (10 sn'lik ifadelerde; kısa komutlarda ~1 sn).
* `hotwords` (domain kelimelerini "önceki metin" olarak verme) desteği var ama **varsayılan kapalı** — §10'da neden.

**`domain_terms.py`:** STT sonrası ürün adı düzeltmesi. Örn. regex
`\b[rzvh]e(?:h?b|h?dd?)a?e?r?[\s-]*(?:r|ar|er|re|ve|z)?[\s-]*(?:2|iki|two)\b` → "Rehber R2": gerçek kullanımda
ve testlerde görülen "Rehber ve iki", "Zehbe Z2", "Vehbar ve iki", "Heddar ve iki" hepsini yakalar; "rehberlik",
"zebra 2", "hedef 2" gibi normal ifadelere dokunmaz (kelime sınırı + rakam/sayı kelimesi zorunluluğu). "Piyomru" →
"pil ömrü" gibi kelime düzeltmeleri de burada. Yeni ürün = yeni satır; her satırın testi `test_speech_text.py`'de.

**`__init__.py`:** `create_stt(settings)` fabrikası.

### 6.7 `app/llm/` — LLM istemcileri

**`base.py`** — arka uçtan bağımsız olay modeli:
* `TextDelta(text)`: akan metin parçası.
* `ToolCalls(calls)`: modelin istediği araç çağrıları (`ToolCall(id, name, arguments)`).
* `Done(stats)`: bitiş + `LLMStats` (prompt/üretim token sayısı, prefill ve üretim süreleri, `tokens_per_s`).
* `ChatLLM` protokolü: `stream_chat(messages, tools) -> AsyncIterator[LLMEvent]`.
* İç mesaj formatı **OpenAI sohbet formatıdır** (role/content/tool_calls/tool_call_id); her arka uç kendi formatına çevirir.

**`ollama_llm.py`** — Ollama'nın yerel `/api/chat` API'si:
* Neden OpenAI uyumlu uç nokta yerine yerel API? Çünkü `options.num_ctx`, `num_thread`, `keep_alive`, `think` gibi
  ayarları ve **prefill/üretim süresi** istatistiklerini yalnızca yerel API veriyor.
* Akış **NDJSON**: her satır bir JSON; `message.content` → `TextDelta`, `message.tool_calls` → toplanır, `done: true`
  satırındaki `prompt_eval_duration`/`eval_duration` (nanosaniye) → `LLMStats`.
* `_to_wire()`: Ollama araç argümanlarını sözlük ister; araç sonucu mesajında `tool_name` bekler.
* `ensure_model()`: `/api/show` 404 dönerse `/api/pull` ile modeli indirtir (açılışta `_warmup` çağırır).
  `hf.co/<kullanıcı>/<repo>:<kuantizasyon>` biçimindeki adlarla Hugging Face'teki GGUF dosyaları da doğrudan çekilebilir.
* `keep_alive="30m"`: model 30 dk boşta kalsa da bellekte tutulur (her istekte yeniden yükleme olmasın).

**`openai_llm.py`** — OpenAI uyumlu her sunucu (vLLM, llama.cpp, LM Studio, OpenAI, Groq…):
* Akış **SSE** (`data: {...}` satırları, sonda `data: [DONE]`).
* Araç çağrıları **parça parça** gelir (`index` ile); `partial` sözlüğünde birleştirilir, sonda JSON'a çevrilir.
* `_to_wire()`: OpenAI argümanları **JSON metni** olarak ister.
* `stream_options.include_usage`: token sayılarını almak için.

**`__init__.py`:** `create_llm` — `LLM_BACKEND=ollama|openai`.

### 6.8 `app/tts/` — ses sentezi

**`base.py`:** `SpeechAudio` (PCM baytları, örnekleme hızı, işlem süresi, `duration_s`, `rtf`) ve `TextToSpeech` protokolü.

**`piper_tts.py`:**
* Her dil için bir `PiperVoice` yüklenir (`tr_TR-dfki-medium`, `en_US-ryan-medium`; `.onnx` + `.onnx.json`).
* `synthesize()`: Piper ses parçaları üretir (`AudioChunk.audio_int16_array`), birleştirilip tek PCM'e çevrilir.
* `SynthesisConfig(length_scale)`: konuşma hızı (<1 daha hızlı).
* `_ascii_safe_espeak_dir()`: **Windows'a özgü bir hata çözümü.** Piper, telaffuz için içinde espeak-ng (C kütüphanesi)
  kullanıyor; bu kütüphane ASCII olmayan yolları (proje yolundaki "akın"daki **ı**) açamıyor. Çözüm: Windows'un 8.3 kısa
  yolunu (`GetShortPathNameW`) kullanmak, olmazsa veriyi geçici klasöre kopyalamak. Docker'da (Linux, yol `/app`) gerekmez.

### 6.9 `app/knowledge/` — bilgi tabanı ve arama

**`text.py`:**
* `turkish_lower()`: Python'un `str.lower()`'ı "I"yı "i" yapar — Türkçede yanlış ("I"→"ı", "İ"→"i" olmalı).
* `tokenize()`: küçük harf → kelimelere ayır → durak kelimeleri (ve, ile, the, is…) at → **ilk 5 harf** (kök kesme).
  "robotların", "robotu", "robotlar" → hepsi "robot". Eklemeli Türkçe için bilinen basit ve etkili yöntem.

**`documents.py`:**
* `Chunk(id, source, title, text)`; `id` = "dosya.md#n" (atıf için), `title` = başlık yolu ("Ürün Kataloğu > Rehber R2 >
  Teknik Özellikler").
* `chunk_markdown()`: markdown'ı başlıklara göre böler; her bölüm bir parça, `max_chars`'ı (600) aşarsa paragraf/cümle
  sınırından bölünür. **Başlık yolu parçaya eklenir** — "Pil ömrü 8 saattir" cümlesi tek başına hangi ürün olduğunu
  söylemez; başlık söyler.
* `load_chunks()`: `data/docs` içindeki tüm `.md`/`.txt` dosyaları.

**`retriever.py`:**
* `BM25`: klasik formül (k1=1.5, b=0.75). Her terim için IDF (nadir kelime → yüksek ağırlık) × TF doygunluğu × uzunluk
  normalizasyonu.
* `Embedder`: fastembed `TextEmbedding`; vektörler **L2 normalize** edilir (böylece iç çarpım = kosinüs benzerliği).
  E5 ailesi modeller için "query:"/"passage:" önekleri.
* `Retriever`:
  * Açılışta tüm parçaları gömer; sonuç `models/index_cache/index-<sha1>.npy` dosyasına önbelleklenir (sha1 = model adı +
    tüm parça metinleri; doküman değişirse otomatik yeniden hesaplanır).
  * `search(query, top_k, min_score)`:
    ```
    dense   = matris @ sorgu_vektörü              (kosinüs, -1..1)
    lexical = bm25 / (bm25 + 4)                   (0..1'e doyurulmuş)
    skor    = dense + 0.15 × lexical              (kelime eşleşmesi en fazla +0.15 bonus)
    ```
  * Neden BM25'i sorgu başına "en yüksek = 1" diye normalize etmiyoruz? Çünkü o zaman alakasız bir sorguda bile en iyi
    sonuç 1 alır; doyurma fonksiyonu mutlak bir ölçek verir ve **eşik anlamlı kalır**.
  * `min_score` altındakiler hiç dönmez → ajan "bilgi bulunamadı" sinyali verebilir.

### 6.10 `app/tools/` — araçlar

**Genel bakış: 10 araç, iki çağrılma yolu** (ayrıntılı tablo: README §2.6)

1. **Yönlendirici (router):** LLM'den önce, soruda niyet kelimesi ve gereken bilgi birlikte varsa regex ile çalışır.
   Doküman aramasıyla paralel yürür, sonuç LLM'e "LIVE DATA" olarak gider. İkinci LLM turu olmaz.
2. **LLM'in araç çağrısı (yedek):** yönlendirici kaçırırsa model şemaya bakıp kendisi çağırır. LLM 8 aracın şemasını
   görür; iki Wikipedia aracını görmez.

| Araç | Nereden | Hangi soruda (yönlendirici) | Örnek |
|---|---|---|---|
| `get_order_status` | SQLite | "sipariş" + 3-7 haneli no | "1003 numaralı siparişim ne durumda?" |
| `get_product_info` | SQLite | fiyat/stok/satın alma kelimesi + ürün | "Taşıyıcı T5'in fiyatı ne kadar?" |
| `find_service_center` | SQLite | "servis/tamir" + şehir | "İzmir'de servis merkeziniz var mı?" |
| `get_weather` | Open-Meteo | hava kelimesi + şehir | "Ankara'da hava nasıl?" |
| `get_current_datetime` | sistem saati | "saat kaç / bugün günlerden ne" | "Saat kaç?" |
| `check_warranty` | MCP Nova Servis | 5 haneli seri no + "garanti/bakım" | "21507 seri numaralı robotumun garantisi devam ediyor mu?" |
| `get_robot_diagnostics` | MCP Nova Servis | 5 haneli seri no + "çalışmıyor/hata/şarj/ne durumda" | "24012 numaralı robotum çalışmıyor." |
| `get_service_ticket_status` | MCP Nova Servis | 4 haneli no + "talep/kayıt/ticket" | "5003 numaralı servis talebim ne durumda?" |
| `wikipedia_tr_get_summary` | MCP wikipedia-mcp | Türkçe "nedir/ne demek" + sözlük terimi | "LiDAR nedir?" |
| `wikipedia_en_get_summary` | MCP wikipedia-mcp | İngilizce "what is" + sözlük terimi, ya da TR maddesi olmayan terim | "What is a humanoid robot?", "SLAM nedir?" |

Araç **çağrılmayan** sorular da önemli: "Rehber R2'nin pil ömrü ne kadar?" ve "Garanti süresi ne kadar?" bilgi
sorusu, dokümandan cevaplanır. "Siparişim ne durumda?" (numara yok) sorusunda yönlendirici çalışmaz, LLM numarayı
sorar. "Rehber R2'nin sensörleri nedir?" bizim ürünümüz olduğu için Wikipedia'ya değil dokümana gider.
Uzunluklarına göre ayrılan numaralar (seri no 5, sipariş ve talep 4 hane) ve niyet kelimeleri yönlendiricileri
birbirinden ayırır.

**`registry.py`:**
* `Tool(name, description, parameters, required, handler, router)` ve `schema()` → LLM'e verilen JSON şema.
* `@registry.tool(...)` dekoratörü: bir fonksiyonu araç olarak kaydeder.
* `execute(call)`:
  * Modelin uydurduğu fazladan argümanları atar (`inspect.signature` ile yalnızca kabul edilenler).
  * Senkron fonksiyonu `asyncio.to_thread` ile, async olanı doğrudan çalıştırır; `wait_for` ile 8 sn zaman aşımı.
  * **Hata asla yukarı fırlatılmaz:** `{"error": "... Tell the user the information is unavailable"}` modele döner —
    model "bilgi şu an yok" der, uydurmaz.
* `route(text)`: her aracın `router` fonksiyonunu çalıştırır; tetiklenenler `ToolCall` olarak döner (§7'de neden).

**`routing.py`:**
* `fold()`: büyük/küçük harf ve aksan duyarsız anahtar ("İstanbul", "ISTANBUL", "istanbul" → "istanbul").
* `mentions(text, pattern)`: katlanmış metinde regex arar (desenler aksansız yazılır: "sipari" hem "sipariş"i hem
  "siparis"i yakalar).
* `extract_city()`: önce 81 il listesi; sonra Türkçe bulunma hâli eki ("Konya'**da**", "Londra'**da**"); sonra İngilizce
  "in Berlin".

**`database.py`:**
* `ensure_database()`: `nova.db` yoksa `seed.sql`'den oluşturur.
* Bağlantı **salt okunur** açılır (`mode=ro`) — LLM'in tetiklediği araçlar veriyi değiştiremez.
* `match_product()`: kod ("T5"), ad ("Taşıyıcı T5"), takma ad ("cleaning" → C3), son çare `difflib` bulanık eşleşme.
* Araçlar ve yönlendiricileri:
  * `get_order_status` ← "sipariş/order" + 3-7 haneli sayı. "1.003", "NR-1003", "1 003" hepsi 1003'e çevrilir
    (konuşmadan gelen sayı biçimleri tutarsız).
  * `get_product_info` ← "fiyat/ücret/stok/price/cost/..." + ürün adı. ("How much does R2 **weigh**" tetiklemez.)
  * `find_service_center` ← "servis/service center" + şehir.
* Bulunamayan kayıt için açık `{"found": false, "note": ...}` — modele "yok" bilgisini net verir.
* `STATUS_TEXT`: durum kodları her iki dilde ("shipped" → "kargoya verildi, yolda").

**`web_apis.py`:**
* `get_weather`: Open-Meteo geocoding (şehir → koordinat) + forecast (anlık sıcaklık, hissedilen, nem, rüzgâr, WMO hava
  kodu, günlük min/maks). `WMO` sözlüğü kodları TR/EN açıklamaya çevirir.
* `get_current_datetime`: `zoneinfo` ile İstanbul saati, Türkçe gün adı.
* Yönlendiriciler: hava kelimeleri + şehir; "saat kaç / bugün günlerden ne / what time" kalıpları.

**`mcp_tools.py`:** MCP istemcisi (README §2.5).
* `MCPConnection`: her sunucu için tek, uzun ömürlü MCP oturumu; **kendi iş parçacığında, kendi olay döngüsünde**
  çalışır (senkron `build_services` içinden açılabilsin, istekler arasında açık kalsın). Windows'ta alt süreç için
  Proactor döngüsü açıkça seçilir. Çağrılar `run_coroutine_threadsafe` ile o döngüye gönderilir.
* `register()`: `tools/list` ile araçları keşfeder, şemadaki pydantic `title` alanlarını atar (gereksiz prompt
  token'ı), aracı yerel araçlarla aynı `Tool` olarak ekler. Sunucu açılamazsa uyarı loglanır ve atlanır.
* `ROUTERS`: MCP araçlarının deterministik yönlendiricileri **ajan tarafında**, araç adına göre. Seri no 5 hane,
  talep no 4 hane. Türkçe ünsüz yumuşamasına dikkat: "talep" → "talebim" (`tale[pb]`).
* Sunucu başına ayar (`MCPServerConfig`, `app/config.py`): `tools` (araç filtresi), `prefix` (aynı sunucunun iki
  örneği için ad öneki), `llm_visible` (false: yalnızca router çağırır, şema LLM'e gitmez), `max_chars` (uzun
  metinleri cümle sonundan kısaltır). Sunucular paralel başlatılır.
* `registry.py`'deki küçük ekler: `add()`, `on_close()/close()` (MCP alt süreci kapanışta düzgün durur),
  `Tool.llm_visible` ve `**kwargs` alan işleyicilere yalnızca şemada tanımlı argümanların geçirilmesi.

**`wikipedia.py`:** üçüncü taraf `wikipedia-mcp` sunucusunun (TR + EN iki örnek) router'ları. Yalnızca "X nedir /
what is X" kalıbında ve X sabit robotik terim sözlüğündeyse çalışır. Sözlük doğrulanmış madde başlıklarını tutar,
çünkü `get_summary` başlığı birebir istiyor ("LiDAR" yok, "LIDAR" var) ve arama konudan kayabiliyor ("Lidar
Höyük"). Soru bizim ürünümüzle ilgiliyse çalışmaz. Araç LLM'e gösterilmez, yani alan dışı sorular eskisi gibi reddedilir.

**`mcp_servers/nova_service/`:** ajandan bağımsız MCP sunucusu (`MCPServer`, resmi `mcp` SDK 2.x). Garanti, uzaktan
teşhis ve servis talebi araçları; kendi `seed.sql`'i bellekteki SQLite'a yüklenir. Kararlar metin olarak döner
("garanti YOK: …"), çünkü 4B model `"in_warranty": false` alanını "garanti devam ediyor" diye okudu.

**`__init__.py`:** `create_tool_registry()` — tüm araçların bağlandığı tek yer (yerel araçlar + `MCP_SERVERS`).

### 6.11 `app/agent/` — ajanın beyni

**`prompts.py`:**
* `SYSTEM_PROMPT` (İngilizce yazıldı — küçük modeller talimatları İngilizcede daha iyi izliyor) 6 kural: yalnızca verilen
  kaynaklardan bilgi; yoksa "bilmiyorum" + yardım edebileceği konular; isim/sayı/tarih uydurma yasak; araç kullanımı;
  alan dışı soruları reddet; kısa, markdown'sız, konuşma dili; istenen dilde cevap.
* **Sistem promptu bilinçli olarak sabittir** → KV önbelleği her turda yeniden kullanılır (§7).
* `_passage()`: `[1] metin (source: başlık)` — başlık **sonda**, çünkü küçük modeller ilk gördüklerini aynen okuyordu
  ("Rehber R2 Kullanım Kılavuzu > Acil Durdurma: ..." diye konuşuyordu).
* `build_system_prompt()`: `full` modda tüm bilgi tabanı `[K1] ... [K31]` olarak sistem promptuna eklenir.
* `build_user_message()`: sırasıyla `CONTEXT` (rag) veya `RELEVANT KNOWLEDGE: K3, K7 / none` (full), `LIVE DATA`
  (yönlendirilmiş araç sonuçları), `USER: soru`, dil talimatı.
* `LANGUAGE_INSTRUCTIONS`: talimat **hedef dilde** — "(Yalnızca Türkçe cevap ver.)". Model hem daha iyi uyuyor hem
  cevabın ilk kelimelerine doğru dille "ısınıyor".
* `TOOL_FILLERS` ("Bir saniye, kontrol ediyorum.") ve `ERROR_REPLIES`.

**`speech_text.py`:**
* `SentenceChunker`: token akışını konuşulabilir parçalara böler.
  * Cümle sonu: `.!?…` + boşluk veya satır sonu. "1.500" bölünmez (noktadan sonra boşluk yok); "Dr." gibi kısaltmalar atlanır.
  * **İlk parça özel:** ≥ 30 karakterden sonraki ilk virgülde kesilebilir — kullanıcı en çok ilk parçayı bekliyor; kısa
    ilk parça = daha az LLM token'ı beklemek + daha kısa TTS.
  * Sonraki parçalar tam cümle; 140 karakteri aşan uzun cümleler son virgülden kesilir.
* `clean_for_speech()`: TTS'e gitmeden önce markdown, bağlantı, `[1]` atıfları, madde işaretleri ve emojiler silinir;
  birimler dile göre okunur hâle gelir ("18°C" → "18 derece", "%60" → "yüzde 60", "@" → "et").
* `detect_text_language()`: yazılı giriş için işlev kelimesi sayacı (ve, bir, mi… / the, is, what…). Türkçe harfler yalnızca
  eşitlikte belirleyici — çünkü İngilizce cevaplar "Taşıyıcı T5" gibi Türkçe ürün adı içeriyor. Kesme işaretli ekler
  önce atılır ("T5'in" içindeki "in" İngilizce sayılmasın).
* `ungrounded_numbers()`: cevaptaki sayıları (nokta/virgül ayraçları kaldırılarak: "1.450.000" = "1450000") kanıt
  metninde arar; bulunamayanları döner. Halüsinasyonun en zararlı türü uydurma sayılar (fiyat, tarih, özellik).

**`agent.py`:**
* `AgentTurn`: bir turda olan her şey — cevap, söylenen metin (dolgu dahil), bulunan parçalar, araç sonuçları, LLM
  istatistikleri, süreler, kaynaksız sayılar, hata. `sources()` arayüz için kaynak listesi üretir.
* `LeakedToolCallGuard`: küçük modeller bazen araç çağrısını **düz metin JSON** olarak yazıyor
  (`,{"name": "get_current_datetime", "arguments": {}}`). Bu asla seslendirilmemeli. Guard, cevap `{ [ , < \``
  ile başlıyorsa turu sonuna kadar **tutar**; sonra `json.JSONDecoder.raw_decode` ile içindeki araç çağrılarını ayrıştırır.
  Geçerli araçsa gerçek çağrıya dönüştürülür, değilse metin olduğu gibi serbest bırakılır.
* `Conversation`: son 4 turun geçmişi. Bağlam (RAG parçaları) geçmişte **saklanmaz** — her turda yeniden aranır;
  böylece prompt kısa kalır.
* `Agent`:
  * `__init__`: moda göre sistem promptu; `full` mod için parça etiketleri (`K1`...).
  * `warmup()`: model yükleme + önbellek ısıtma.
  * `_retrieve()`: arama; **takip sorusu desteği** — sonuç yoksa veya soru ≤ 6 kelimeyse ("peki boyu?") önceki
    kullanıcı sorusuyla birleştirip tekrar arar. Ardından **bağıl kesme**: en iyi skorun 0.15 altındakileri at.
  * `respond()` — çekirdek döngü:
    1. Yönlendirilmiş araçlar ve arama **paralel** başlar.
    2. Mesajlar: sistem + geçmiş + yeni kullanıcı mesajı.
    3. En fazla `max_tool_rounds + 1` (3) LLM turu: metin akarken yield edilir (guard'dan geçerek); model araç isterse
       ve henüz hiçbir şey söylenmediyse dolgu cümlesi söylenir, araçlar paralel çalışır, sonuçlar `tool` mesajı olarak
       eklenir, LLM tekrar çağrılır. Son turda araç verilmez (sonsuz döngü engeli).
    4. Hata olursa özür cümlesi söylenir (kullanıcı sessizlikte kalmaz).
    5. Kaynaksız sayı kontrolü; geçmişe ekleme (dolgu cümlesi geçmişe girmez).

### 6.12 `app/pipeline/` — oturum ve metrikler

**`metrics.py`:**
* `TurnMetrics`: girişte `endpoint_ms`, `stt_ms`, `stt_rtf`; akıl yürütmede `retrieval_ms`, `llm_ttft_ms`, `llm_prefill_ms`,
  `llm_tokens_per_s`, `tool_ms`; çıkışta `tts_first_ms`, `tts_rtf`; ve kullanıcı için asıl önemli olan
  **`first_audio_ms`** (kullanıcı sustu → ilk ses gönderildi).
* `MetricsStore`: son 1000 tur; `summary()` her metrik için n, ortalama, p50, p90 (sesli/yazılı ayrı; kesilen turlar hariç).

**`session.py`** — `VoiceSession`, sistemin orkestra şefi:
* **Durum:** `endpointer`, `conversation`, `turn_task` (çalışan tur), `client_playing` (istemci hâlâ ses çalıyor mu),
  `_turn_audio`, `_turn_spoke` (bu tur ses gönderdi mi), `_carry_audio`, `_send_lock`.
* **Neden `_send_lock`?** Tur görevi, TTS görevi ve metrik gönderimi aynı WebSocket'e eşzamanlı yazabilir; kilit mesajların
  karışmasını önler.
* **Protokol:** ikili ses mesajları `struct.pack("<II", turn_id, sample_rate) + pcm` — 8 baytlık başlık sayesinde istemci,
  kesilmiş bir turun geç gelen sesini (`turn_id <= droppedTurn`) atabilir.
* `on_audio()`:
  * Yarı çift yönlü mod (`BARGE_IN=false`): ajan konuşurken mikrofon yok sayılır.
  * Ajan konuşurken `min_speech_ms` 350'ye çıkar — hoparlörden mikrofona sızan kendi sesiyle kesilmesin.
  * `SPEECH_START` + ajan meşgul → `interrupt("barge_in")`.
* `interrupt()`:
  * **Devam eden cümle birleştirme:** kullanıcı yarım saniye duraklayıp devam ettiyse VAD turu bitirmiş, STT başlamış olur;
    kullanıcı devam edince bu bir "araya girme" sayılır. Ajan henüz ses göndermemişse, kesilen turun sesi
    `_carry_audio`'ya alınır ve bir sonraki ifadenin **başına eklenir** — cümlenin ilk yarısı kaybolmaz.
* `_run_turn()`: STT (executor'da) → `fix_domain_terms` → `transcript` olayı → TTS görevi başlat → ajan token'larını
  `SentenceChunker` ile cümlelere böl, kuyruğa koy → bitince `answer` olayı. `CancelledError`'da söylenen kısım "…" ile
  geçmişe yazılır (takip sorusu anlamlı kalsın). `finally`'de TTS görevi iptal + metrikler.
* `_speak()`: kuyruktan cümle al → **kontroller** → temizle → `sentence` olayı → Piper (executor'da) → ilk seste
  `first_audio_ms`'i kaydet → ses gönder. **LLM bir sonraki cümleyi üretirken bu cümle sentezleniyor** — boru hattı
  (pipelining) burada. Okunmadan önceki kontroller:
  * **Sayı kontrolü:** cümlede soru/bağlam/araç çıktısında olmayan bir sayı varsa cümle okunmaz, bir kez "Bu konuda
    kaynaklarımda kesin bir bilgi yok." denir. Cümle virgülde kesilmiş bir ilk parçaysa devamı da atılır (yoksa
    "bulutlu." gibi anlamsız bir kuyruk tek başına okunuyordu).
  * **Tekrar:** aynı cümle ikinci kez gelirse okunmaz (küçük modellerin döngüye girmesi).
* `_finish_metrics()`: `AgentTurn` + LLM istatistiklerinden `TurnMetrics`'i doldurur, loglar, istemciye yollar
  (`_send_quietly`: soket kapanmışsa sessizce geç).

### 6.13 `web/` — tarayıcı istemcisi

**`pcm-worklet.js`** (ses iş parçacığında çalışır):
* **63 katsayılı FIR alçak geçiren filtre** (Blackman pencereli sinc, kesim 7 kHz): 16 kHz'e inerken 8 kHz üstü
  frekanslar aliasing yapıp konuşma bandına katlanmasın. İlk sürümdeki tek kutuplu filtre (`lp += α·(x − lp)`) 12 kHz'i
  yalnızca ~5 dB bastırıyordu — "s, z, ş" seslerinin enerjisi bozulmuş halde Whisper'a gidiyordu; FIR ile −96 dB.
* **Doğrusal enterpolasyon** ile yeniden örnekleme: `step = sampleRate / 16000` (48k için 3, 44.1k için 2.756);
  `pos` kesirli okuma konumu bloklar arası taşınır.
* 1024 örnek (64 ms) dolunca `Int16Array` ve RMS (ses seviyesi; mikrofon halkası animasyonu için) gönderilir;
  `transfer` listesi sayesinde kopyalama olmadan.
* Node'da sahte bir AudioWorklet ortamıyla test edildi: 48 kHz ve 44.1 kHz'de 440 Hz sinüs → çıkışta 439.8 Hz.

**`app.js`:**
* **Durumlar:** idle / listening / hearing / thinking / speaking → düğme rengi ve durum metni.
* `connect()`: WebSocket; koparsa 1.5 sn sonra yeniden bağlanır.
* `startMic()`: `getUserMedia` ile **yankı giderme ve otomatik kazanç açık, gürültü bastırma kapalı** (tarayıcının
  gürültü bastırması konuşmanın uçlarını da kırpıp tanımayı bozuyordu; Whisper gürültüye zaten dayanıklı). Worklet
  çıktısı 0 kazançlı bir düğüme bağlanır — tarayıcı grafı işlemeye devam etsin ama mikrofon hoparlöre geri verilmesin.
  Mikrofon bir kez açılır; "Dinlemeyi durdur" akışı serbest bırakmaz, yalnızca göndermeyi duraklatır ve sunucuya
  `{type: "mic", on}` bildirir (sunucu tur sonu durumunu sıfırlar).
* `onAudio()`: başlık okunur, `AudioBuffer` oluşturulur (tarayıcı 22.05 kHz'i kendi hızına çevirir), `playhead` ile
  kesintisiz sırayla planlanır (`max(şimdi + 30 ms, önceki parçanın bitişi)`).
* `reportPlaying()`: çalma başladı/bitti bilgisini sunucuya bildirir (barge-in kararları için).
* `stopPlayback()`: araya girmede tüm planlanmış kaynakları anında durdurur.
* Arayüz: kullanıcı/ajan balonları, dil rozeti, kaynak çipleri (📄 doküman, 🔧 araç), ⚠ kaynaksız sayı uyarısı.
* `showMetrics()`: son turun gecikmesini renkli yığılmış çubukla gösterir; "LLM ilk cümle" dilimi
  `first_audio − (VAD + STT + arama + TTS)` olarak hesaplanır.

**`index.html` / `style.css`:** tek sayfa; CSS değişkenleriyle açık/koyu tema (`prefers-color-scheme`), 820 px altında tek sütun.

### 6.14 `data/` — örnek bilgi kaynakları (kurgusal)

* `docs/`: 6 doküman — şirket bilgisi, ürün kataloğu, SSS, Rehber R2 kılavuzu (TR); garanti/iade, servis/bakım (EN).
  Karışık dil bilinçli: diller arası aramayı göstermek için.
* `seed.sql`: `products` (fiyat, stok, teslim süresi, takma adlar), `orders` (5 sipariş, farklı durumlar),
  `service_centers` (5 şehir). **Değişken bilgi (fiyat/stok) dokümanda değil veritabanında** — gerçek hayatta da böyle
  olmalı; dokümandaki eski fiyat halüsinasyon kaynağı olur.

### 6.15 `eval/questions.json`

26 soru; her biri `q`, `lang` ve şunlardan bazıları: `expect` (cevapta geçmesi gereken ifadelerden biri), `refuse`
(kaynakta yok, reddetmeli), `forbid` (ret cevabında geçmemesi gerekenler — ör. "Ankara": "bilgim yok ama başkent
Ankara" bir ret değildir), `tool` (çağrılması gereken araç). Canlı kullanımda görülen hatalar sete eklenir:
"Akınsoft şirketi nerede?" ve STT'nin bozduğu "Zehbe Z2'nin piyomru ne kadar?".

### 6.16 `scripts/`

| Betik | Ne yapar | Önemli teknik |
|---|---|---|
| `download_models.py` | Whisper, Piper sesleri, embedding modelini indirir (idempotent) | Piper sesleri HF'den `local_dir` ile indirilip düz isimle taşınır |
| `evaluate.py` | 26 soruyu `/api/chat`'e sorar; olgu, ret, araç, dil, kaynaksız sayı ölçer | Her soru yeni oturum (geçmiş etkisi yok) |
| `eval_matrix.ps1` | Model × bilgi modu kombinasyonlarında sunucuyu yeniden başlatıp değerlendirir | Ortam değişkenleriyle yapılandırma; `-Port 8001` ile Docker'daki ajan açıkken de çalışır, `-Suffix _gpu` rapor adına eklenir |
| `benchmark_voice.py` | Sentetik konuşmayı **gerçek zamanlı** akıtıp uçtan uca gecikmeyi ölçer | 32 ms'lik paketleri saat ayarlı gönderir; sonra sessizlik (açık mikrofon gibi) |
| `benchmark_stt.py` | FLEURS gerçek insan sesiyle WER/RTF; standart vs. bizim çözümleme; `--models`, `--variants`, `--langs`, `--threads`, `--beam` | tar.gz'yi **akışla** okuyup ilk N kaydı alır (750 MB indirmeden) |
| `report_table.py` | `reports/eval_*.json` → README tablosu | — |

### 6.17 `tests/` — 40 birim testi (model gerektirmez, ~10 sn)

| Dosya | Test ettiği | Teknik |
|---|---|---|
| `test_vad.py` | Tur sonu durum makinesi | `ScriptedVAD`: sahte olasılık dizisi → deterministik |
| `test_agent.py` | Ajan döngüsü, araç çağrısı, dolgu, sızan JSON, full mod, hata | `ScriptedLLM`: önceden yazılmış olay dizileri |
| `test_tools.py` | Sipariş numarası biçimleri, ürün eşleştirme, yönlendiriciler, şehir çıkarma | Geçici SQLite (`tmp_path`) |
| `test_mcp_tools.py` | Nova Servis mantığı, MCP ve Wikipedia yönlendiricileri, gerçek stdio bağlantısı, sunucu ayarları, açılamayan sunucunun atlanması | Sabit "bugün" (`monkeypatch`); Wikipedia'ya ağ isteği atılmaz |
| `test_knowledge.py` | Türkçe küçük harf, kök kesme, parçalama, BM25, eşik | `FakeEmbedder` (harf frekansı vektörü) |
| `test_speech_text.py` | Cümle bölme, TTS temizliği, dil tespiti, sayı kontrolü, terim düzeltme, tekrar kırpma | — |

**Ders:** dış bağımlılıkları (LLM, VAD modeli) arayüz arkasına koymak, onları sahteleriyle değiştirip **hızlı ve
deterministik** test yazmayı mümkün kılıyor.

### 6.18 Docker ve yapılandırma dosyaları

* **`Dockerfile`:** `python:3.11-slim` + `libgomp1` (CTranslate2/ONNX Runtime'ın OpenMP bağımlılığı) + `curl` (sağlık
  kontrolü). Modeller imaja **gömülmez**; ilk açılışta `/models` birimine indirilir → imaj küçük (~960 MB), sonraki
  açılışlar anında. `CMD`: önce `download_models`, sonra `exec python -m app` (`exec`: sinyaller doğrudan Python'a gitsin).
* **`docker-compose.yml`:** üç servis:
  * `ollama`: LLM sunucusu; sağlık kontrolü `ollama list`.
  * `ollama-pull`: tek seferlik iş; modeli çeker ve çıkar.
  * `agent`: `depends_on: ollama-pull: service_completed_successfully` → model inmeden başlamaz.
  * YAML anchor (`x-llm-model: &llm_model`) — model adı tek yerde.
  * `LLM_BASE_URL` sabit — `.env`'deki yerel adresin konteyneri bozmasını önlemek için.
* **`docker-compose.host-ollama.yml`:** override (GPU yolu); `host.docker.internal` ile bilgisayardaki Ollama'ya
  bağlanır, `depends_on: !reset {}` ile Ollama servislerine bağımlılığı kaldırır. GPU'ya göre ayarları da değiştirir:
  `LLM_MODEL=${LLM_MODEL_GPU:-hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:IQ4_XS}`, `LLM_NUM_CTX=3072`,
  `LLM_NUM_THREAD=4`. Neden ayrı dosya? Docker Desktop (WSL2) AMD GPU'yu konteynere geçirmiyor; GPU yalnızca Windows'ta
  çalışan Ollama'dan kullanılabiliyor. Konteynerdeki Ollama (tam Docker kurulumu) CPU'da çalışır; orada aynı 4B
modelin Q4_K_M sürümü kullanılır (k-quant'lar CPU'da IQ türlerinden hızlı).
* **`.env.example`:** önemli ayarlar ve açıklamaları. **`requirements.txt`:** sabitlenmiş sürümler (tekrarlanabilir kurulum).
* **`.gitignore` / `.dockerignore`:** `models/`, `.venv/`, üretilen `nova.db`, loglar dışarıda.

---

## 7. Gecikme mühendisliği

Sesli ajanda en önemli metrik: **kullanıcı sustu → ilk ses**. Bu makinede yapılanlar, etkileriyle (1-12 yalnızca CPU döneminde, 13-16 GPU geldikten sonra):

| # | Teknik | Neyi kısaltır | Etki (ölçülen) |
|---|---|---|---|
| 1 | VAD ile otomatik tur sonu (500 ms) | Bas-konuş düğmesi yok | ~0.65 sn sabit maliyet |
| 2 | Whisper tek geçiş + kısa pencere | STT | 2-3 sn'lik ifadede ~7-11 sn → ~1 sn; doğruluk korunarak |
| 3 | LLM akışı + cümle bölme + boru hattı | Cevabın tamamını bekleme | İlk cümle bitince ses başlar |
| 4 | Sabit sistem promptu (KV önbelleği) | LLM prefill | Bağlamsız soruda TTFT ~1.4 sn (önbelleksiz onlarca sn olurdu) |
| 5 | Az bağlam (top-2 + bağıl kesme + eşik) | LLM prefill | Prefill, bağlam token'ıyla doğrusal artıyor |
| 6 | Deterministik araç yönlendirme | İkinci LLM turu | qwen2.5:3b'de TTFT p50 10.2 → 5.2 sn (8 iş parçacığıyla birlikte) |
| 7 | `LLM_NUM_THREAD=8` | Prefill | Bu CPU'da ~%35 |
| 8 | İlk parçayı virgülde kesme | LLM ilk cümle + TTS ilk parça | Uzun ilk cümlelerde ~1-2 sn |
| 9 | Araç beklerken dolgu cümlesi | **Algılanan** gecikme | Sessizlik yerine "Bir saniye..." |
| 10 | Barge-in: HTTP akışını kapatma | Boşa üretim | Ollama üretimi hemen keser, CPU serbest kalır |
| 11 | Modelleri önceden yükleme + ısıtma | İlk tur | İlk istekte model yükleme beklemesi yok |
| 12 | Model seçimi (1.7B) | Prefill + üretim | 3B'ye göre TTFT 4.6 → 3.0 sn |
| 13 | LLM'i GPU'ya almak (ana makinedeki Ollama, Vulkan) | Prefill + üretim + TTS | 1.7B ile ilk ses 7.6 → 3.4 sn |
| 14 | GPU'da bağlam 4096 → 3072, IQ4_XS kuantizasyon | 4B'nin CPU'ya taşan katmanları | 4B'nin ~%90'ı GPU'da |
| 15 | GPU kurulumunda `LLM_NUM_THREAD=4` | CPU'daki katmanların TTS ile yarışması | 4B ile ilk ses 7.4 → 6.0 sn, TTS ilk cümle 1.1 → 0.56 sn |
| 16 | Docker'da `STT_CPU_THREADS=4` (8 değil) | STT | 1.91 → 1.27 sn |

**Darboğaz analizi:** CPU'da ölçülen 7.6 sn'nin ~%40'ı LLM, ~%15'i STT, ~%10'u VAD beklemesi, gerisi ilk cümlenin
üretimi ve TTS. LLM GPU'ya alınınca (1.7B) 3.4 sn'ye indi ve darboğaz STT oldu. GPU varsayılanı 4B olunca ilk ses ~6 sn:
bu bilinçli bir doğruluk ↔ gecikme takası (kullanıcı kararı). Sonraki kazançlar: akışlı STT (konuşurken çözümleme),
Whisper'ı GPU'ya almak (whisper.cpp Vulkan), modeli tamamen alabilecek 8 GB'lık bir kart.

**KV önbelleği neden bu kadar önemli?** LLM her istekte promptun tamamını işlemek zorunda. Prompt
`[sistem promptu + araç şemaları (~650 token)] + [geçmiş] + [yeni mesaj]` ise ve ilk kısım önceki istekle **birebir aynıysa**,
sunucu o kısmın ara sonuçlarını (KV) yeniden kullanır. Bu yüzden: tura özgü hiçbir şey sistem promptuna konmaz; araç
listesi her turda aynıdır (bazı turlarda araç vermemek önbelleği bozardı); bağlam en sondaki mesajdadır.

---

## 8. Doğruluk ve halüsinasyon kontrolü

Katmanlı savunma — tek bir önlem yetmiyor:

1. **Prompt kuralları:** yalnızca verilen kaynak; yoksa "bilmiyorum"; isim/sayı/tarih uydurma yasak; alan dışı ret.
2. **Açık "bilgi yok" sinyali:** eşik altı aramada bağlam yerine `(no relevant information found)`. Ölçümlerde **en etkili
   önlem**: bu sinyalin olmadığı full modda model "Genel müdür Ahmet Yılmaz" diye uydurdu.
3. **Eşik kalibrasyonu:** bilgi sorularının skoru ≥ 0.62, alan dışı soruların ≤ 0.43 → eşik 0.45.
4. **Yapılandırılmış veri araçtan:** fiyat, stok, sipariş, adres, hava → modelin hafızasından değil araçtan;
   bulunamayan kayıt açıkça `found: false`.
5. **Deterministik yönlendirme:** küçük modelin "sorguluyorum, lütfen bekleyin" deyip aracı hiç çağırmaması engellenir.
6. **Sızan araç çağrısı onarımı:** JSON metni seslendirilmez, gerçek çağrıya çevrilir.
7. **Seslendirme öncesi sayı kontrolü:** her cümle **okunmadan önce** kontrol edilir; kaynakta olmayan bir sayı içeren
   cümle okunmaz, yerine bir kez "Bu konuda kaynaklarımda kesin bir bilgi yok." denir; loglanır ve arayüzde ⚠ ile
   işaretlenir. Gerçek örnekler: "pil ömrü fiyatı 499 lira", hava aracı 8°C/24°C dediği halde "12°C ile 18°C arası".
8. **Tekrar koruması:** aynı cümle ikinci kez okunmaz (1.7B'nin bir cevabı aynı cümleyi üç kez tekrarlamıştı).
9. **Model seçimi:** 4B Instruct, 1.7B'ye göre alan dışı ve bozuk tanınmış sorularda belirgin biçimde daha az uyduruyor
   (ret 3/5 → 4/5, kaynaksız sayı 1 → 0). Model boyutu, prompt kurallarının uygulanıp uygulanmadığını belirliyor.
10. **Şeffaflık:** her cevapta kullanılan kaynaklar gösterilir ve loglanır.
11. **Düşük sıcaklık (0.2), kısa cevap (200 token).**
12. **STT tarafı:** bilinen Whisper uydurmaları, konuşma-yok çıktıları ve tekrar döngüleri atılır; yanlış duyulan ürün
    adları düzeltilir ("Zehbe Z2" → "Rehber R2"); yönlendirici bitişik yazımları tanır ("havanası"); LLM'e konuşmanın
    tanıma hatası içerebileceği söylenir (prompt kuralı 7).
13. **Ölçüm:** değerlendirme seti ile her değişikliğin etkisi sayıyla görülür; canlı kullanımda görülen hatalar sete eklenir.

**Bilinen sınırlar:** 4B dahil denenen hiçbir model "Türkiye'nin başkenti" sorusunu reddetmiyor (cevabı biliyor ve
söylüyor); sözel uydurmalar (sayı içermeyen) sayı kontrolüyle yakalanmıyor. Çözüm önerileri: alan dışı soru
sınıflandırıcısı, daha güçlü model, cevap sonrası LLM tabanlı doğrulama (gecikme pahasına).

---

## 9. Ölçüm metodolojisi ve sonuçlar

**Donanım:** AMD Ryzen 7 3750H (4 çekirdek/8 iş parçacığı), 16 GB RAM, Windows 11. İlk ölçümler yalnızca CPU ile;
sürücü güncellemesinden sonra LLM, Radeon RX 560X 4 GB'da (Vulkan) — "GPU" diye belirtilen ölçümler.

### 9.1 Uçtan uca sesli gecikme (`benchmark_voice.py`)

**İlk ölçüm — yalnızca CPU, qwen3:1.7b + RAG, 16 tur:**

| Adım | p50 |
|---|---|
| VAD tur sonu | 0.65 sn |
| STT | 0.88 sn (RTF 0.42) |
| Arama | 11 ms |
| LLM ilk token | 3.0 sn (~9 token/sn) |
| TTS ilk cümle | 1.0 sn (RTF 0.21; LLM ile eşzamanlı) |
| **Kullanıcı sustu → ilk ses** | **7.6 sn** (p90 10.7 sn) |

**Aynı makine, LLM GPU'da (RX 560X, Vulkan, sürücü güncellemesinden sonra), 8 tur:** LLM ilk token 0.72 sn,
~26 token/sn, TTS ilk cümle 0.31 sn, STT 1.24 sn (CPU, Docker içinde) → **ilk ses 3.4 sn**. Darboğaz LLM'den STT'ye kaydı.

**Son hal, A kurulumu — GPU, Qwen3-4B-Instruct-2507 IQ4_XS, 8 tur:** VAD 0.75 sn, STT 1.28 sn, LLM ilk token 1.97 sn,
~8.6 token/sn, TTS ilk cümle 0.56 sn → **ilk ses 6.0 sn**. Daha doğru model için ~2.6 sn ek gecikme kabul edildi.

**Son hal, B kurulumu — tam Docker (her şey konteynerde, CPU), 4B Instruct Q4_K_M, 8 tur:** STT 1.19 sn, LLM ilk token 10.4 sn, ~4.2 token/sn
→ **ilk ses 16.0 sn** (p90 24.2). Cevaplar doğru; bedel gecikme. Ders: aynı model, aynı kod — yalnızca *nerede*
çalıştığı ilk sesi 6 sn ile 16 sn arasında değiştiriyor. Ürün kararı (doğruluk mu hız mı) donanımla birlikte verilmeli.

*Yöntem:* sorular Piper ile sese çevrilir, 32 ms'lik paketlerle saat ayarlı gönderilir (gerçek mikrofon gibi), ardından
sessizlik akar. Süre istemcide, son konuşma örneği gönderildiğinde başlar, ilk ses paketi geldiğinde biter.

### 9.2 STT, gerçek insan sesi (`benchmark_stt.py`, FLEURS, 10.5 sn ortalama)

| Yol | TR WER | TR süre/ifade |
|---|---|---|
| Standart faster-whisper | 21.3% | 12.8 sn |
| Bizim (kısa pencere, promptsuz — varsayılan) | 20.9% | 3.0 sn |
| Bizim + hotwords promptu | 35.5% | 2.9 sn |

### 9.3 Doğruluk (`evaluate.py`; CPU tablosu setin ilk hali olan 24 soruyla)

| LLM | Mod | Geçen | Doğru ret | Kaynaksız sayı | TTFT p50 |
|---|---|---|---|---|---|
| **qwen3:1.7b** | **rag** | **23/24** | **3/4** | **0** | 3.0 sn |
| qwen3:1.7b | full | 22/24 | 2/4 | 0 | 3.7 sn |
| qwen2.5:3b | rag | 22/24 | 3/4 | 1 | 4.6 sn |
| qwen2.5:3b | full | 20/24 | 1/4 | 1 | 3.3 sn |
| qwen2.5:1.5b | rag | 20/24 | 1/4 | 0 | 2.4 sn |
| qwen2.5:1.5b | full | 20/24 | 1/4 | 0 | 1.9 sn |

**GPU gelince model değişti (RX 560X 4 GB, 26 soru — canlı kullanımda görülen hatalar sete eklendi):**

| LLM (4-bit) | VRAM'e sığan | Geçen | Doğru ret | Kaynaksız sayı | TTFT p50 | İlk ses (sesli test) |
|---|---|---|---|---|---|---|
| qwen3:1.7b Q4_K_M | %100 | 23/26 | 3/5 | 1 | 1.2 sn | 3.4 sn |
| qwen3:4b Q4_K_M | %75 | 24/26 | 3/5 | 0 | 2.5 sn | — |
| **Qwen3-4B-Instruct-2507 IQ4_XS** | %90-95 | 24/26 | **4/5** | **0** | 1.9 sn | 6.0 sn |

Ders: bir modelin "boyutu" kadar **karta sığan kısmı** önemli. `qwen3:4b` (2.5 GB) 4 GB kartın kullanılabilir ~2.9 GB'ına
bağlam belleğiyle birlikte sığmadı; aynı 4B'nin talimat sürümünün biraz daha küçük 4-bit kuantizasyonu (IQ4_XS:
"importance-matrix" ile hangi ağırlıkların daha hassas olduğuna bakarak bit dağıtan yöntem) ve bağlamı 4096 → 3072
indirmek neredeyse tamamını GPU'ya aldı. Bedel: daha yavaş üretim, ilk ses 3.4 → 6.0 sn; karşılığında "Akınsoft şirketi
nerede?" gibi alan dışı sorularda uydurma yok. GPU kurulumunun varsayılanı bu (`docker-compose.host-ollama.yml`),
tam Docker kurulumunda (CPU) aynı modelin Q4_K_M sürümü. Bir de iş parçacığı dersi: Whisper Windows'ta 8 iş parçacığıyla daha hızlıyken Docker
(WSL2) içinde daha yavaştı (1.27 → 1.91 sn); ayarlar **çalışacağı ortamda** ölçülmeli.

**Ölçüm dürüstlüğü notları** (mülakatta sorulabilir): sesli test sentetik sesle yapıldı (Piper'ın Türkçe sesi robotik →
tanıma hataları gerçekten kötü), bu yüzden STT doğruluğu ayrıca gerçek sesle ölçüldü; eşik değerlendirme setiyle
kalibre edildi (aşırı uyum riski); bazı son iyileştirmelerden sonra sesli test yeniden çalıştırılmadı.

---

## 10. Geliştirme sırasında karşılaşılan sorunlar ve çözümleri

Bu bölüm "nasıl düşünüldü" sorusunun cevabı — mülakatta en çok değer gören kısım genelde budur.

| # | Sorun | Nasıl fark edildi | Kök neden | Çözüm |
|---|---|---|---|---|
| 1 | Piper Windows'ta açılmıyor | `D:/a/piper1-gpl/...phontab: No such file` hatası | espeak-ng C kütüphanesi ASCII olmayan yolu ("akın") açamıyor, derlenmiş varsayılan yola düşüyor | 8.3 kısa yol / geçici kopya (`_ascii_safe_espeak_dir`) |
| 2 | STT 2-3 sn'lik cümleye 11-13 sn | Bileşen testi | 30 sn dolgu + kodlayıcının iki kez çalışması | Tek geçiş + kısa pencere |
| 3 | Kısa pencerede "Merhaba Merhaba Merhaba..." | Tekrar testi | Az sessizlik payıyla çözümleyici bitişi göremiyor | 2 sn pay + token bütçesi + tekrar kırpma |
| 4 | LLM TTFT ~10 sn | İlk değerlendirme | CPU'da prefill 25 token/sn; bağlam + ikinci araç turu | Az bağlam, yönlendirme, 8 iş parçacığı, daha küçük model |
| 5 | 3B model aracı çağırmıyor ("sorgulanıyor, bekleyin") | Değerlendirme: araç 5/7 | Küçük modelin araç çağırma güvenilirliği | Deterministik yönlendirici (→ 7/7) |
| 6 | Araç çağrısı JSON olarak sesleniyordu | Değerlendirme çıktısı | Model çağrıyı metin kanalına yazdı | `LeakedToolCallGuard` |
| 7 | İngilizce cevap Türkçe sayıldı | Dil kontrolü başarısız | "Taşıyıcı"daki "ş"; "T5'in" → "in" İngilizce kelime | İşlev kelimesi önceliği + kesme eklerini atma |
| 8 | Full modda isim uydurma | Ret 1/4 | "Bilgi yok" sinyali yok | Kimlik ipucu ekleme; yine de RAG varsayılan |
| 9 | "Türkiye'nin başkenti" bağlam alıyordu | Skor analizi | Alakasız parça 0.407 skorla eşiği (0.40) geçiyordu | Veriyle kalibrasyon → 0.45 |
| 10 | Model bağlam başlığını okuyordu | Cevaplarda "Kılavuz > Acil Durdurma:" | Küçük model ilk gördüğünü kopyalıyor | Başlığı parçanın sonuna taşıma |
| 11 | Hotwords Türkçe WER'i %21 → %36 yaptı | Gerçek sesle STT benchmark'ı | Ürün adı listesi "önceki metin" olarak çözümleyicinin üslubunu bozuyor | Hotwords kapalı, STT sonrası regex düzeltme |
| 12 | Cümle ortası duraklamada cümlenin yarısı kayboluyor | Kod incelemesi | Duraklama tur bitirir, devam etmek barge-in sayılır | Kesilen turun sesini sonraki ifadeye ekleme |
| 13 | Docker'da `.env`'deki localhost adresi | Kod incelemesi | Compose `.env`'i değişken ikamesi için de okuyor | Compose'ta adresi sabitleme |
| 15 | Ekran kartı (RX 560X) kullanılmıyordu | Ollama logu: "inference compute ... library=cpu" | 2020 tarihli AMD sürücüsüyle Vulkan kartı listelemiyordu | Sürücü güncellemesi → model %100 GPU'da, ilk ses 7.6 → 3.4 sn (`ollama ps` ile doğrulama) |
| 14 | Değerlendirme matrisi yanlış yapılandırmalarla çalıştı | Çıktıda "rag,qwen2.5:3b full,..." | PowerShell `-File` dizi parametresini tek metin olarak geçirdi | Virgülle ayrılmış tek metin parametresi |
| 16 | 1.7B alan dışı soruda uyduruyor ("Akınsoft şirketi … Ankara'da"), aynı cümleyi 3 kez tekrarlıyor | Kullanıcının canlı denemesi | Küçük modelin sınırı; kurallar promptta olduğu halde uygulanmıyor | Hatalar değerlendirme setine eklendi; GPU'da 4B Instruct (ret 3/5 → 4/5, kaynaksız sayı 1 → 0); tekrar eden cümle okunmuyor |
| 17 | `qwen3:4b` GPU'da yavaş (cevap 6.7 sn) | `ollama ps`: "25%/75% CPU/GPU" | 2.5 GB model + bağlam belleği 4 GB kartın kullanılabilir 2.9 GB'ını aşıyor | Daha küçük 4-bit (IQ4_XS) talimat sürümü + bağlam 3072 → %90 GPU |
| 18 | 4B ile TTS yavaşladı (ilk cümle 0.3 → 1.1 sn) | Sesli benchmark'ın adım dökümü | GPU'ya sığmayan katmanlar 8 CPU iş parçacığıyla TTS'e rakip | GPU kurulumunda `LLM_NUM_THREAD=4` → 0.56 sn |
| 19 | STT Docker'da yavaşladı (1.24 → 1.91 sn) | Sesli benchmark | Windows'ta ölçüp 8 iş parçacığı seçilmişti; WSL2'de 8 iş parçacığı daha yavaş | Varsayılan 4; ayarlar çalışacağı ortamda ölçülmeli |
| 20 | "havan nasıl" / "havanası" hava aracını tetiklemedi, model sıcaklık uydurdu (sayı kontrolü yakaladı) | Sesli benchmark logu: "sentence withheld" | Yönlendirici regex'i "hava nasıl"ı boşluklu bekliyordu | `\bhavan?\s*(durumu|nas|kac)` + test |

---

## 11. Mülakatta sorulabilecek sorular ve cevapları

**S: Neden uçtan uca (speech-to-speech) tek bir model değil de ayrık bir boru hattı (VAD→STT→LLM→TTS)?**
C: Ödev doğruluk, kaynak bağlılığı, değiştirilebilirlik ve açık platform istiyor. Ayrık boru hattında her adım ölçülebilir,
değiştirilebilir ve araya RAG/araç eklenebilir; açık kaynak sesten sese modeller Türkçede ve araç kullanımında henüz zayıf
ve CPU'da çok ağır.

**S: Gecikmeyi nasıl ölçtün, neden o tanım?**
C: Kullanıcı açısından anlamlı olan, son konuşma sesinden ilk cevap sesine kadar geçen süre. Hem sunucuda (son sesli VAD
çerçevesinden ilk ses paketine) hem istemcide (gerçek zamanlı akıtan benchmark) ölçtüm; ikisi tutarlı (7.3 / 7.4 sn ort.).

**S: RTF nedir, değerlerin ne anlama geliyor?**
C: İşlem süresi / ses süresi. STT 0.42: 1 sn ses 0.42 sn'de yazıya döner. TTS 0.1-0.2: 1 sn konuşma 0.1-0.2 sn'de
üretilir — oynatmadan çok daha hızlı, bu yüzden konuşma başladıktan sonra takılma olmaz.

**S: En büyük darboğaz ne, GPU olsa ne değişirdi?**
C: CPU'da LLM (~%40). Bunu ölçtüm: sürücü güncellemesinden sonra Ollama aynı dizüstündeki 4 GB'lık RX 560X'i Vulkan ile
kullandı; 1.7B modelde LLM ilk token 3.0 → 0.72 sn, üretim 9 → 26 token/sn, kullanıcının beklediği süre 7.6 → 3.4 sn.
Kodda tek satır değişmedi — bileşenler arayüz arkasında olduğu için. Sonra GPU'nun kazandırdığı zamanın bir kısmını
doğruluğa harcadım: 4B Instruct modele geçtim (ilk ses ~6 sn, daha az uydurma).

**S: 4B model neden 4 GB'lık karta sığmadı, ne yaptın?**
C: Ollama günlüğüne göre kullanılabilir VRAM ~2.9 GB. `qwen3:4b` Q4_K_M (2.5 GB) + bağlam belleği + hesap tamponları
bunu aşınca katmanların %25'i CPU'ya taştı ve cevap süresi 6.7 sn'ye çıktı. Üç şey yaptım: aynı boyutta talimat sürümünün
daha küçük 4-bit kuantizasyonunu (IQ4_XS, 2.1 GB) seçtim; gerçek prompt uzunluğunu ölçüp (en fazla ~1000 token) bağlamı
4096 → 3072'ye indirdim; CPU'da kalan katmanların iş parçacığını 8 → 4'e indirdim, çünkü TTS ile yarışıyorlardı. Sonuç:
%90 GPU, ilk ses 7.4 → 6.0 sn.

**S: Tamamen Docker'da neden bu kadar yavaş (16 sn)?**
C: Docker Desktop AMD GPU'yu konteynere geçirmiyor; 4B model CPU'da ~4 token/sn üretiyor ve prompt işlemesi ~10 sn
sürüyor. Bu kurulum taşınabilirlik için var; GPU'lu kullanım için ana makinedeki Ollama'ya bağlanan override dosyası var.
NVIDIA kartlı bir Linux sunucuda konteyner GPU'yu doğrudan kullanabilir.

**S: Halüsinasyonu nasıl kontrol ettin? Hangisi en etkiliydi?**
C: §8'deki katmanlar. Ölçümle en etkili olan: kalibre edilmiş eşikle "ilgili bilgi yok" sinyali — RAG'ın ret oranı
CAG'dan yüksek çıktı. Yapılandırılmış veriyi araçtan almak ve sayı kontrolü de önemli.

**S: Neden hibrit arama?**
C: Embedding anlamı yakalar ve diller arası çalışır (TR soru → EN doküman) ama "T5", "E02" gibi kodları bulanıklaştırır;
BM25 tam eşleşmeyi yakalar. Toplamsal birleştirme + doyurulmuş BM25 eşiği anlamlı tutar.

**S: Neden vektör veritabanı kullanmadın?**
C: 31 parça için NumPy matris çarpımı ~10 ms. Gereksiz bağımlılık eklemedim; `Retriever.search` arayüzü arkasında,
büyük korpusta Qdrant/pgvector'e geçmek yalnızca o fonksiyonu değiştirmek.

**S: Deterministik yönlendirme LLM'in araç çağırmasını gereksiz kılmıyor mu?**
C: Hayır, tamamlayıcı. Açık niyetler (numaralı sipariş, ürün + fiyat) regex ile hızlı ve kesin; muğlak ifadelerde LLM'in kendi
araç çağırması yedek. Kazanç: bir LLM turu daha az (CPU'da saniyeler) ve küçük modelin unutkanlığına karşı güvence.

**S: Barge-in'de yankıyı nasıl çözüyorsun?**
C: Tarayıcının yankı gidermesi (getUserMedia `echoCancellation`), ajan konuşurken daha katı konuşma başlangıcı (350 ms) ve
gerekirse yarı çift yönlü mod (`BARGE_IN=false`). Kulaklıkla sorun olmaz.

**S: Dil tespiti nasıl çalışıyor, ya karışık dilde konuşursa?**
C: Whisper kodlayıcı çıktısından dil olasılıkları; yalnızca TR/EN arasından seçiliyor. İfade başına tek dil kabul ediliyor;
cevap o dilde. Karışık konuşmada baskın dil seçilir.

**S: Yeni bir veri kaynağı (ör. CRM API) nasıl eklenir?**
C: `@registry.tool(...)` ile bir fonksiyon yazıp `create_tool_registry`'de kaydetmek; istenirse bir `router`. Hata
yönetimi, zaman aşımı, LLM şeması otomatik.

**S: Ölçümlerinin zayıf yanları neler?**
C: Sesli test sentetik sesle (Türkçe Piper sesi robotik); eşik değerlendirme setiyle kalibre edildi; 26 soru küçük bir set;
sesli ölçümler 8'er turluk küçük örneklemler; değerlendirme setinin 26 soruluk hali yalnızca GPU'da çalıştırıldı (CPU
tablosu 24 soruluk ilk hal). README'de hepsi açıkça yazıyor.

**S: Bir sonraki adım ne olurdu?**
C: Akışlı STT (konuşurken kısmi çözümleme — ~1 sn kazanç), Whisper'ı da GPU'ya almak (whisper.cpp Vulkan; Windows için
hazır derlemesi yok), alan dışı soru sınıflandırıcısı, daha büyük değerlendirme seti, gerçek kullanıcı kayıtlarıyla test,
müşteri kimlik doğrulaması.

---

## 12. Projeyi genişletmek: alıştırmalar

Öğrendiklerinizi pekiştirmek için kolaydan zora:

1. **Yeni doküman:** `data/docs/` içine bir `.md` ekleyin, sunucuyu yeniden başlatın, sorun. (Arama önbelleği otomatik yenilenir.)
2. **Yeni araç:** `seed.sql`'e `invoices` tablosu ekleyip `get_invoice` aracı yazın; `test_tools.py`'ye test ekleyin.
3. **Ayar deneyi:** `VAD_MIN_SILENCE_MS=300` ve `800` ile konuşun; doğallık / gecikme farkını gözlemleyin.
4. **Model deneyi:** `powershell -File scripts/eval_matrix.ps1 -Configs "qwen2.5:3b|rag" -Port 8001` ile başka bir
   modeli değerlendirip §9.3 tablosuyla karşılaştırın; `ollama ps` ile modelin ne kadarının GPU'ya sığdığına bakın.
5. **Yeni dil (Almanca):** `LANGUAGES=["tr","en","de"]`, bir `de_DE` Piper sesi, `LANGUAGE_INSTRUCTIONS`, `TOOL_FILLERS`,
   `ERROR_REPLIES` girdileri.
6. **OpenAI uyumlu arka uç:** llama.cpp `llama-server` başlatıp `LLM_BACKEND=openai` ile bağlanın.
7. **İleri:** akışlı STT — VAD konuşma devam ederken her ~1 sn'de kısmi çözümleme yapıp tur bitince yalnızca son parçayı
   çözümleyin; `first_audio_ms` değişimini ölçün.
