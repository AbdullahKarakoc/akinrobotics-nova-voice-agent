from app.agent.speech_text import (
    SentenceChunker,
    clean_for_speech,
    detect_text_language,
    ungrounded_numbers,
)
from app.stt.domain_terms import fix_domain_terms
from app.stt.faster_whisper_stt import trim_repetitions


def stream(chunker: SentenceChunker, text: str, step: int = 3) -> list[str]:
    out = []
    for i in range(0, len(text), step):
        out += chunker.push(text[i : i + step])
    return out + chunker.flush()


def test_chunker_emits_sentences_as_soon_as_they_end():
    chunker = SentenceChunker()
    assert chunker.push("Rehber R2'nin pil ömrü 8 saattir. Şarj") == ["Rehber R2'nin pil ömrü 8 saattir."]
    assert chunker.flush() == ["Şarj"]


def test_first_piece_may_end_at_a_comma_to_start_speaking_sooner():
    text = "Bakım her 6 ayda bir yapılmalıdır, ilk yıl ücretsizdir, sonrası sözleşmeye dahildir. Başka bir şey?"
    pieces = stream(SentenceChunker(), text)
    assert pieces[0] == "Bakım her 6 ayda bir yapılmalıdır,"
    assert pieces[1] == "ilk yıl ücretsizdir, sonrası sözleşmeye dahildir."  # later pieces: whole sentences
    assert stream(SentenceChunker(), "Evet, kiralanabilir.") == ["Evet, kiralanabilir."]  # too short to split


def test_chunker_keeps_decimals_and_abbreviations_together():
    pieces = stream(SentenceChunker(), "Temizlik C3 saatte 1.500 metrekare temizler. Dr. Ayşe onayladı! Tamam mı?")
    assert pieces == ["Temizlik C3 saatte 1.500 metrekare temizler.", "Dr. Ayşe onayladı!", "Tamam mı?"]


def test_chunker_splits_long_run_on_sentences_at_a_comma():
    text = "Bu cümle çok uzun, " * 12 + "bitti."
    pieces = stream(SentenceChunker(soft_max_chars=80), text)
    assert len(pieces) > 1
    assert all(len(p) <= 120 for p in pieces[:-1])
    assert " ".join(pieces).replace("  ", " ") == text.strip()


def test_clean_for_speech_removes_markup_and_verbalises_units():
    assert clean_for_speech("**Sıcaklık** 18°C, nem %60 [1]", "tr") == "Sıcaklık 18 derece, nem yüzde 60"
    assert clean_for_speech("- Humidity 60% 😀 see https://x.io", "en") == "Humidity 60 percent see"


def test_detect_text_language():
    assert detect_text_language("Rehber R2'nin pil ömrü ne kadar?") == "tr"
    assert detect_text_language("What is the battery life of Taşıyıcı T5?") == "en"
    assert detect_text_language("Your order 250 for the Taşıyıcı T5 is in production.") == "en"
    assert detect_text_language("Taşıyıcı T5'in fiyatı 890.000 TL'dir.") == "tr"
    assert detect_text_language("Merhaba") == "tr"
    assert detect_text_language("hello there", default="tr") == "en"


def test_ungrounded_numbers_flags_only_invented_figures():
    evidence = "Rehber R2 pil ömrü 8 saat, fiyat 1450000 TL, boy 1,65 metre"
    assert ungrounded_numbers("Pil 8 saat, fiyatı 1.450.000 TL, boyu 1.65 m.", evidence) == []
    assert ungrounded_numbers("Pil ömrü 12 saattir.", evidence) == ["12"]


def test_domain_terms_are_normalised_without_touching_normal_words():
    assert fix_domain_terms("Rehber ve iki robotunun pil ömrü") == "Rehber R2 robotunun pil ömrü"
    assert fix_domain_terms("rehber ar 2 kaç kilo") == "Rehber R2 kaç kilo"
    assert fix_domain_terms("Taşıyıcı te beş fiyatı") == "Taşıyıcı T5 fiyatı"
    assert fix_domain_terms("temizlik C 3 robotu") == "Temizlik C3 robotu"
    assert fix_domain_terms("eğitim bir hafta sürer") == "eğitim bir hafta sürer"
    assert fix_domain_terms("rehberlik hizmeti") == "rehberlik hizmeti"
    # misrecognitions seen in real use
    assert fix_domain_terms("Zehbe Z2'nin Piyomru ne kadar?") == "Rehber R2'nin pil ömrü ne kadar?"
    assert fix_domain_terms("zebra 2 tane") == "zebra 2 tane"
    assert fix_domain_terms("Vehbar ve iki robotunun") == "Rehber R2 robotunun"
    assert fix_domain_terms("Heddar ve iki robotunun") == "Rehber R2 robotunun"
    assert fix_domain_terms("hedef 2 yıl") == "hedef 2 yıl"


def test_trim_repetitions():
    assert trim_repetitions("Merhaba nasılsın? Merhaba nasılsın? Merhaba nasılsın?") == "Merhaba nasılsın?"
    assert trim_repetitions("Hello Hello Hello Hello") == "Hello"
    assert trim_repetitions("Garanti iki yıldır.") == "Garanti iki yıldır."
    # real logs: a long phrase looped twice plus a cut-off tail
    looped = "Merhaba, sesim geliyor mu? Merhaba, sesim geliyor mu? Merhaba, ses"
    assert trim_repetitions(looped) == "Merhaba, sesim geliyor mu?"
    assert trim_repetitions("Ankara'da havanası Ankara'da havanası Ankara'da") == "Ankara'da havanası"
    assert trim_repetitions("Rehber R2 mi yoksa Taşıyıcı T5 mi?") == "Rehber R2 mi yoksa Taşıyıcı T5 mi?"
