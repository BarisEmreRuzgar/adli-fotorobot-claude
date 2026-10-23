# Gemini planı: uygulama ve test günlüğü

Başlangıç: 2026-10-18. Her adım Colab T4'te aynı tohumla eski sürüme karşı ölçülür;
kazanç yoksa geri alınır. Deney betikleri `deneme` dalında `deneyler/`, görselleri Colab'da `/content/ab_*.png`.

| # | Adım | Durum | Sonuç |
|---|---|---|---|
| 1 | Aşama 3: ControlNet inpaint + en az 256 px kırpma | ❌ geri alındı | ControlNet yeni ayrıntıyı bastırıyor (0/6 vs 5/6) |
| 1b | Aşama 3: kısa, ağırlıklı ayrıntı tarifi (ControlNet yerine) | ✅ tutuldu (`deneme`) | Ben: ağırlık 1.35 + fırça 9 → 5/5 doğal; yara izi çözülmedi (en iyi 1/5) |
| 3 | Aşama 1: prompt'u anlamsal bloklara ayırma (BREAK) | ✅ tutuldu (`deneme`) | Cinsiyet tutarlılığı ve uzun saç iyileşti; renk (beyaz bıyık/sakal) çözülmedi |
| 3b | Erkeklik vurgusu her erkek yüzde | ✅ tutuldu (`deneme`) | Siyah saç + beyaz sakal 4/4 erkek (önce 1/4 kadın), ince bıyıklı genç 4/4 erkek |
| 5 | YZ tahmini uyarı bandı, negatif prompt, manifest.sha256 | ✅ kod hazır (`deneme` dalı) | Birim testleri geçiyor |
| 4 | Gemini önbelleği (JSON şema zorlaması gerekmedi) | ✅ kod hazır (`deneme` dalı) | Aynı ifade → aynı analiz, kota harcanmaz |
| 2 | Aşama 2: bölgesel maskeli düzenleme (MediaPipe yerine InsightFace noktaları) | ✅ tutuldu (`deneme`) | Kimlik +4/+7 puan, değişmeyen her şey birebir korunuyor |
| 2b | Saç rengi: bölgesel değişim gücü 0.9 vs 1.0 | ✅ 0.9 kaldı | 1.0 uzun saçta yüzü bozuyor; 0.9 uzun saçta 3/3 açık renk, kısa saçta gri |
| 2c | Saç maskesi omuz/kıyafeti kapsamasın | ✅ tutuldu (`deneme`) | Uzun saçta tişört 3/3 korundu (önce değişiyordu), saç 3/3 açık sarı, kimlik aynı |
| 3c | Yeri belli ben: Aşama 1 prompt'u yerine yüz noktalarından otomatik Aşama 3 | ✅ eklendi | Yanak 0/4 → 4/4, kaş üstü 0/4 → 4/4, doğru tarafta |
| 1c | Aşama 3: 9 kanallı RV inpainting UNet'i | ❌ eklenmedi | Hiçbir şey çizmiyor: ben 0/5 (4 kanal 4/5), yara 0/10 |

## Ayrıntılar

### Adım 1: Aşama 3 küçük ben
Kaynak: vaka_20261018_041513 (genç erkek, Aşama 2 v1), maske ağız yanında ~24 px, tohum 20261018.
- Eski yöntem, dünkü tarif: maske içi fark 5.9/6.6/7.8 (dün ile birebir aynı → ölçüm güvenilir). Görselde ben yok.
- ControlNet + 256 px kutu, aynı tarif: 5.0/5.1/6.1. Görselde ben yok. → ControlNet tek başına çözmedi.
- Tarif deneyleri (ControlNet açık):
  - "prominent dark brown mole on the skin, RAW photo, close-up of human skin..." → ben yok
  - "a large dark brown mole, beauty mark on the cheek" (geniş fırça / CFG 11 dahil) → ben yok
  - **"(dark brown mole:1.6), (beauty spot:1.3), skin"**, yarıçap ~20 → 1/3 net ben, 2/3 soluk
  - "(large raised dark brown nevus:1.5) on the cheek" → 3/3 koyu iz ama lekemsi
- Sonuç: "close-up of human skin" gibi cilt tarifleri modeli düz cilde itiyor; ağırlıklı, kısa nesne tarifi gerekli.
- ControlNet açık/kapalı, aynı ağırlıklı tarif, 6 tohum: ControlNet'siz 5/6 iz, ControlNet'li 0/6 → ControlNet geri alındı.
- Ağırlık 1.6 ile (ControlNet'siz): ben r12 6/6 ama iri, yara gibi; yara izi 6/6 ama şeritli/dövme gibi yapay.
  Dünkü "çalışan" yara izi tarifi bu tohumda 0/6 → dünkü başarı şanslı tohummuş.

### Adım 1b: Aşama 3 ağırlık taraması (ControlNet'siz, 5 tohum)
- Ben: ağırlık 1.2 → 1-2/5; **1.35 + yarıçap 9 → 5/5 doğal ben**; 1.6 → iri, yara gibi. → 1.35, varsayılan fırça 9.
- Yara izi: her ağırlıkta (1.2/1.35/1.6) dövme, bant, şerit gibi yapay çizgiler. → düşük değişim gücüyle ayrı test.

### Adım 2: Bölgesel Aşama 2 (aynı kaynak, tohum 2026, 3 varyasyon; ArcFace kimlik)
| Komut | Genel (eski) | Bölgesel (yeni) | Görsel |
|---|---|---|---|
| sakalı temizle + saçları sarı yap | %63 / %61 / %60 | %57 / %70 / %70 | ikisinde de sakal gitti, saç sarı olmadı; bölgeselde gözlük/arka plan aynı |
| gözlüğü çıkar | %66 / %69 / %67 | **%70 / %76 / %78** | ikisinde de 3/3 kalktı; genelde yüz, ışık, tişört değişti, bölgeselde aynı |
| gözlük ekle | %75 / %76 / %76 | %81 / %57 / %84 | ikisinde 3/3 gözlük; bölgeselde 1 gözlük bozuk |
Düzeltme: gözlük çıkarılırken dokunulmayan küpe yüzünden kulaklar da maskeye giriyordu → yalnızca değişen aksesuarın bölgesi.

### Adım 3: Bloklu prompt (sabit özellik tablosu, Gemini yok, tohum 4242, 4 varyasyon)
| Vaka | Eski | Yeni (BREAK blokları) |
|---|---|---|
| Omuza kadar uzun saçlı genç erkek | 1/4 uzun saç | 2/4 |
| Kel, kara kaşlı, beyaz pala bıyıklı 60 yaş | bıyık 0/4 beyaz | bıyık 0/4 beyaz; 1 yüz kel değil (gerileme) |
| Siyah saç + beyaz sakal, 50 yaş | **3/4 kadın** çıktı | 1/4 kadın; sakal koyu |
Sonuç: tutuldu. Renk çelişkileri (beyaz bıyık/sakal) SD 1.5'te prompt'la çözülmüyor → Aşama 2/3'te bölgesel düzeltme gerekir.
Yan bulgu: sakal tarif edilse de kadın yüz çıkabiliyor → erkeklik vurgusu her erkek yüze genişletildi (3b).

### Adım 3b: Erkeklik vurgusu (tohum 4242, 4 varyasyon)
- Siyah saç + beyaz sakal, 50 yaş: 4/4 erkek, ikisinde gri sakal (önce 1/4 kadın, sakal koyu).
- İnce bıyıklı genç erkek: 4/4 erkek. Tutuldu.

### Adım 1b devamı: yara izi (5 tohum)
- Ağırlıklı tarifler (1.2/1.35/1.6): dövme, bant, şerit gibi yapay çizgiler.
- Negatife "tattoo, stripe, bandage" eklenince yapaylık gidiyor ama iz de neredeyse kayboluyor.
- En iyisi "pale raised scar tissue" 1.35, değişim 0.65: 1/5 gerçekçi. Yara izi SD 1.5 + 4 kanallı UNet ile güvenilir değil.

### Adım 2b: Saç rengi (sarı), bölgesel, tohum 2026, 3 varyasyon
| Yüz | Güç 0.9 | Güç 1.0 |
|---|---|---|
| Kısa, seyrek saçlı 50'li erkek | %86/%87/%84, saç griye döndü, sarı değil | %87/%84/%85, aynı; tişört siyahtan beyaza döndü |
| Omuza kadar saçlı genç erkek | %89/%87/%85, 3/3 açık kumral/sarı uç | %77/%76/%77, 2/3 yüz saç bölgesinde tekrarlanıp bozuldu |
Sonuç: 0.9 kaldı. Saç maskesi yüz ovalinin dışındaki her şeyi aldığı için kıyafet de değişiyordu → maske çenenin 0.3 yüz boyu altında kesildi (2c).

### Adım 1c: 9 kanallı inpainting UNet'i (Uminosachi/realisticVisionV51_v51VAE-inpainting), 5 tohum
| Deney | 4 kanal (mevcut) | 9 kanal |
|---|---|---|
| Ben, (dark brown mole, beauty spot:1.35), fırça 9 | 4/5 doğal ben | 0/5, cilt olduğu gibi |
| Yara, (pale raised scar tissue:1.35) | 5/5 dövme gibi yapay şerit | 0/5, düz cilt |
| Yara, "healed scar" (ağırlıksız) | 4/5 yapay iz | 0/5, düz cilt |
Sonuç: 9 kanallı model maskeli bölgeyi çevresine uydurmaya çok yatkın; küçük yeni ayrıntı üretmiyor. Ek ~1.7 GB indirmeye değmez.

### Adım 2c: Saç maskesinin alt sınırı
Maske çenenin 0.3 yüz boyu altında kesiliyor (resmin dışına taşarsa kesilmiyor; ilk sürüm bu durumda
PIL hatası verdi, deneyde yakalandı ve test eklendi). Aynı tohumla uzun saçlı yüz: %90/%87/%84
(önce %89/%87/%85), gri tişört 3/3 korundu, saç 3/3 açık sarı. Kısa saçlı yüzde yüz kadraja çok
yakın olduğu için kesim resmin dışında kalıyor, sonuç değişmedi.

### Adım 3c: Ben, Aşama 1'den otomatik Aşama 3'e (Gemini planı c'nin ikinci yarısı; tohum 4242, 4 varyasyon)
| Tarif | Prompt'ta (eski) | Yüz noktalarından Aşama 3 (yeni) |
|---|---|---|
| small dark mole on the left cheek | 0/4 görünür ben | 4/4, kişinin sol yanağında |
| small dark mole above the right eyebrow | 0/4 | 4/4, kişinin sağ kaşının üstünde (bazıları biraz yüksek → nokta 0.6'dan 0.45 göz arası mesafeye indi) |
Uygulama: `ben_konumlari` ozel alanından yeri ve tarafı belli benleri ayıklar (yanak/kaş üstü için taraf şart;
alın, çene, dudak üstü, burun tarafsız olabilir), Aşama 1 prompt'undan çıkarır; `ben_noktasi` noktayı 5 yüz
noktasından hesaplar, her varyasyona fırça 9 + ağırlık 1.35 ile çizilir. Kapatmak için `FOTOROBOT_AUTO_MOLE=0`.
Yara izi otomatiğe alınmadı: Aşama 3'te güvenilir değil (1b). Sınır: Aşama 2 tüm yüzü yeniden çizerse ben kaybolabilir.

# Otonom Ar-Ge (2026-10-21/22, dal `arge`)

Ölçüm: yerel RTX 3050 6 GB. `deneyler/benchmark_and_test.py` (temel), `deneyler/uctan_uca.py` (arayüzün kendi akışı,
kural tabanlı analiz). İfade uyumu Gemini ile denetlenir (`deneyler/vlm.py`: iddia doğru mu + gerçekçilik 1-5),
kimlik ArcFace. Senaryolar: siyah saç + beyaz sakal (55), kısa sarı saç (35), sol yanakta yara izi (40),
kel + beyaz pala bıyık (60); Aşama 2: temel yüz + 5 komut.

| Senaryo | Temel | Yeni | Yöntem |
|---|---|---|---|
| A1 siyah saç + beyaz sakal | 0/4 | 3/4 (r 3-4) | rötuş: SegFormer ayrıştırma + Lab boyama + inpaint 0.6 |
| A1 kısa sarı saç | 0/4 | 4/4 (r 2-4) | rötuş: saç boyama + inpaint 0.45 |
| A1 sol yanakta yara izi | 0/4 (turuncu leke) | 3/4 (r 4) | prompt'tan çıkar, yüz noktalarından prosedürel iz + inpaint 0.3 |
| A1 beyaz pala bıyık | 0/4 (ayrıca analizci rengi düşürüyordu) | 3/4 (r 2-4) | analizci düzeltmesi + yalnız bıyık maskesi |
| A2 saçları sarı yap | 0/3 | 3/3, kimlik 0.83 | bölgesel FaceID + rötuş |
| A2 yara izi ekle | 0/3 | 3/3, kimlik 0.99 | yüz yeniden üretilmez, iz çizilir |
| A2 beyaz sakal ekle | 0/3 (kirli sakal) | 3/3, kimlik 0.66-0.71 | FaceID'siz alt yüz çizimi (güç 0.95) |
| A2 15 yaş yaşlandır | 0/3 | 3/3, kimlik 0.71-0.73 (r 4) | FaceID'siz alın/göz/ağız kenarı çizimi (güç 0.55) |
| A2 burnu kemerli yap | 0/3 (analizci komutu anlamıyordu) | 0/3 | açık: önden vesikalıkta kemer görünmüyor |

Reddedilenler (ölçümle):
- FLUX.1-dev NF4 (Colab T4): görsel başına ~9 dk (T4'te bf16 yok), siyah saç + beyaz sakal 0/4. Ücretsiz T4'te uygulanamaz.
- Aşama 2'de değişen alana ağırlık / düşük kimlik gücü / yüksek değişim: burun ve yaşta 0/3 (FaceID geri çekiyor).
- Tam yüz FaceID'siz img2img ile yaşlandırma: 3/3 ama kimlik 0.37-0.43 (başka kişi).
- Sakal için "snow white": tıraş köpüğü; ağırlık 1.3 + "thick": örgülü/kostüm sakal (r 2).
- Rötuşta tam Lab uzaklığı (ΔE>15): model beyazı kır çizince yeniden boyuyordu → renk ailesi kontrolü (`renk_uyumlu`).

### Gerileme testi (2026-10-23, `deneyler/regresyon.py`, tohum 777, 2 görsel/senaryo)
| Senaryo | Rötuş kapalı | Rötuş açık |
|---|---|---|
| Kadın, uzun sarı saç | 1/2 | 1/2 (saç gerçekten sarıya döndü) |
| Esmer, kıvırcık, kirli sakal | 0/2 | 0/2 (rötuş ten/sakal tipine dokunmuyor; model kirli sakal yerine keçi sakalı çiziyor) |
| Kızıl kadın | 1/2 | 2/2 |
| Kır saç, gözlük, bıyık | 2/2 (saç aslında siyah, L*≈16) | 2/2 |
| Kaşta yara izi | 0/2 | 1/2 |
| **Toplam** | **4/10** | **6/10**, gerçekçilik 4-5 korunuyor |

Düzeltme: kır saç düz gümüşe boyanınca peruk gibi duruyordu → yalnız ortancadan açık teller griye çekiliyor (karışık kır, r4/4).
Açık: "kirli sakal" isteğinde keçi sakalı/bıyık çıkması; kaş yarasının bazen görünmemesi.

### 2026-10-23 devam
- **Kirli sakal**: "kirli sakal" isteğinde keçi sakalı/dolgun bıyık çıkıyordu → yalnız kirli sakalda keçi sakalı/dolgun bıyık negatifi + "görünür kısa, eşit kıl" ipucu (ağırlık 1.25). Esmer senaryosu 0/2 → 1/2, keçi sakalı 2/2 → 0/2, r4/4 (kalan başarısızlık kıvırcık saçın tutmaması).
- **Kaş yarası**: kâkül alnı örtünce izin üst yarısı saçta kalıp çizilmiyordu ve kaş içindeki parça kaş+cilt ortancasıyla boyandığı için kayboluyordu → yol saçta kalan kısım ≤%20 olana dek aşağı (kaş + göz kapağı) kaydırılıyor, iz rengi yalnız cilt piksellerinden alınıyor (kaşta ten renkli boşluk). Rötuşla 1/2 → 2/2.
- **Beyaz sakal "yapıştırılmış" görünümü** (iki Gemini modeli de r2-3, "digitally pasted"): (1) kıl maskesi gölgeli yanağı göz altına kadar kıl sayıyordu → her yanağın kendi parlaklığı + uyarlanır eşik (cilt ile sakalın en koyu %5'i arasındaki farkın 0.25'i); (2) inceltme kıvrık kostüm bıyığı uyduruyordu → ağırlıklı "handlebar/curled/fake beard/costume" negatifi (yalnız-bıyıkta kapalı: pala nane yeşiline kayıyordu); (3) inceltme bazı tohumlarda sakalı yeniden griye koyulaştırıyordu → renk ailesi tutana dek 3 tohum. Gerileme seti (tohum 777): beyaz sakal rötuşla 1/2 → 2/2 (r2/r4, kıvrık/leke yok), pala 1/2 (r3/r4, beyaz, doğal).
- **Aşama 2 sakal ekleme**: yerel sakal tohuma göre 6 adayın 4'ünde kahverengi/sarı çıkıyor, kimlik 0.49-0.69 oynuyordu (büyük/beyaz sakal kimliği doğal olarak düşürür). `Motor.yerel_ciz_sec`: 3 aday, önce sakal rengi ailesi tutan, sonra en yüksek ArcFace. "beyaz sakal ekle" 1-2/3 → 3/3, kimlik 0.60-0.66 → 0.66-0.67, r4/4/4. Yaşlandırmada denendi, en az yaşlananı seçtiği için (3/3 → 2/3) kapalı. Reddedilen: açık sakala "blonde/brown beard" negatifi (1/3, r1-2).
- Reddedilen (beyaz sakal ekleme, bıyığın sarımsı kalması): kaynağa göre değişen piksellerde ton kilidi (sarı kıl ile cilt farkı ΔE 6-9, ayrılamıyor; etki yok); önce koyu sakal çizip rötuşla beyaza boyama (kimlik 0.68-0.71 ama r2/2/2, grimsi boya görünümü).
- **Kemerli burun (teknoloji değişikliği: üretim → geometri)**: FaceID'li yol burnu değiştirmiyor, FaceID'siz yerel çizim kimliği 0.60'a düşürüyordu (0/3). `rotus.burun_kemerli`: burun ucu piksel bükmeyle aşağı uzatılır (cv2.remap), burun sırtına doygunluğu azaltılmış ışık çıkıntısı + altında gölge + yanlarda sırt gölgesi; yüz yeniden üretilmez. Aşama 2 "burnunu kemerli yap" **0/3 → 3/3, kimlik 0.80, r4/4/4**. Genelleme (7 farklı Aşama 1 yüzü): 5/7, kimlik 0.77-0.89, r3-5; kontrol: aynı kaynakların 4/5'i bükmeden "kemerli değil" → etki gerçek. Düşük güçlü inceltme iddiayı siliyordu, kullanılmadı.
- **Yaşlandırma kimliği**: FaceID tamamen kapalı yerel çizim yerine düşük güçlü FaceID'li bölgesel inpainting (kimlik gücü 0.45, aynı maske/güç). Deney: 0.15/0.3 kimliği düşürüyor, 0.45 → 0.73-0.75 yaşlı, 0.5 uçtan uca 0/3, 0.6+ yaşlanmayı siliyor. Uçtan uca "15 yaş yaşlandır" 3/3, kimlik 0.71-0.74 → **0.74-0.76**, r4/4/4. Reddedilen: yaşlandırma sonrası gözleri kaynaktan geri koyma (+0.01).
- **Beyaz sakal kimliği** (`deneyler/sakal_kimlik_analiz.py`): kaynaktan bölge bölge geri yapıştırma gösterdi ki sakal çizimi burnu da değiştiriyor; yalnız burun geri konunca kimlik 0.66-0.67 → 0.75 (göz/kaş +0, dudak +0-0.03; sakalı kaynağın üstüne tek başına yapıştırmak bile 0.72-0.80 → bu tavana yakın). Yerel çizimde `koru` (sakal: burun) bölgeleri kaynaktan geri konur. Uçtan uca "beyaz sakal ekle" (2 varyasyon) **2/2, kimlik 0.75/0.75, r4/4**.
- **Aday seçimi düzeltmesi**: `sakal_rengi_olc` yalnız cilde göre koyu pikselleri saydığından beyaz ve kahverengi adaylarda aynı rengi veriyordu (hepsi "uyumsuz" → seçim fiilen yalnız kimlikle, seyrek sakal seçilebiliyordu). Yeni ölçüm `rotus.sakal_bolge_rengi` (bıyık hariç sakal bölgesinin ortancası; `deneyler/biyik_renk.py`: beyaz 12-17, kahverengi/seyrek 27-33). Hedefe en yakından `ADAY_RENK_PAYI` (6) içindekilerden kimliği en yüksek seçilir.
- **Farklı yüzler** (`deneyler/sakal_genel.py`): esmer yüzde 6 adayın 6'sı koyu kirli sakal → ilk adayların hiçbiri `ADAY_RENK_ESIGI` (22) içinde değilse aynı sayıda ek aday, açık renklerde sert prompt (ağırlık 1.2 + koyu sakal negatifi). Ağırlık 1.35 kâğıt/pamuk maske üretti (r2); hedefe 8'den yakın (düz boya, r1-2) adaylar elenir. Genelleme **4/4 yüz**, kimlik 0.68-0.71 (esmer yüzde 0.53: rengi tutan tek aday alt yüzü çok değiştiriyor).
- Reddedilen: sakalda düşük güçlü FaceID (0.15/0.3/0.45): kimlik 0.73-0.82 ama sakal her tohumda koyu, beyaz tutmuyor (uu18 0/1); doku (yüksek frekans) ile düz boya ayrımı (kâğıt 1.69, doğal koyu 1.6-1.9, ayrılmıyor).
- **Yaşlandırmada saç çizgisi lekesi**: alın maskesi saç çizgisine değdiği için orada koyu kırmızı leke/kıvrım çıkıyordu (uu22/uu24, Gemini fark etmedi). Yaşlanma çiziminde saç kaynaktan geri konur (`YAS_KORU`): leke yok, kimlik 0.74-0.75 → 0.77-0.79 (uu23 ve uu24 aynı tohum). Yaş iddiası iki koşuda da 0/2 (aynı görsel başka koşuda OK → Gemini yaş kararı oynak).
- Reddedilen: sakal çizim gücü 0.85 (esmer yüzde seyrek, kimlik değişmedi); sakal çiziminde yalnız belirgin değişen pikselleri tutma (kimlik 0.76-0.79 ama "yapıştırılmış", r2, 0/3); yaşlandırmada şakak kırlaştırma (keskin gri şeritler, r2-3, yaş kararına etkisi belirsiz).
- **Aşama 2 tam set (uu22, 1 varyasyon)**: sarı saç 0.82, beyaz sakal 0.75, kemerli burun 0.79, yaşlandırma 0.75, yanak yarası 0.99 — **5/5 OK, hepsi r4, kimlik ≥ 0.75**.
- **Tam sakalda dudaklar da korunur** (`SAKAL_DUDAK_KORU`; yalnız bıyıkta değil): esmer yüzde bölge analizi kaybı dudakta gösterdi (geri koyunca 0.53 → 0.63). Genelleme (sg10): esmer 0.53 → 0.63, kel 0.71 → 0.72, sarışın 0.68 → 0.72, 3/3. Ana senaryo (uu25): **2/2, kimlik 0.75 → 0.79**, r3/4.
- **Aşama 2 genelleme (farklı yüzler, `sakal_genel.py` + `KOMUT`)**: sarı saç 2/2 (kimlik 0.77-0.81, r4); yanak yarası 2/2 (kimlik 0.99; ilk yargı 0/2 ama yara doğru tarafta, aynı görsellerin yeniden yargısı 5/6 OK → Gemini oynaklığı); yaşlandırma 1/1 (0.78, r4); beyaz sakal 3/3 (0.63-0.72).
- Reddedilen (bıyık sarımsı tonu): bıyık bölgesinde yanaktan açık kıl yoksa (`biyik_acik_kil_rengi`) adaya +10 renk cezası → bıyık beyaz, kimlik 0.80-0.81 ama seçilen sakallar parlak/kostüm gibi, r2/3 (öncesi r3-4/4). Sarımsı bıyık Gemini'de geçiyor; gerçekçilik öncelikli.
- **Kıvırcık saç** (gerileme esmer_kivircik 1/2: bir tohumda düz saç): saç tarifinde "curl" varsa 1.2 ağırlık + "tight natural curls, curly hair texture" ipucu, negatifte "straight hair, slicked back hair". Esmer senaryosu **1/2 → 2/2** (rötuş kapalı ve açık), r3-4/4.
- **Uzun saç** (kadin_sari 1/2: saç omuz hizasında): saç ipuçları tabloya (`SAC_IPUCLARI`) taşındı; "long" → "long hair falling past the shoulders to the chest", negatif "short hair, bob cut, shoulder length hair" ("shoulder length" tarifinde uygulanmaz). kadin_sari **1/2 → 2/2**, r4/4.

### Gerileme testi (2026-10-24, tohum 777, 7 senaryo × 2 görsel)
| Senaryo | Rötuş kapalı | Rötuş açık |
|---|---|---|
| Kadın, uzun sarı saç | 2/2 | 2/2 |
| Esmer, kıvırcık, kirli sakal | 2/2 | 2/2 |
| Kızıl kadın | 1/2 | 2/2 |
| Kır saç, gözlük, bıyık | 2/2 | 2/2 |
| Siyah saç + beyaz sakal | 0/2 | 1/2 (görseller reg11 ile aynı, orada 2/2 → Gemini oynaklığı) |
| Kel, beyaz pala bıyık | 0/2 | 2/2 |
| Kaşta yara | 0/2 | 2/2 |
| **Toplam** | **7/14** | **13/14**, gerçekçilik 3-5 |
(2026-10-23: 4/10 → 6/10, 5 senaryo.)
- **Aşama 1 uçtan uca (uu27, 2 görsel)**: beyaz sakal 1/2, sarı kısa 2/2, yara 0/2, pala 2/2. Yara görselleri uu5 ile birebir aynı (orada 4/4) → Gemini oynaklığı; beyaz sakal/pala görselleri uu5'ten doğal (kıvrık kostüm bıyığı yok). Kalan kusur: beyaz sakal boyamasında seyrek koyu yanak kılları düz gri yama/şerit oluyor (bir görselde r2).
- Reddedilen (o gri yama için, `deneyler/ab_sakal_rotus.py`, `ab_sakal_ciz.py`): kıl tespitine yerel doku şartı (yamayı azaltıyor, gür sakalda siyah delikler açıyor); sakal bölgesinin yan üst sınırını aşağı çekmek (yama sınırda değil, etki yok); seyrek kılı boyamama/yoğunluk şartı (kenarlar kararıyor, r2); boyamak yerine Aşama 2 yerel çizimiyle sakalı yeniden çizmek (korunan dudak eski koyu bıyığı geri getiriyor, yeni bıyık kızıl; 0/2).

### Güncel durum (2026-10-24, çoğunluk oylamalı denetim `OYLAMA=3`)
- **Aşama 1** (uu28, 2 görsel/senaryo): beyaz sakal 2/2, sarı kısa 2/2, yara 2/2, pala 2/2 → **8/8**; gerçekçilik 4,2 / 3,4 / 4,4 / 2,3.
- **Aşama 2** (uu29): sarı saç 0.82, beyaz sakal 0.79, kemerli burun 0.79, yaşlandırma 0.79, yanak yarası 0.99 → **5/5, kimlik ≥ 0.79, r4/4/4/4/4**.
