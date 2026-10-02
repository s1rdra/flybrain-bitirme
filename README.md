# Sanal Meyve Sineği Beyni — Bitirme Projesi

Meyve sineğinin tam beyin haritası (FlyWire connectome, ~127 bin aktif nöron) üzerinde
spiking (LIF) simülasyonu. Kaynak proje: [qp-hiro/flybrain](https://github.com/qp-hiro/flybrain)
— Shiu ve ark. 2024 (Nature) modelinin saf NumPy hali (orijinalle r = 0.94 uyumlu).

Her şey Docker içinde çalışır; **Mac (M çip) ve Windows'ta aynı komutlar.**

## 1. Kurulum (bir kere)

1. **Docker Desktop** kur ve aç: https://www.docker.com/products/docker-desktop/
   - Windows: kurulumda **WSL 2** seçeneği işaretli olsun.
   - Ayarlar → Resources → Memory: en az **4 GB** ver (tam beyin simülasyonu RAM ister).
2. Bu klasörde terminal aç (Windows'ta PowerShell) ve imajı oluştur:

   ```
   docker compose build
   ```

   İlk seferde ~5–10 dk sürer; connectome verisi (~120 MB) imajın içine iner.

## 2. Kurulum testi

```
docker compose run --rm flybrain test
```

Beklenen çıktı: `MN9 rate: ~80 Hz` → sanal sinek şekeri tattı ve hortumunu uzatmaya karar verdi.

## 3. Arayüz

```
docker compose up
```

Tarayıcıdan aç: **http://localhost:8000** (durdurmak için `Ctrl+C`).
Şimdilik kaynak projenin hazır arayüzü geliyor; kendi beyin şemalı arayüzümüz `app/` altında yazılacak.

## 4. Deney komutları

Her komut `docker compose run --rm flybrain ...` ile başlar. Çıktılar `results/` klasörüne düşer.

| Deney | Komut |
|---|---|
| Şeker → beslenme | `python sim.py --stim sugar --rate 100 --trials 3 --name seker` |
| Şeker + acı (karar) | `python sim.py --stim sugar --rate 100 --stim2 bitter --rate2 100 --trials 3 --name seker_aci` |
| Nöron susturma (lezyon) | `python sim.py --stim sugar --rate 100 --silence type:CB0248 --trials 3 --name lezyon` |
| Işık | `python sim.py --stim type:R1-6 --rate 30 --name isik` |
| Feromon (koku) | `python sim.py --stim type:ORN_DA1 --rate 50 --name feromon` |
| Kullanılabilir gruplar | `python groups.py --list class subclass` |
| Konteynerin içine gir | `bash` |

## Klasör yapısı

```
Dockerfile           ortam tanımı (kaynak proje sabit commit'e kilitli)
docker-compose.yml   port + klasör bağlantıları
docker/entrypoint.sh gui / test / serbest komut
requirements.txt     bizim ek Python paketlerimiz
app/                 bizim kodumuz (arayüz, deneyler) — konteynerde /opt/app
results/             simülasyon çıktıları (git'e girmez)
```

## Notlar

- Tam beyin gerçek zamanın ~1/20 hızında çalışır (1 sn simülasyon ≈ 20 sn). Canlı arayüz için
  kaynak projedeki ~4.000 nöronluk gerçek zamanlı alt devre kullanılabilir.
- Docker içinde GPU yok (Mac'te Metal konteynere geçmiyor); model CPU'da çalışacak şekilde yazılmış.
- Kaynaklar: [Shiu ve ark. 2024](https://www.nature.com/articles/s41586-024-07763-9) ·
  [FlyWire](https://flywire.ai/) · [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
