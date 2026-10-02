#!/bin/sh
set -e
cd /opt/flybrain

case "$1" in
  gui)
    # Bizim arayüzümüz: beyin şeması + koşul paneli
    exec python /opt/app/server.py
    ;;
  eski)
    # Kaynak projenin hazır (Japonca) sayfası
    echo "  Eski arayüz -> http://localhost:8000"
    exec python -m http.server 8000 --directory docs
    ;;
  test)
    # Kurulum testi: şeker uyaranı -> MN9 (hortum uzatma motor nöronu) ateşlemeli
    exec python sim.py --stim sugar --rate 100 --trials 1 --name kurulum_testi
    ;;
  *)
    # Diğer her şey olduğu gibi çalıştırılır, örn: python sim.py --stim bitter
    exec "$@"
    ;;
esac
