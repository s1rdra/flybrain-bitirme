#!/bin/sh
set -e
cd /opt/flybrain

case "$1" in
  gui)
    # Şimdilik kaynak projenin hazır arayüzü; kendi arayüzümüz app/ altında gelecek
    echo ""
    echo "  Arayüz hazır  ->  http://localhost:8000"
    echo "  Robot sayfası ->  http://localhost:8000/robot.html"
    echo ""
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
