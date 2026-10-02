"""Sanal sinek beyni — web arayüzü sunucusu (yalnızca Python standart kütüphanesi).

    python server.py            ->  http://localhost:8000
"""

import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import beyin as B

HERE = Path(__file__).parent
STATIC = HERE / 'static'
SONUC_DIR = Path(os.environ.get('SONUC_DIR', B.FLYBRAIN_DIR / 'results')) / 'arayuz'
PORT = int(os.environ.get('PORT', 8000))

durum = {'hazir': False, 'mesaj': 'Başlatılıyor ...'}
beyin = None
harita = None
isler = {}            # iş kimliği -> {durum, ilerleme, sonuc, hata, kosul}
kilit = threading.Lock()   # aynı anda tek simülasyon


def log(msg):
    durum['mesaj'] = msg
    print(msg, flush=True)


def yukle():
    global beyin, harita
    try:
        beyin = B.Beyin(log=log)
        harita = beyin.beyin_haritasi()
        durum['hazir'] = True
        log('Arayüz hazır  ->  http://localhost:%d' % PORT)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        durum['mesaj'] = f'Yükleme hatası: {e}'


def is_calistir(is_id, kosul):
    j = isler[is_id]
    with kilit:
        j['durum'] = 'calisiyor'
        try:
            def ilerleme(p):
                j['ilerleme'] = round(p, 3)
            sonuc = beyin.calistir(
                duyular=kosul.get('duyular', {}),
                nt_olcek=kosul.get('ilac', {}),
                lezyon=kosul.get('lezyon', []),
                sure=min(max(float(kosul.get('sure', 0.5)), 0.05), 5.0),
                canli=kosul.get('canli') or None,
                seed=kosul.get('seed'),
                ilerleme=ilerleme,
            )
            sonuc['kosul'] = kosul
            sonuc['id'] = is_id
            sonuc['zaman_damgasi'] = time.strftime('%Y-%m-%d %H:%M:%S')
            j['sonuc'] = sonuc
            j['durum'] = 'bitti'
            try:
                SONUC_DIR.mkdir(parents=True, exist_ok=True)
                ozet = {k: v for k, v in sonuc.items() if k not in ('kare_noron_b64', 'kare_ofset')}
                (SONUC_DIR / f'{time.strftime("%Y%m%d-%H%M%S")}_{is_id}.json').write_text(
                    json.dumps(ozet, ensure_ascii=False, indent=1), encoding='utf-8')
            except OSError as e:
                print('kaydedilemedi:', e)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            j['durum'] = 'hata'
            j['hata'] = str(e)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # sessiz
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split('?')[0]
        if p == '/api/durum':
            return self._send(200, durum)
        if p == '/api/meta':
            if not durum['hazir']:
                return self._send(503, durum)
            return self._send(200, {
                'duyular': B.DUYULAR, 'nt': B.NT_LISTESI, 'ilac_hazir': B.ILAC_HAZIR, 'madde_kat': B.MADDE_KATEGORI,
                'davranislar': B.DAVRANISLAR, 'bolge_ad': [B.SUPER_AD[s] for s in B.SUPER_SIRA],
                'n': beyin.n, 'tipler': beyin.tipler(),
            })
        if p == '/api/harita.bin':
            if not durum['hazir']:
                return self._send(503, durum)
            return self._send(200, harita, 'application/octet-stream')
        if p.startswith('/api/is/'):
            j = isler.get(p.rsplit('/', 1)[1])
            if not j:
                return self._send(404, {'hata': 'iş yok'})
            out = {k: j.get(k) for k in ('durum', 'ilerleme', 'hata')}
            if j['durum'] == 'bitti':
                out['sonuc'] = j['sonuc']
            return self._send(200, out)

        # statik dosyalar
        name = 'index.html' if p in ('/', '') else p.lstrip('/')
        f = (STATIC / name).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._send(404, 'bulunamadı', 'text/plain; charset=utf-8')
        ctype = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css'}.get(
            f.suffix, 'application/octet-stream') + '; charset=utf-8'
        return self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if self.path != '/api/calistir':
            return self._send(404, {'hata': 'yok'})
        if not durum['hazir']:
            return self._send(503, durum)
        n = int(self.headers.get('Content-Length', 0))
        try:
            kosul = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self._send(400, {'hata': 'geçersiz JSON'})
        for t in kosul.get('lezyon', []):
            try:
                if not len(beyin.idx([t])):
                    return self._send(400, {'hata': f'"{t}" için nöron bulunamadı'})
            except ValueError as e:
                return self._send(400, {'hata': str(e)})
        is_id = uuid.uuid4().hex[:8]
        isler[is_id] = {'durum': 'sirada', 'ilerleme': 0.0, 'kosul': kosul}
        threading.Thread(target=is_calistir, args=(is_id, kosul), daemon=True).start()
        return self._send(200, {'id': is_id})


if __name__ == '__main__':
    threading.Thread(target=yukle, daemon=True).start()
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Sunucu açıldı: http://localhost:{PORT}  (beyin yükleniyor, sayfa bekleyecek)', flush=True)
    srv.serve_forever()
