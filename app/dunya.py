"""Açık dünya: sineğin beyni ile çevresi arasında kapalı döngü.

Her 20 ms'de bir:
  1. beyindeki davranış nöronlarının son 100 ms ateşlemesinden hareket hesaplanır
     (yürüme, dönme, kaçış/uçuş, yeme),
  2. sinek dünyada hareket ettirilir (ağaçlara çarpma, arena sınırı),
  3. sineğin yeni konumuna göre duyu nöronlarına giden uyarı yeniden hesaplanır:
     koku (sol/sağ anten ayrı), tat (temas), ışık (gölge), sıcaklık, nem,
     yaklaşan tehdit (kuş gölgesi -> LPLC2/LC4 görsel nöronları).

Birimler: mm ve saniye. Dünya merkezi (0, 0); yön 0 = +y (kuzey), saat yönünde artar.

Varsayımlar (raporda yazılmalı):
  * Koku: kaynaktan üstel azalan yoğunluk (λ = 14 mm), rüzgâr/türbülans yok.
  * Antenler başın 1,5 mm önünde, 1,2 mm yanlarda kabul edildi (gerçekte daha yakın;
    sol/sağ farkı görünür olsun diye büyütüldü).
  * 'Temel yürüme': beyin bir şey söylemese de sinek yavaşça ileri yürür. Bu, beyinde
    olmayan omurilik (VNC) yürüme ritmini temsil eden bir varsayımdır; kapatılabilir.
"""

import math

import numpy as np

ARENA = 70.0          # yarıçap [mm] (kare arena: -70..70)
KONTROL_MS = 20       # döngü adımı [ms]
PENCERE = 5           # hız hesabı için son 5 adım = 100 ms
LAMBDA = 14.0         # koku yayılma uzunluğu [mm]
ANT_ON, ANT_YAN = 1.5, 1.2

V_YURU = 16.0         # en hızlı yürüme [mm/s]
V_TEMEL = 6.0         # temel yürüme [mm/s]
V_UCUS = 45.0         # uçuş hızı [mm/s]
OMEGA = 3.0           # en hızlı dönüş [rad/s]

# Dünya duyuları -> nöron grupları. 'yan': sol/sağ ayrı uyarılabilir mi
KANALLAR = {
    'meyve_koku': {'ad': 'Meyve kokusu (sirke ORN)', 'yan': True,
                   'spec': ['type:ORN_DM1', 'type:ORN_DM4', 'type:ORN_VA2', 'type:ORN_DP1m']},
    'cicek_koku': {'ad': 'Çiçek kokusu (ester ORN, yaklaşık)', 'yan': True,
                   'spec': ['type:ORN_DM2', 'type:ORN_DL5', 'type:ORN_VA3']},
    # Not: önceki sürümde DA1 (Or67d, cVA) kullanılıyordu; cVA aslında ERKEK feromonudur ve
    # erkekte kurlaşmayı bastırır. Dişi/sinek kokusunu algılayan Or47b (VA1v) daha doğru.
    'feromon':    {'ad': 'Dişi kokusu (Or47b, VA1v ORN)', 'yan': True, 'spec': ['type:ORN_VA1v']},
    # Kurlaşmada erkek dişiyi gözüyle takip eder; LC10a görsel nöronları bu takipte rol oynar
    # (Ribeiro ve ark. 2018). Dişi görüş alanının hangi tarafındaysa o gözün LC10a'sı uyarılır.
    'disi_gorme': {'ad': 'Dişiyi görme (LC10a)', 'yan': True, 'spec': ['type:LC10a']},
    'tat':        {'ad': 'Şeker tadı (temas)', 'yan': False, 'spec': ['sugar']},
    'isik':       {'ad': 'Işık (R1-6)', 'yan': True, 'spec': ['type:R1-6']},
    'sicak':      {'ad': 'Sıcaklık', 'yan': False, 'spec': ['subclass:heating']},
    'nem':        {'ad': 'Nem', 'yan': False, 'spec': ['subclass:humid', 'subclass:moist']},
    'tehdit':     {'ad': 'Yaklaşan gölge (LPLC2/LC4)', 'yan': True, 'spec': ['type:LPLC2', 'type:LC4']},
}


def harita_olustur(seed=1, tip='bahce'):
    """Rastgele ama tekrarlanabilir bir bahçe ya da kur odası."""
    rng = np.random.default_rng(seed)
    if tip == 'kur':   # laboratuvardaki kur odası gibi: küçük arena, sadece dişi
        a = rng.uniform(0, 2 * math.pi)
        return {'arena': 22.0, 'tip': 'kur', 'nesneler': [],
                'disi': {'x': float(10 * math.sin(a)), 'y': float(10 * math.cos(a)), 'yon': float(rng.uniform(0, 2 * math.pi))},
                'sinek': {'x': 0.0, 'y': 0.0, 'yon': float(rng.uniform(0, 2 * math.pi))}}
    nes = []

    def bos_yer(r, min_merkez=12.0):
        for _ in range(400):
            x, y = rng.uniform(-ARENA + r + 4, ARENA - r - 4, 2)
            if math.hypot(x, y) < min_merkez + r:
                continue
            if all(math.hypot(x - o['x'], y - o['y']) > r + o['r'] + 6 for o in nes):
                return float(x), float(y)
        return float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))

    for _ in range(4):
        x, y = bos_yer(16)
        nes.append({'tip': 'agac', 'x': x, 'y': y, 'r': 3.2, 'tac': float(rng.uniform(13, 18)),
                    'boy': float(rng.uniform(30, 42))})
    for tur in ['elma', 'muz', 'elma']:
        x, y = bos_yer(5)
        nes.append({'tip': 'meyve', 'tur': tur, 'x': x, 'y': y, 'r': 4.0 if tur == 'elma' else 3.0})
    renkler = ['pembe', 'mor', 'beyaz', 'sari']
    for i in range(5):
        x, y = bos_yer(3)
        nes.append({'tip': 'cicek', 'renk': renkler[i % 4], 'x': x, 'y': y, 'r': 2.5,
                    'boy': float(rng.uniform(7, 12))})
    for _ in range(2):
        x, y = bos_yer(4)
        nes.append({'tip': 'su', 'x': x, 'y': y, 'r': 3.5})
    for _ in range(2):
        x, y = bos_yer(5)
        nes.append({'tip': 'tas', 'x': x, 'y': y, 'r': 4.5})
    # ilk meyveyi sineğe yakın koy ki koku hemen alınsın
    for o in nes:
        if o['tip'] == 'meyve':
            a = rng.uniform(0, 2 * math.pi)
            o['x'], o['y'] = float(18 * math.sin(a)), float(18 * math.cos(a))
            break
    dx, dy = bos_yer(2, 20)
    return {'arena': ARENA, 'tip': 'bahce', 'nesneler': nes,
            'disi': {'x': dx, 'y': dy, 'yon': float(rng.uniform(0, 2 * math.pi))},
            'sinek': {'x': 0.0, 'y': 0.0, 'yon': float(rng.uniform(0, 2 * math.pi))}}


class Dunya:
    def __init__(self, beyin, harita, sure, canli=None, taban=None, temel_yurume=True,
                 kus=True, seed=1):
        self.b = beyin
        self.h = harita
        self.nes = harita['nesneler']
        self.x, self.y, self.th = harita['sinek']['x'], harita['sinek']['y'], harita['sinek']['yon']
        self.fx, self.fy, self.fth = harita['disi']['x'], harita['disi']['y'], harita['disi']['yon']
        self.A = float(harita.get('arena', ARENA))
        self.alt = 0.0
        self.kalkis = 0.0
        self.disi_hiz = 1.5 if harita.get('tip') == 'kur' else 3.0
        self.temel = temel_yurume
        self.canli = canli
        self.taban = taban or {}
        self.rng = np.random.default_rng(seed + 1000)
        self.dt = KONTROL_MS / 1000.0
        # davranış nöronları -> indeks
        self.dav = [d['id'] for d in beyin.DAVRANISLAR]
        self.dav_n = []
        self.harita_noron = np.full(beyin.n, -1, dtype=np.int16)
        for k, d in enumerate(beyin.DAVRANISLAR):
            ix = beyin.idx(d['spec'])
            self.harita_noron[ix] = k
            self.dav_n.append(max(1, len(ix)))
        self.sayac = np.zeros((PENCERE, len(self.dav)))
        self.adim = 0
        # duyu kanalları: (sol, sağ) indeksleri
        self.kanal = {}
        for k, c in KANALLAR.items():
            if c['yan']:
                L = beyin.idx([s + '@L' if s.startswith('type:') or s.startswith('subclass:') else s for s in c['spec']])
                R = beyin.idx([s + '@R' if s.startswith('type:') or s.startswith('subclass:') else s for s in c['spec']])
                self.kanal[k] = (L, R)
            else:
                ix = beyin.idx(c['spec'])
                self.kanal[k] = (ix, None)
        # kuş gölgesi olayı
        self.kus = None
        if kus:
            t0 = float(self.rng.uniform(0.35, 0.6)) * sure
            self.kus = {'t0': t0, 'sure': 0.45, 'aci': float(self.rng.uniform(0, 2 * math.pi))}
        self.kayit = []
        self.temas = None

    # ---------------------------------------------------------- yardımcılar
    def _koku(self, px, py, tip):
        c = 0.0
        for o in self.nes:
            if o['tip'] == tip:
                d = max(0.0, math.hypot(px - o['x'], py - o['y']) - o['r'])
                c += math.exp(-d / LAMBDA)
        return c

    def _disi_koku(self, px, py):
        d = math.hypot(px - self.fx, py - self.fy)
        return math.exp(-d / (LAMBDA * 0.7))

    @staticmethod
    def _hz(c, mx=110.0, k=0.35):
        return mx * c / (c + k)

    # ---------------------------------------------------------- döngü adımı
    def adim_at(self, t_sn, yeni_spike):
        """yeni_spike: son adımda ateşleyen nöron indeksleri. Dönüş: (stim_idx, stim_p)."""
        # 1) davranış hızları
        k = self.harita_noron[yeni_spike]
        k = k[k >= 0]
        row = np.bincount(k, minlength=len(self.dav)).astype(float)
        self.sayac[self.adim % PENCERE] = row
        self.adim += 1
        hz = self.sayac.sum(0) / np.array(self.dav_n) / (PENCERE * self.dt)
        H = {d: max(0.0, hz[i] - (self.taban.get(d, 0.0) if self.canli else 0.0))
             for i, d in enumerate(self.dav)}

        # 2) hareket
        bw, bt = (20.0, 40.0) if self.canli else (60.0, 60.0)
        cl = lambda q: max(-1.0, min(1.0, q))
        if H['kacis'] > 30:
            self.kalkis = 1.0
        self.kalkis *= math.exp(-self.dt / 0.6)
        ucuyor = H['ucus'] > 4 or self.kalkis > 0.25
        self.alt += ((1.0 if ucuyor else 0.0) - self.alt) * min(1.0, self.dt * (6 if ucuyor else 3))
        yiyor = self.temas is not None and self.temas[0] in ('meyve', 'cicek') and H['yeme'] > 20
        yeme_k = 0.0 if yiyor else 1.0
        v = cl((H['ileri'] - H['geri']) / bw) * V_YURU
        if self.temel:
            v += V_TEMEL * (1 - self.alt)
        v = v * yeme_k * (1 - self.alt) + V_UCUS * self.alt
        om = cl((H['sag'] - H['sol']) / bt) * OMEGA
        self.th += om * self.dt
        nx = self.x + math.sin(self.th) * v * self.dt
        ny = self.y + math.cos(self.th) * v * self.dt
        # arena sınırı: yansı
        if abs(nx) > self.A - 2:
            self.th = -self.th; nx = max(-self.A + 2, min(self.A - 2, nx))
        if abs(ny) > self.A - 2:
            self.th = math.pi - self.th; ny = max(-self.A + 2, min(self.A - 2, ny))
        # ağaç gövdesine çarpma (uçarken taç altından geçer)
        for o in self.nes:
            if o['tip'] in ('agac', 'tas') and self.alt < 0.5:
                d = math.hypot(nx - o['x'], ny - o['y'])
                if d < o['r'] + 1.2:
                    a = math.atan2(nx - o['x'], ny - o['y'])
                    nx = o['x'] + math.sin(a) * (o['r'] + 1.2)
                    ny = o['y'] + math.cos(a) * (o['r'] + 1.2)
        self.x, self.y = nx, ny

        # dişi: yavaş rastgele yürüyüş
        self.fth += self.rng.normal(0, 0.6) * self.dt * 3
        self.fx += math.sin(self.fth) * self.disi_hiz * self.dt
        self.fy += math.cos(self.fth) * self.disi_hiz * self.dt
        if abs(self.fx) > self.A - 3 or abs(self.fy) > self.A - 3:
            self.fth += math.pi; self.fx = max(-self.A + 3, min(self.A - 3, self.fx))
            self.fy = max(-self.A + 3, min(self.A - 3, self.fy))

        # 3) duyular
        s, c = math.sin(self.th), math.cos(self.th)
        hx, hy = self.x + s * ANT_ON, self.y + c * ANT_ON
        Lx, Ly = hx - c * ANT_YAN, hy + s * ANT_YAN      # sol anten (yönün solu)
        Rx, Ry = hx + c * ANT_YAN, hy - s * ANT_YAN
        R = {}
        R['meyve_koku'] = (self._hz(self._koku(Lx, Ly, 'meyve')), self._hz(self._koku(Rx, Ry, 'meyve')))
        R['cicek_koku'] = (self._hz(self._koku(Lx, Ly, 'cicek'), 80), self._hz(self._koku(Rx, Ry, 'cicek'), 80))
        R['feromon'] = (self._hz(self._disi_koku(Lx, Ly), 90), self._hz(self._disi_koku(Rx, Ry), 90))
        # temas / tat
        self.temas = None
        tat = 0.0
        for i, o in enumerate(self.nes):
            if o['tip'] in ('meyve', 'cicek', 'su') and math.hypot(hx - o['x'], hy - o['y']) < o['r'] + 0.8 and self.alt < 0.3:
                self.temas = (o['tip'], i)
                if o['tip'] == 'meyve':
                    tat = 100.0
                elif o['tip'] == 'cicek':
                    tat = 70.0
        R['tat'] = (tat, None)
        # dişiyi görme: görüş alanında (önde ±150°, 40 mm içinde) hangi taraftaysa o göz
        ddx, ddy = self.fx - self.x, self.fy - self.y
        dd = math.hypot(ddx, ddy)
        rel = (math.atan2(ddx, ddy) - self.th + math.pi) % (2 * math.pi) - math.pi
        gl, gr = 0.0, 0.0
        if dd < 40 and abs(rel) < 2.6:
            boyut = 2 * math.atan(1.25 / max(dd, 1.0))            # dişinin açısal büyüklüğü
            hzg = min(120.0, 120.0 * boyut / 0.35)
            if abs(rel) < 0.15:
                gl = gr = 0.6 * hzg
            elif rel > 0:
                gr = hzg
            else:
                gl = hzg
        R['disi_gorme'] = (gl, gr)
        # ışık: ağaç tacının altında gölge
        golge = any(o['tip'] == 'agac' and math.hypot(self.x - o['x'], self.y - o['y']) < o['tac'] for o in self.nes)
        isik = 3.0 if golge else 15.0
        R['isik'] = (isik, isik)
        sic = sum(70 * math.exp(-max(0, math.hypot(self.x - o['x'], self.y - o['y']) - o['r']) / 5)
                  for o in self.nes if o['tip'] == 'tas')
        R['sicak'] = (min(120.0, sic), None)
        nem = sum(70 * math.exp(-max(0, math.hypot(self.x - o['x'], self.y - o['y']) - o['r']) / 7)
                  for o in self.nes if o['tip'] == 'su')
        R['nem'] = (min(120.0, nem), None)
        # kuş gölgesi: belirli bir yönden yaklaşıp büyür
        kus_durum = None
        tl, tr = 0.0, 0.0
        if self.kus and self.kus['t0'] <= t_sn <= self.kus['t0'] + self.kus['sure']:
            p = (t_sn - self.kus['t0']) / self.kus['sure']
            yak = math.sin(p * math.pi)                         # yaklaş-uzaklaş
            rel = (self.kus['aci'] - self.th + math.pi) % (2 * math.pi) - math.pi
            sag_tarafta = rel > 0
            hzk = 120.0 * yak
            tl, tr = (0.25 * hzk, hzk) if sag_tarafta else (hzk, 0.25 * hzk)
            kus_durum = {'p': round(p, 3), 'aci': self.kus['aci']}
        R['tehdit'] = (tl, tr)

        # uyarı dizisi
        idx_l, p_l = [], []
        for kk, (a, b2) in R.items():
            L, Rr = self.kanal[kk]
            if Rr is None:
                if a > 0.5 and len(L):
                    idx_l.append(L); p_l.append(np.full(len(L), 1 - math.exp(-a * 1e-4)))
            else:
                if a > 0.5 and len(L):
                    idx_l.append(L); p_l.append(np.full(len(L), 1 - math.exp(-a * 1e-4)))
                if b2 > 0.5 and len(Rr):
                    idx_l.append(Rr); p_l.append(np.full(len(Rr), 1 - math.exp(-b2 * 1e-4)))

        self.kayit.append([round(t_sn, 3), round(self.x, 3), round(self.y, 3), round(self.th, 4),
                           round(self.alt, 3), round(self.fx, 2), round(self.fy, 2),
                           {k2: [round(a, 1), None if b2 is None else round(b2, 1)] for k2, (a, b2) in R.items()},
                           self.temas[0] if self.temas else None, 1 if yiyor else 0, kus_durum,
                           round(dd, 2)])
        if idx_l:
            return np.concatenate(idx_l), np.concatenate(p_l)
        return np.zeros(0, dtype=int), np.zeros(0)

    def ozet(self):
        mesafe = [k[11] for k in self.kayit]
        return {'harita': self.h, 'kanallar': {k: v['ad'] for k, v in KANALLAR.items()},
                'disi_ozet': {'baslangic_mm': mesafe[0] if mesafe else None,
                              'en_yakin_mm': min(mesafe) if mesafe else None,
                              'son_mm': mesafe[-1] if mesafe else None,
                              'yakinda_sn': round(sum(1 for m in mesafe if m < 5) * self.dt, 2)},
                'adim_ms': KONTROL_MS, 'iz': self.kayit,
                'kus': self.kus}
