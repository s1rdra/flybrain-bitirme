"""Sanal sinek beyni — simülasyon motoru.

Kaynak projenin (qp-hiro/flybrain, Shiu ve ark. 2024 LIF modeli) sabitlerini ve
nöron grubu çözümleyicisini kullanır; üstüne şunları ekler:
  * istediğin kadar uyaran (duyu) aynı anda
  * "ilaç" etkisi: nörotransmitter tipine göre sinaps ağırlıklarını ölçekleme
  * lezyon: nöron gruplarını susturma
  * ilerleme bildirimi (arayüzdeki çubuk için)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

FLYBRAIN_DIR = Path(os.environ.get('FLYBRAIN_DIR', '/opt/flybrain'))
sys.path.insert(0, str(FLYBRAIN_DIR))

import sim  # noqa: E402  (kaynak proje: model sabitleri)
from groups import MN9, annotations, resolve  # noqa: E402

DT = sim.DT
BIN_MS = 5                       # arayüze gönderilen kare aralığı [ms]
BIN_STEPS = int(round(BIN_MS * 1e-3 / DT))

# ---------------------------------------------------------------- koşullar
# Duyular: her biri bir veya birden fazla nöron grubu (groups.py söz dizimi)
DUYULAR = {
    'seker':    {'ad': 'Şeker tadı',            'grup': 'tat',   'spec': ['sugar'],                 'hz': 100},
    'aci':      {'ad': 'Acı tadı',              'grup': 'tat',   'spec': ['bitter'],                'hz': 100},
    'tuz':      {'ad': 'Az tuzlu tat',          'grup': 'tat',   'spec': ['subclass:low-salt'],     'hz': 100},
    'isik':     {'ad': 'Işık (göz, R1-6)',      'grup': 'görme', 'spec': ['type:R1-6'],             'hz': 30},
    'renk':     {'ad': 'Renkli ışık (R7/R8)',   'grup': 'görme', 'spec': ['type:R7', 'type:R8'],    'hz': 30},
    'sirke':    {'ad': 'Elma sirkesi kokusu',   'grup': 'koku',  'spec': ['type:ORN_DM1', 'type:ORN_DM4', 'type:ORN_VA2', 'type:ORN_DP1m'], 'hz': 50},
    'co2':      {'ad': 'CO₂ kokusu',            'grup': 'koku',  'spec': ['type:ORN_V'],            'hz': 50},
    'kuf':      {'ad': 'Küf kokusu (geosmin)',  'grup': 'koku',  'spec': ['type:ORN_DA2'],          'hz': 50},
    'feromon':  {'ad': 'Feromon (cVA)',         'grup': 'koku',  'spec': ['type:ORN_DA1'],          'hz': 50},
    'sicak':    {'ad': 'Sıcak',                 'grup': 'beden', 'spec': ['subclass:heating'],      'hz': 50},
    'soguk':    {'ad': 'Soğuk',                 'grup': 'beden', 'spec': ['subclass:cooling', 'subclass:cold'], 'hz': 50},
    'nem':      {'ad': 'Nem',                   'grup': 'beden', 'spec': ['subclass:humid', 'subclass:moist'], 'hz': 50},
    'ruzgar':   {'ad': 'Rüzgâr / yerçekimi',    'grup': 'beden', 'spec': ['subclass:wind_gravity'], 'hz': 50},
    'ses':      {'ad': 'Ses (kur şarkısı)',     'grup': 'beden', 'spec': ['subclass:auditory'],     'hz': 50},
    'dokunma':  {'ad': 'Başa dokunma',          'grup': 'beden', 'spec': ['subclass:head bristle'], 'hz': 50},
}

# "İlaçlar": nörotransmitter tipine göre giden sinapsları ölçekle (1.0 = normal)
NT_LISTESI = {
    'gaba':          'GABA (fren)',
    'glutamate':     'Glutamat (fren)',
    'acetylcholine': 'Asetilkolin (gaz)',
    'dopamine':      'Dopamin',
    'serotonin':     'Serotonin',
    'octopamine':    'Oktopamin',
}
ILAC_HAZIR = {
    'pikrotoksin': {'ad': 'Pikrotoksin benzeri — GABA engelleyici', 'nt': {'gaba': 0.0}},
    'diazepam':    {'ad': 'Diazepam benzeri — GABA güçlendirici',   'nt': {'gaba': 2.0}},
    'imidakloprid': {'ad': 'İmidakloprid benzeri — ACh aşırı uyarım', 'nt': {'acetylcholine': 1.5}},
    'ivermektin':  {'ad': 'İvermektin benzeri — glutamat freni artar', 'nt': {'glutamate': 2.0}},
}

# Davranış göstergeleri: kararı okuduğumuz çıkış nöronları
DAVRANISLAR = [
    {'id': 'yeme',   'ad': 'Yeme (hortum uzatma)', 'spec': ['mn9'],         'not': 'MN9 motor nöronu'},
    {'id': 'kacis',  'ad': 'Kaçış / sıçrama',      'spec': ['type:DNp01'],  'not': 'Dev lif (Giant Fiber)'},
    {'id': 'ileri',  'ad': 'İleri yürüme',         'spec': ['type:DNp09'],  'not': 'DNp09 inen nöronu'},
    {'id': 'geri',   'ad': 'Geri yürüme',          'spec': ['type:MDN'],    'not': 'Moonwalker (MDN)'},
    {'id': 'sol',    'ad': 'Sola dönme',           'spec': ['type:DNa02@L'], 'not': 'DNa02 sol'},
    {'id': 'sag',    'ad': 'Sağa dönme',           'spec': ['type:DNa02@R'], 'not': 'DNa02 sağ'},
]

SUPER_AD = {
    'sensory': 'Duyu', 'sensory_ascending': 'Duyu (yükselen)', 'optic': 'Görme lobu',
    'visual_projection': 'Görsel izdüşüm', 'visual_centrifugal': 'Görsel geri besleme',
    'central': 'Merkez beyin', 'ascending': 'Yükselen', 'descending': 'İnen (komut)',
    'motor': 'Motor', 'endocrine': 'Hormon', None: 'Etiketsiz',
}
SUPER_SIRA = ['sensory', 'sensory_ascending', 'optic', 'visual_projection',
              'visual_centrifugal', 'central', 'ascending', 'descending', 'motor',
              'endocrine']


class Beyin:
    """Ağı bir kere yükler, her deneyde kopyası üzerinde çalışır."""

    def __init__(self, log=print):
        t0 = time.time()
        log('Ağ yükleniyor (connectome) ...')
        self.flyids, self.w = sim.load_network()
        self.n = len(self.flyids)
        self.id2idx = {int(f): i for i, f in enumerate(self.flyids)}
        log(f'{self.n:,} nöron, {self.w.nnz:,} bağlantı')

        ann = annotations().drop_duplicates('root_id').set_index('root_id')
        ann = ann.reindex(self.flyids)
        self.ann = ann
        self.super = ann['super_class'].where(ann['super_class'].notna(), None).to_numpy()
        self.ctype = ann['cell_type'].fillna('').to_numpy()
        self.nt = ann['top_nt'].fillna('').to_numpy()
        self.has_pos = ann['pos_x'].notna().to_numpy()

        # satır (presinaptik nöron) başına NT -> ağırlık ölçekleme için
        self.row_len = np.diff(self.w.indptr)

        self._types = sorted({t for t in self.ctype if t})
        log(f'Hazır ({time.time() - t0:.0f} sn)')

    # ------------------------------------------------------------ yardımcılar
    def idx(self, specs):
        out = []
        for s in specs:
            out += [self.id2idx[f] for f in resolve(s) if f in self.id2idx]
        return np.unique(np.array(out, dtype=np.int64))

    def tipler(self):
        return self._types

    def beyin_haritasi(self):
        """Nöron konumları (Int16, normalize) + bölge kodu (Uint8). 255 = konumsuz."""
        a = self.ann
        # FlyWire voxel: 4 x 4 x 40 nm
        x = a['pos_x'].to_numpy() * 4.0
        y = a['pos_y'].to_numpy() * 4.0
        z = a['pos_z'].to_numpy() * 40.0
        ok = self.has_pos
        c = np.array([np.nanmean(v[ok]) for v in (x, y, z)])
        s = max(np.nanmax(np.abs(v[ok] - cc)) for v, cc in zip((x, y, z), c))
        pos = np.zeros((self.n, 3), dtype=np.int16)
        for k, (v, cc) in enumerate(zip((x, -y, -z), (c[0], -c[1], -c[2]))):
            pos[ok, k] = np.round((v[ok] - cc) / s * 32000).astype(np.int16)
        code = np.full(self.n, 255, dtype=np.uint8)
        for i, sc in enumerate(SUPER_SIRA):
            code[self.super == sc] = i
        code[~ok] = 255
        return pos.tobytes() + code.tobytes()

    # ------------------------------------------------------------ simülasyon
    def calistir(self, duyular=None, nt_olcek=None, lezyon=None, sure=0.5, seed=None,
                 ilerleme=None):
        """duyular: {anahtar: hz}; nt_olcek: {nt: çarpan}; lezyon: [spec]; sure: sn."""
        t_start = time.time()
        w = self.w.copy()

        # ilaç: presinaptik nöronun NT'sine göre giden ağırlıkları ölçekle
        row_scale = np.ones(self.n, dtype=np.float32)
        for nt, k in (nt_olcek or {}).items():
            if nt in NT_LISTESI and float(k) != 1.0:
                row_scale[self.nt == nt] = float(k)
        if not np.all(row_scale == 1.0):
            w.data *= np.repeat(row_scale, self.row_len)

        # lezyon: giden sinapsları sıfırla
        sus = self.idx(lezyon) if lezyon else np.zeros(0, dtype=np.int64)
        for i in sus:
            w.data[w.indptr[i]:w.indptr[i + 1]] = 0.0

        stim = []
        for key, hz in (duyular or {}).items():
            if key in DUYULAR and float(hz) > 0:
                ix = self.idx(DUYULAR[key]['spec'])
                if len(ix):
                    stim.append((ix, float(hz)))

        n_steps = int(round(sure / DT))
        rng = np.random.default_rng(seed)
        st, si = _run(w, stim, rng, n_steps, ilerleme)
        sim_sn = time.time() - t_start
        return self._ozet(st, si, n_steps, sure, sim_sn, stim, sus)

    def _ozet(self, st, si, n_steps, sure, sim_sn, stim, sus):
        n_bins = int(np.ceil(n_steps / BIN_STEPS))
        b = st // BIN_STEPS
        # kare başına ateşleyen nöronlar (tekil)
        order = np.lexsort((si, b))
        b, si_s = b[order], si[order]
        if len(b):
            keep = np.ones(len(b), dtype=bool)
            keep[1:] = (b[1:] != b[:-1]) | (si_s[1:] != si_s[:-1])
            b, si_s = b[keep], si_s[keep]
        offs = np.searchsorted(b, np.arange(n_bins + 1)).astype(int)

        counts = np.bincount(si, minlength=self.n)
        rate = counts / sure

        davranis = []
        for d in DAVRANISLAR:
            ix = self.idx(d['spec'])
            hz = float(rate[ix].mean()) if len(ix) else 0.0
            davranis.append({'id': d['id'], 'ad': d['ad'], 'not': d['not'],
                             'hz': round(hz, 1), 'n': int(len(ix)),
                             'idx': ix.astype(int).tolist()})
        girdi = np.unique(np.concatenate([ix for ix, _ in stim])) if stim else np.zeros(0, int)

        bolgeler = []
        for i, sc in enumerate(SUPER_SIRA + [None]):
            m = (self.super == sc) if sc else np.array([s is None for s in self.super])
            tot = int(m.sum())
            if not tot:
                continue
            akt = int((counts[m] > 0).sum())
            bolgeler.append({'kod': i if sc else 255, 'ad': SUPER_AD[sc], 'toplam': tot,
                             'aktif': akt, 'spike': int(counts[m].sum())})

        # en aktif hücre tipleri
        df = pd.DataFrame({'tip': self.ctype, 'c': counts, 'nt': self.nt})
        df = df[(df.tip != '') & (df.c > 0)]
        g = df.groupby('tip').agg(spike=('c', 'sum'), n=('c', 'size'), nt=('nt', 'first'))
        g['hz'] = g.spike / g.n / sure
        top = g.sort_values('spike', ascending=False).head(15).reset_index()

        zaman = np.bincount(st // BIN_STEPS, minlength=n_bins)[:n_bins]

        return {
            'sure': sure, 'bin_ms': BIN_MS, 'sim_sn': round(sim_sn, 1),
            'toplam_spike': int(len(st)), 'aktif_noron': int((counts > 0).sum()),
            'uyarilan': int(sum(len(ix) for ix, _ in stim)), 'susturulan': int(len(sus)),
            'davranis': davranis, 'bolgeler': bolgeler,
            'en_aktif': [{'tip': r.tip, 'hz': round(float(r.hz), 1), 'n': int(r.n), 'nt': r.nt}
                         for r in top.itertuples()],
            'zaman': zaman.astype(int).tolist(),
            'girdi_idx': girdi.astype(int).tolist(),
            'sus_idx': np.asarray(sus).astype(int).tolist(),
            'kare_ofset': offs.tolist(),
            'kare_noron': si_s.astype(int).tolist(),
        }


def _run(w, stim, rng, n_steps, ilerleme=None):
    """sim.run_trial ile aynı model; süre ayarlı ve ilerleme bildirimli."""
    n = w.shape[0]
    V_0, V_RST, V_TH = sim.V_0, sim.V_RST, sim.V_TH
    DLY, RFC = sim.DLY_STEPS, sim.RFC_STEPS
    v = np.full(n, V_0, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    rfc = np.zeros(n, dtype=np.int32)
    buf = np.zeros((DLY, n), dtype=np.float32)

    b = np.float32(np.exp(-DT / sim.T_MBR))
    a = np.float32(np.exp(-DT / sim.TAU))
    c = np.float32(sim.TAU / (sim.TAU - sim.T_MBR)
                   * (np.exp(-DT / sim.TAU) - np.exp(-DT / sim.T_MBR)))
    w_poi = np.float32(sim.W_SYN * sim.F_POI)

    if stim:
        stim_idx = np.concatenate([ix for ix, _ in stim])
        stim_p = np.concatenate([np.full(len(ix), 1.0 - np.exp(-r * DT)) for ix, r in stim])
    else:
        stim_idx = np.zeros(0, dtype=int)
        stim_p = np.zeros(0)
    is_exc = np.zeros(n, dtype=bool)
    is_exc[stim_idx] = True

    spk_t, spk_i = [], []
    for t in range(n_steps):
        slot = t % DLY
        g += buf[slot]
        buf[slot] = 0.0
        active = rfc <= 0
        vm = v[active]
        gm = g[active]
        v[active] = V_0 + (vm - V_0) * b + gm * c
        g[active] = gm * a
        rfc[~active] -= 1
        if len(stim_idx):
            ev = rng.random(len(stim_idx)) < stim_p
            if ev.any():
                v[stim_idx[ev]] += w_poi
        spiked = (v > V_TH) & (rfc <= 0)
        if spiked.any():
            idx = np.nonzero(spiked)[0]
            spk_t.append(np.full(len(idx), t, dtype=np.int32))
            spk_i.append(idx.astype(np.int32))
            buf[slot] += np.asarray(w[idx].sum(axis=0)).ravel()
            v[idx] = V_RST
            g[idx] = 0.0
            rfc[idx] = np.where(is_exc[idx], 0, RFC)
        if ilerleme and t % 100 == 0:
            ilerleme(t / n_steps)
    if ilerleme:
        ilerleme(1.0)
    if spk_t:
        return np.concatenate(spk_t), np.concatenate(spk_i)
    return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
