import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# app klasörünü sys.path'e ekle
APP_DIR = Path(__file__).resolve().parent.parent / 'app'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import dunya as D


class TestDunyaHarita(unittest.TestCase):
    """Açık dünya harita oluşturma fonksiyonlarının testi."""

    def test_harita_bahce_yapisi(self):
        harita = D.harita_olustur(seed=42, tip='bahce')
        self.assertEqual(harita['arena'], D.ARENA)
        self.assertEqual(harita['tip'], 'bahce')
        self.assertIn('sinek', harita)
        self.assertIn('disi', harita)
        self.assertIn('nesneler', harita)
        self.assertGreater(len(harita['nesneler']), 0)

        # Ağaç, meyve, çiçek nesnelerinin varlığını doğrula
        tipler = {obj['tip'] for obj in harita['nesneler']}
        self.assertTrue({'agac', 'meyve', 'cicek'}.issubset(tipler))

    def test_harita_kur_yapisi(self):
        harita = D.harita_olustur(seed=42, tip='kur')
        self.assertEqual(harita['arena'], 22.0)
        self.assertEqual(harita['tip'], 'kur')
        self.assertEqual(len(harita['nesneler']), 0)
        self.assertIn('sinek', harita)
        self.assertIn('disi', harita)

    def test_deterministik_seed(self):
        harita1 = D.harita_olustur(seed=123, tip='bahce')
        harita2 = D.harita_olustur(seed=123, tip='bahce')
        self.assertEqual(harita1['sinek'], harita2['sinek'])
        self.assertEqual(harita1['disi'], harita2['disi'])
        self.assertEqual(len(harita1['nesneler']), len(harita2['nesneler']))


class TestDunyaFizikVeDuyular(unittest.TestCase):
    """Duyusal tepki fonksiyonları ve açık dünya motoru testi."""

    def test_hz_tepki_egrisi(self):
        # 0 derişimde frekans 0 olmalı
        self.assertEqual(D.Dunya._hz(0.0), 0.0)

        # Derişim arttıkça frekans monoton artmalı
        hz1 = D.Dunya._hz(0.2)
        hz2 = D.Dunya._hz(1.0)
        hz3 = D.Dunya._hz(10.0)
        self.assertLess(hz1, hz2)
        self.assertLess(hz2, hz3)

        # Asimptotik olarak mx değerine yaklaşmalı
        mx = 110.0
        self.assertLess(D.Dunya._hz(1000.0, mx=mx), mx)
        self.assertGreater(D.Dunya._hz(1000.0, mx=mx), mx * 0.99)

    def test_dunya_adim_ve_ozet(self):
        # Mock beyin oluştur
        mock_beyin = MagicMock()
        mock_beyin.n = 100
        mock_beyin.DAVRANISLAR = [
            {'id': 'yeme', 'ad': 'Yeme', 'spec': ['mn9']},
            {'id': 'kacis', 'ad': 'Kaçış', 'spec': ['type:DNp01']},
            {'id': 'ileri', 'ad': 'İleri', 'spec': ['type:DNp09']},
            {'id': 'geri', 'ad': 'Geri', 'spec': ['type:MDN']},
            {'id': 'sol', 'ad': 'Sol', 'spec': ['type:DNa02@L']},
            {'id': 'sag', 'ad': 'Sağ', 'spec': ['type:DNa02@R']},
            {'id': 'ucus', 'ad': 'Uçuş', 'spec': ['prefix:DNg02_']},
        ]
        # Her spec için sembolik bir indeks listesi döndür
        mock_beyin.idx.side_effect = lambda spec: np.array([1, 2], dtype=np.int64)

        harita = D.harita_olustur(seed=1, tip='bahce')
        dw = D.Dunya(mock_beyin, harita, sure=0.5, seed=1)

        # İlk adımı simüle et
        t_sn = 0.02
        yeni_spike = np.array([1], dtype=np.int32)
        stim_idx, stim_p = dw.adim_at(t_sn, yeni_spike)

        # Çıktı tiplerini ve aralıklarını kontrol et
        self.assertIsInstance(stim_idx, np.ndarray)
        self.assertIsInstance(stim_p, np.ndarray)
        if len(stim_p) > 0:
            self.assertTrue(np.all((stim_p >= 0.0) & (stim_p <= 1.0)))

        # Sinek arenanın sınırları içinde kalmalı
        self.assertLessEqual(abs(dw.x), dw.A)
        self.assertLessEqual(abs(dw.y), dw.A)

        # Özet çıktısını kontrol et
        ozet = dw.ozet()
        self.assertIn('harita', ozet)
        self.assertIn('kanallar', ozet)
        self.assertIn('iz', ozet)
        self.assertIn('disi_ozet', ozet)
        self.assertGreater(len(ozet['iz']), 0)


if __name__ == '__main__':
    unittest.main()
