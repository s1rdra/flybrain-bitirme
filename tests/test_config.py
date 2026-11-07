import math
import sys
import unittest
from pathlib import Path
from types import ModuleType

APP_DIR = Path(__file__).resolve().parent.parent / 'app'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Gerekli mock modülleri (bağımsız ve hızlı çalışabilmesi için)
if 'scipy' not in sys.modules:
    scipy_mock = ModuleType('scipy')
    scipy_mock.sparse = ModuleType('sparse')
    sys.modules['scipy'] = scipy_mock
    sys.modules['scipy.sparse'] = scipy_mock.sparse

if 'pandas' not in sys.modules:
    pd_mock = ModuleType('pandas')
    sys.modules['pandas'] = pd_mock

if 'sim' not in sys.modules:
    sim_mock = ModuleType('sim')
    sim_mock.DT = 1e-4
    sim_mock.V_0 = -52.0
    sim_mock.V_RST = -52.0
    sim_mock.V_TH = -45.0
    sim_mock.DLY_STEPS = 20
    sim_mock.RFC_STEPS = 20
    sim_mock.T_MBR = 0.02
    sim_mock.TAU = 0.005
    sim_mock.W_SYN = 1.0
    sim_mock.F_POI = 1.0
    sim_mock.load_network = lambda: ([], None)
    sys.modules['sim'] = sim_mock

if 'groups' not in sys.modules:
    groups_mock = ModuleType('groups')
    groups_mock.MN9 = 127193
    groups_mock.annotations = lambda: None
    groups_mock.resolve = lambda spec: []
    sys.modules['groups'] = groups_mock

import beyin as B


class TestBeyinYapilandirma(unittest.TestCase):
    """Beyin modeli duyusal, farmakolojik ve davranışsal yapılandırma testleri."""

    def test_duyular_tutarliligi(self):
        self.assertGreater(len(B.DUYULAR), 0)
        for key, veri in B.DUYULAR.items():
            self.assertIn('ad', veri, f"Duyu '{key}' için 'ad' eksik")
            self.assertIn('grup', veri, f"Duyu '{key}' için 'grup' eksik")
            self.assertIn('spec', veri, f"Duyu '{key}' için 'spec' eksik")
            self.assertIn('hz', veri, f"Duyu '{key}' için 'hz' eksik")
            self.assertIsInstance(veri['spec'], list)
            self.assertGreater(len(veri['spec']), 0)
            self.assertGreater(veri['hz'], 0)

    def test_norotransmitter_ve_ilaclar(self):
        # Temel NT listesi doğrulaması
        beklenen_nt = {'gaba', 'glutamate', 'acetylcholine', 'dopamine', 'serotonin', 'octopamine'}
        self.assertEqual(set(B.NT_LISTESI.keys()), beklenen_nt)

        # İlaç tanımları ve nörobiyolojik etki tutarlılığı
        for ilac_id, ilac in B.ILAC_HAZIR.items():
            self.assertIn(ilac['kat'], B.MADDE_KATEGORI, f"Geçersiz kategori: {ilac['kat']}")
            self.assertIn('temsil', ilac)
            self.assertIn('aciklama', ilac)

            # Değiştirilen tüm NT'ler NT_LISTESI'nde kayıtlı olmalı
            for nt in ilac.get('nt', {}).keys():
                self.assertIn(nt, B.NT_LISTESI, f"{ilac_id} içindeki '{nt}' tanımlı değil")

            # Eğer tat duyusu tetikliyorsa DUYULAR içinde olmalı
            for duyu in ilac.get('duyular', {}).keys():
                self.assertIn(duyu, B.DUYULAR, f"{ilac_id} içindeki duyu '{duyu}' tanımlı değil")

    def test_davranis_ciktilari(self):
        # 7 temel motor davranışın varlığı
        beklenen_davranislar = {'yeme', 'kacis', 'ileri', 'geri', 'sol', 'sag', 'ucus'}
        mevcut_davranislar = {d['id'] for d in B.DAVRANISLAR}
        self.assertEqual(beklenen_davranislar, mevcut_davranislar)

        for d in B.DAVRANISLAR:
            self.assertIn('ad', d)
            self.assertIn('spec', d)
            self.assertIn('not', d)

    def test_super_bolge_eslesmesi(self):
        # SUPER_SIRA içindeki her bölgenin SUPER_AD karşılığı olmalı
        for s in B.SUPER_SIRA:
            self.assertIn(s, B.SUPER_AD)

    def test_lif_entegrasyon_katsayilari(self):
        # b, a, c katsayılarının analitik geçerliliği
        DT = 1e-4
        T_MBR = 0.02
        TAU = 0.005

        b = math.exp(-DT / T_MBR)
        a = math.exp(-DT / TAU)
        c = (TAU / (TAU - T_MBR)) * (math.exp(-DT / TAU) - math.exp(-DT / T_MBR))

        self.assertGreater(b, 0.0)
        self.assertLess(b, 1.0)
        self.assertGreater(a, 0.0)
        self.assertLess(a, 1.0)
        self.assertGreater(c, 0.0)


if __name__ == '__main__':
    unittest.main()
