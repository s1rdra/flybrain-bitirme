# Sanal meyve sineği beyni — Mac (Apple Silicon) ve Windows (x86) için ortak ortam
FROM python:3.12-slim

# Kaynak proje: qp-hiro/flybrain (Shiu ve ark. 2024 LIF modelinin saf NumPy hali)
ARG FLYBRAIN_REPO=https://github.com/qp-hiro/flybrain.git
ARG FLYBRAIN_COMMIT=baf53a1f2947738a47c07a4e53f54f4462c82873

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git nodejs ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Kaynak kodu sabit bir commit'e kilitle (herkeste aynı sürüm çalışsın)
RUN git clone ${FLYBRAIN_REPO} /opt/flybrain \
 && cd /opt/flybrain && git checkout ${FLYBRAIN_COMMIT}

WORKDIR /opt/flybrain
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r requirements.txt -r /tmp/requirements.txt

# Nöron etiket dosyası FlyWire tarafında 2026-09-29'da güncellendi; kaynak projenin
# beklediği sürüme (boyut kontrolü tutsun diye) sabitliyoruz
ARG ANNOTATIONS_COMMIT=17fc57722002e1a7d38cdd0c89ac382bf92718da
RUN sed -i "s#flywire_annotations/main/#flywire_annotations/${ANNOTATIONS_COMMIT}/#" scripts/get_data.py

# Connectome verisi (~120 MB) imajın içine bir kere indirilir
RUN python scripts/get_data.py

# Bizim kodumuz (compose ile çalışınca ./app klasörü bunun üstüne bağlanır)
COPY app /opt/app

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["entrypoint.sh"]
CMD ["gui"]
