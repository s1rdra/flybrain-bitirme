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

# Connectome verisi (~120 MB) imajın içine bir kere indirilir
RUN python scripts/get_data.py

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["entrypoint.sh"]
CMD ["gui"]
