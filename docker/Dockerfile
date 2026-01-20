FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

EXPOSE 8080

CMD ["python", "SynthesisServer_fastAPI.py"]