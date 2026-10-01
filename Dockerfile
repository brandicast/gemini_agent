FROM python:3.11-slim


# 指定 Image 中的工作目錄
WORKDIR /code

# 將 Dockerfile 所在目錄下的所有檔案複製到 Image 的工作目錄 /code 底下
ADD . /code


# 在 Image 中執行的指令：安裝 requirements.txt 中所指定的 dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Automatically find and install all plugin-specific dependencies
RUN find src/tools -name "requirements.txt" -exec pip install --no-cache-dir -r {} \;

RUN mkdir -p history 

ENV API_KEY={$YOU_API_KEY}
ENV CWA_API_KEY={$CWA_API_KEY}

VOLUME '/code/history'

# Container 啟動指令：Container 啟動後通過 python 運行 server.py
CMD ["python", "src/server.py"]
