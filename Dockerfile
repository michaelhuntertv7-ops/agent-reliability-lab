FROM python:3.11-slim
WORKDIR /app
COPY . /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARL_HOST=0.0.0.0
EXPOSE 8765
CMD ["python", "agent_reliability_lab_app_v0_3_public.py"]
