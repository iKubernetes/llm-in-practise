FROM python:3.11-slim

WORKDIR /app
COPY fault_prediction_service.py gbc_fault_prediction_model.pkl scaler.pkl /app/
RUN pip install flask pandas scikit-learn numpy

EXPOSE 5000
CMD ["python", "fault_prediction_service.py"]
