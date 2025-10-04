# Healthcare Face Recognition System

A secure, privacy-preserving face recognition system specifically designed for healthcare environments. The system automates patient identification, provides optional AI-based genetic disorder screening, implements comprehensive bias detection, and includes strong anti-spoofing security measures.

## 🏥 Features

### Core Functionality
- **Privacy-First Patient Identification**: Secure facial recognition with encryption and consent management
- **Genetic Disorder Pre-Screening**: AI-based facial feature analysis for preliminary genetic disorder detection
- **Bias Detection & Correction**: Comprehensive fairness monitoring across demographic groups
- **Anti-Spoofing Security**: Multi-method liveness detection to prevent fraudulent use
- **HIPAA/GDPR Compliance**: Built-in privacy controls and audit logging

### Security Features
- AES-256 encryption for all sensitive data
- JWT-based authentication and authorization
- Role-based access controls
- Comprehensive audit logging
- Anti-spoofing with multiple liveness detection methods
- Secure key management

### Privacy Features
- Patient consent management system
- Data retention policies
- Anonymized audit logs
- On-device processing capabilities
- Encrypted facial embeddings
- Right to be forgotten implementation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV
- TensorFlow 2.13+
- SQLite
- Redis (optional, for caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face_recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python -c "from src.core.patient_identification import PatientIdentificationSystem; PatientIdentificationSystem()"
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

### Core Endpoints

#### Patient Registration
```http
POST /register-patient
Content-Type: multipart/form-data

{
  "patient_id": "P001",
  "name": "John Doe",
  "date_of_birth": "1990-01-01",
  "medical_record_number": "MRN123456",
  "consent_given": true,
  "image": <image_file>
}
```

#### Patient Identification
```http
POST /identify-patient
Content-Type: multipart/form-data

{
  "session_id": "session_123",
  "image": <image_file>
}
```

#### Consent Management
```http
POST /update-consent
Content-Type: application/json

{
  "patient_id": "P001",
  "consent_type": "face_recognition",
  "consent_given": true
}
```

#### Genetic Screening
```http
GET /genetic-screening/{patient_id}
Authorization: Bearer <jwt_token>
```

#### Fairness Reports
```http
GET /fairness-report
Authorization: Bearer <jwt_token>
```

### Response Examples

#### Successful Identification
```json
{
  "status": "identified",
  "patient_id": "P001",
  "name": "John Doe",
  "confidence_score": 0.95,
  "processing_time_ms": 150.5,
  "genetic_screening_results": [
    {
      "disorder": "down_syndrome",
      "risk_score": 0.3,
      "confidence": 0.8,
      "recommendation": "Low risk for Down Syndrome. No immediate genetic testing recommended.",
      "disclaimer": "This is a preliminary screening tool..."
    }
  ]
}
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Healthcare FR System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Patient   │  │  Genetic    │  │    Bias     │         │
│  │Identification│  │ Screening   │  │ Monitoring  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Privacy   │  │  Security   │  │   Consent   │         │
│  │  Manager    │  │  Manager    │  │  Manager    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Encrypted Database                       │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Patient Registration**
   - Face image captured
   - Facial embedding extracted and encrypted
   - Patient data encrypted and stored
   - Consent recorded

2. **Patient Identification**
   - Face image captured
   - Anti-spoofing checks performed
   - Facial embedding extracted
   - Encrypted database searched
   - Consent verified
   - Optional genetic screening performed

3. **Bias Monitoring**
   - Performance metrics collected by demographic group
   - Fairness metrics calculated
   - Bias alerts generated
   - Recommendations provided

## 🔒 Security & Privacy

### Encryption
- **AES-256-CBC** encryption for all sensitive data
- **PBKDF2-HMAC-SHA256** key derivation
- **HMAC-SHA256** authentication
- Encrypted facial embeddings
- Secure key management

### Anti-Spoofing
- **Blink Detection**: Analyzes eye movement patterns
- **Head Movement**: Detects natural head motion
- **Texture Analysis**: Identifies real vs. fake textures
- **Multi-frame Analysis**: Temporal consistency checks
- **Liveness Scoring**: Combined confidence assessment

### Privacy Compliance
- **HIPAA Compliance**: Healthcare data protection
- **GDPR Compliance**: EU privacy regulations
- **Consent Management**: Granular consent controls
- **Data Retention**: Configurable retention policies
- **Audit Logging**: Comprehensive access tracking
- **Right to be Forgotten**: Data deletion capabilities

## 🧬 Genetic Screening

### Supported Disorders
- Down Syndrome
- Williams Syndrome
- Angelman Syndrome
- Prader-Willi Syndrome
- Noonan Syndrome
- Turner Syndrome
- Marfan Syndrome
- Achondroplasia

### Features
- **Facial Feature Analysis**: 68-point landmark detection
- **Risk Scoring**: AI-based risk assessment
- **Confidence Metrics**: Reliability indicators
- **Clinical Recommendations**: Actionable guidance
- **Medical Disclaimers**: Clear limitations

## ⚖️ Bias Detection

### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates
- **Equalized Odds**: Equal TPR and FPR across groups
- **Equal Opportunity**: Equal TPR across groups
- **Disparate Impact**: 80% rule compliance
- **Statistical Parity**: Equal prediction distributions

### Monitoring
- **Real-time Bias Detection**: Continuous monitoring
- **Demographic Group Analysis**: Performance by group
- **Fairness Reports**: Comprehensive assessments
- **Recommendations**: Bias mitigation guidance
- **Alert System**: Threshold-based notifications

## 🛠️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Security
SECURITY_ENCRYPTION_KEY=your_encryption_key
SECURITY_JWT_SECRET=your_jwt_secret

# Privacy
PRIVACY_DATA_RETENTION_DAYS=365
PRIVACY_ENABLE_CONSENT_MANAGEMENT=true

# Models
MODEL_CONFIDENCE_THRESHOLD=0.85
MODEL_GENETIC_SCREENING_THRESHOLD=0.7

# Bias Detection
BIAS_BIAS_THRESHOLD=0.1
BIAS_ENABLE_BIAS_MONITORING=true
```

### Database Configuration

The system uses SQLite by default but can be configured for other databases:

```python
DATABASE_DATABASE_URL=sqlite:///./data/encrypted_db/healthcare_fr.db
# or
DATABASE_DATABASE_URL=postgresql://user:pass@localhost/healthcare_fr
```

## 📊 Monitoring & Analytics

### System Statistics
- Patient identification rates
- Consent compliance metrics
- Bias detection statistics
- Genetic screening results
- Security incident tracking

### Performance Metrics
- Processing times
- Accuracy rates by demographic group
- Model performance tracking
- Resource utilization

### Audit Logs
- All patient identifications
- Consent changes
- System access
- Data modifications
- Security events

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Security Testing
```bash
# Run security tests
pytest tests/security/ -v

# Run bias detection tests
pytest tests/bias_detection/ -v
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t healthcare-fr .

# Run container
docker run -p 8000:8000 healthcare-fr
```

### Production Considerations
- Use HTTPS in production
- Configure proper JWT secrets
- Set up database backups
- Enable monitoring and alerting
- Configure log aggregation
- Set up security scanning

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## ⚠️ Disclaimer

This system is designed for assistive screening purposes only. It is NOT a diagnostic tool and should NOT replace professional medical evaluation. Any positive results should be confirmed through proper genetic testing and consultation with qualified healthcare professionals.

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
  - Patient identification
  - Genetic screening
  - Bias detection
  - Privacy controls
  - Security features
