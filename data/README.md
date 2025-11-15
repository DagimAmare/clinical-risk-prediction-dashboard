# Dataset Documentation

## Diabetes 130-US Hospitals for Years 1999-2008

### Source
UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **License**: CC BY 4.0
- **Citation**: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records," BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

### Dataset Overview
- **Total Records**: 101,766 patient encounters
- **Time Period**: 1999-2008 (10 years)
- **Hospitals**: 130 US hospitals and integrated delivery networks
- **Features**: 50 variables (demographics, clinical measures, medications, outcomes)

### Target Variable
- **readmitted**: Whether patient was readmitted to hospital
  - `<30`: Readmitted within 30 days
  - `>30`: Readmitted after 30 days
  - `NO`: Not readmitted

### Key Features

#### Demographics
- `race`: Patient race (Caucasian, AfricanAmerican, Hispanic, Asian, Other)
- `gender`: Male/Female
- `age`: Age groups in 10-year bins ([0-10), [10-20), ..., [90-100))

#### Clinical Measures
- `time_in_hospital`: Days in hospital (1-14)
- `num_lab_procedures`: Number of lab tests performed
- `num_procedures`: Number of procedures performed
- `num_medications`: Number of distinct medications
- `number_diagnoses`: Number of diagnoses (1-16)

#### Medical History
- `number_outpatient`: Outpatient visits in year before encounter
- `number_emergency`: Emergency visits in year before encounter
- `number_inpatient`: Inpatient visits in year before encounter

#### Medications
- Various diabetes medications (metformin, insulin, glipizide, etc.)
- `diabetesMed`: Whether diabetes medication prescribed
- `change`: Whether medication changed during stay

#### Admission Details
- `admission_type_id`: Type of admission (emergency, urgent, elective, etc.)
- `discharge_disposition_id`: Where patient discharged to
- `admission_source_id`: Where patient admitted from

### Preprocessing Steps

1. **Data Cleaning**
   - Removed records with race = '?'
   - Replaced '?' with NaN
   - Dropped columns with >40% missing data (weight, medical_specialty, max_glu_serum, A1Cresult)
   - Removed duplicate patient encounters (kept most recent)
   - **Result**: 69,668 unique patients (68.5% of original)

2. **Feature Engineering**
   - Created `age_numeric`: Numeric age from age groups
   - Created `elderly`: Binary indicator for age ≥65
   - Created `polypharmacy`: Binary for ≥10 medications
   - Created `high_comorbidity`: Binary for ≥7 diagnoses
   - Created `long_stay`: Binary for >7 days in hospital
   - Created `high_utilization`: Binary for many procedures/labs
   - Created `emergency_admit`: Binary for emergency admission
   - Created `prior_utilization`: Sum of prior visits
   - Created `uncontrolled_diabetes`: A1C >8 or not tested
   - Created `total_med_changes`: Count of medication changes

3. **Target Variable**
   - Created `readmitted_30days`: Binary indicator (1 if <30, 0 otherwise)
   - **30-day readmission rate**: 4.55% (3,170 of 69,668)

4. **Feature Selection**
   - Selected 20 most clinically relevant features
   - Encoded categorical variables with LabelEncoder
   - No scaling (tree-based model used)

### Final Processed Dataset
- **Samples**: 69,668 patients
- **Features**: 20 clinical and engineered features
- **Target**: Binary 30-day readmission (4.55% positive rate)
- **Location**: `data/processed/processed_data.csv`

### Data Quality Notes
- Original dataset has imbalanced target (11% readmitted within 30 days)
- After removing duplicates, rate decreased to 4.55%
- SMOTE used during training to balance classes
- Missing A1C data limits diabetes control assessment

### Ethical Considerations
- Dataset contains protected health information that has been de-identified
- Race is included as a demographic variable but should be used carefully to avoid bias
- Model performance should be evaluated across demographic subgroups
- Clinical validation required before deployment
