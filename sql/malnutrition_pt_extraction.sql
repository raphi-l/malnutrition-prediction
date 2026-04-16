BEGIN TRANSACTION;
CREATE VIEW "d_nutrition_labs" AS SELECT *
FROM d_labitems
WHERE fluid LIKE 'blood' AND
	(label LIKE '%albumin'
	OR label LIKE 'hemoglobin'
	OR label LIKE 'hema%'
	OR label LIKE 'phos%'
	OR label LIKE '%glucose%'
	OR label LIKE '%potassium%'
	OR label LIKE '%magnesium%'
	OR label LIKE 'Alk%'
	OR label LIKE 'Aspar%'
	OR label LIKE 'prealb%'
	OR label LIKE 'c-%');
CREATE VIEW "mal_pt_data" AS SELECT
	a.subject_id,
	a.hadm_id,
	p.gender,
	p.anchor_age as age,
	a.admission_type,
	a.admission_location,
	a.marital_status,
	a.race,
	nutrition_lab_events.*,
	omr_admit_antho.*,
	CASE
		WHEN d.icd_code IS NOT NULL THEN 1
		ELSE 0
	END as has_malnutrition
FROM admissions a
LEFT JOIN  patients p ON a.subject_id = p.subject_id
LEFT JOIN omr_admit_antho ON a.subject_id = omr_admit_antho.subject_id
LEFT JOIN nutrition_lab_events ON a.hadm_id = nutrition_lab_events.hadm_id
LEFT JOIN (
    SELECT DISTINCT subject_id, hadm_id, icd_code
    FROM diagnoses_icd
    WHERE icd_code LIKE 'E4%'
        OR icd_code LIKE '263%'
) d ON a.hadm_id = d.hadm_id;
CREATE VIEW "nutrition_lab_events" AS SELECT
    le.hadm_id,
    MAX(CASE WHEN dnl.label = 'Albumin' THEN le.valuenum END) as albumin_admit,
    MAX(CASE WHEN dnl.label = 'Alkaline Phosphatase' THEN le.valuenum END) as alkaline_phosphatase_admit,
    MAX(CASE WHEN dnl.label = 'Asparate Aminotransferase' THEN le.valuenum END) as asparate_aminotransferase_admit,
    MAX(CASE WHEN dnl.label = 'Asparate Aminotransferase (AST)' THEN le.valuenum END) as asparate_aminotransferase_ast_admit,
    MAX(CASE WHEN dnl.label = 'C-Reactive Protein' THEN le.valuenum END) as c_reactive_protein_admit,
    MAX(CASE WHEN dnl.label = 'Glucose' THEN le.valuenum END) as glucose_admit,
    MAX(CASE WHEN dnl.label = 'Glucose, Whole Blood' THEN le.valuenum END) as glucose_whole_blood_admit,
    MAX(CASE WHEN dnl.label = 'Hematocrit' THEN le.valuenum END) as hematocrit_admit,
    MAX(CASE WHEN dnl.label = 'Hematocrit, Calculated' THEN le.valuenum END) as hematocrit_calculated_admit,
    MAX(CASE WHEN dnl.label = 'Hemoglobin' THEN le.valuenum END) as hemoglobin_admit,
    MAX(CASE WHEN dnl.label = 'Magnesium' THEN le.valuenum END) as magnesium_admit,
    MAX(CASE WHEN dnl.label = 'Phosphate' THEN le.valuenum END) as phosphate_admit,
    MAX(CASE WHEN dnl.label = 'Potassium' THEN le.valuenum END) as potassium_admit,
    MAX(CASE WHEN dnl.label = 'Potassium, Whole Blood' THEN le.valuenum END) as potassium_whole_blood_admit
FROM labevents le
INNER JOIN d_nutrition_labs dnl ON le.itemid = dnl.itemid
INNER JOIN admissions a ON le.hadm_id = a.hadm_id
WHERE le.charttime BETWEEN a.admittime AND datetime(a.admittime, '+24 hours')
GROUP BY le.hadm_id;
CREATE VIEW "omr_admit_antho" AS SELECT
	subject_id,
	MAX(CASE WHEN result_name LIKE '%height%' AND rn =1 THEN result_value END) as height_in,
	MAX(CASE WHEN result_name LIKE '%weight%' AND rn =1 THEN result_value END) as weight_lb,
	MAX(CASE WHEN result_name LIKE '%bmi%' AND rn =1 THEN result_value END) as bmi,
	MAX(CASE WHEN result_name LIKE '%blood%' AND rn =1 THEN result_value END) as blood_pressure
FROM (
	SELECT
		subject_id,
		result_name,
		result_value,
		ROW_NUMBER() OVER (PARTITION BY subject_id, result_name ORDER BY chartdate ASC) as rn
	FROM omr)
WHERE rn = 1
GROUP BY subject_id;
COMMIT;
