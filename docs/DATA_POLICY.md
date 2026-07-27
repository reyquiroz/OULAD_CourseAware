# Data Policy

## Dataset Used

This repository uses exclusively the **Open University Learning Analytics
Dataset (OULAD)**, a publicly available dataset released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
licence.

**Citation**:
> Kuzilek J., Hlosta M., Zdrahal Z. (2017) Open University Learning Analytics
> dataset. *Scientific Data* 4:170171 doi:[10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171)

**Download**: https://analyse.kmi.open.ac.uk/open_dataset

### Canonical OULAD files

The `data/` directory must contain **only** these seven files:

| File | Description |
|------|-------------|
| `assessments.csv` | Assessment metadata (type, weight, due date) |
| `courses.csv` | Course-presentation metadata |
| `studentAssessment.csv` | Student submission scores |
| `studentInfo.csv` | Student demographic data and final outcomes |
| `studentRegistration.csv` | Registration and unregistration dates |
| `studentVle.csv` | Student VLE interaction clicks by day (433 MB — gitignored) |
| `vle.csv` | VLE resource metadata |

---

## What Is Not in This Repository

The following categories of data **must not** be committed to this repository
under any circumstances:

| Category | Examples | Why excluded |
|----------|----------|--------------|
| Employer-generated exports | IBM learning platform completion reports | Unrelated to OULAD research; may contain PII |
| Personally identifiable information | Employee names, email addresses, employee IDs | Privacy risk; not needed for the research |
| Institutional student records | Canvas LMS exports, university database extracts | Subject to IRB approval and data sharing agreements; must not be stored in a public repository |
| Proprietary or restricted datasets | Any dataset not listed in the Canonical OULAD files table above | Not covered by the CC BY 4.0 licence |

### Enforcement

`.gitignore` contains explicit block patterns to prevent accidental commits of
known problem file types:

```
data/*Completion*.csv
data/*Report*.csv
data/raw/*Completion*.csv
data/raw/*Report*.csv
```

See also the **Data Directory Policy** section in `CONTRIBUTING.md`.

---

## Canvas Authorization and IRB

This project **does not currently use any Canvas data or data collected
under an IRB protocol**. All analyses are performed on the public OULAD dataset.

If future work involves:

- **Canvas LMS data** — a formal data access agreement with the relevant
  institution must be in place before any data is collected or processed. No
  Canvas data should be stored in this repository.
- **IRB-approved data collection** — any data collected under an IRB protocol
  must be stored in a secure location external to this repository (e.g., a
  university-managed storage system). Reference the data by absolute path in a
  local configuration file that is gitignored (e.g., `local_config.py` or
  `.env`). The gitignored configuration file itself should never contain
  identifiable data — only paths.

A suggested pattern for referencing external data:

```python
# local_config.py  (gitignored — never commit this file)
EXTERNAL_DATA_PATH = "/secure/institutional/storage/study_data.csv"
```

```gitignore
# .gitignore
local_config.py
.env
```

---

## Summary

| Question | Answer |
|----------|--------|
| Does the repository contain PII? | No |
| Does the repository contain Canvas data? | No |
| Does the repository contain IRB-protected data? | No |
| What licence covers the data? | CC BY 4.0 (OULAD) |
| Where to download the data? | https://analyse.kmi.open.ac.uk/open_dataset |
| What protects against future accidental PII commits? | `.gitignore` block patterns + `CONTRIBUTING.md` policy |
