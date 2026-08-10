# Data Request — Institutional Canvas Extract
## Draft Letter to Ms. Keitha Pearce, UNT

---

**[Date]**

Ms. Keitha Pearce
[Title / Office]
University of North Texas
Denton, TX

---

Dear Ms. Pearce,

I am writing on behalf of **[PI Name]**, Principal Investigator for the study
*"Early At-Risk Student Identification Using Learning Analytics"* (hereafter "the Study").
The Study aims to develop and validate predictive models that identify students who may
need academic support early in a course term, using de-identified traces of student
engagement and course design drawn from your institution's learning management system.

We are requesting a **governed, de-identified data extract** from UNT's Canvas
environment for a defined historical cohort of courses and terms. We are specifically
*not* requesting Canvas administrator credentials, instructor-level access, or any form
of direct system access. All data handling will comply with FERPA, UNT's institutional
data governance policies, and the IRB protocol under which this Study operates. A copy
of the IRB approval letter is available upon request.

The sections below specify precisely which fields we require, which fields are optional,
and which fields we explicitly do not need.

---

## 1. Required Fields

We request a flat or relational extract containing the following fields, organized by
category. **All identifiers must be pseudonymized** by the data-owning office before
delivery; we do not need — and ask that you not include — any value that could directly
re-identify a student, instructor, or course section.

### 1.1 Pseudonymized Identifiers

| Identifier | Description |
|---|---|
| `study_student_id` | Study-assigned pseudonymous student identifier |
| `study_course_id` | Study-assigned pseudonymous course identifier |
| `study_section_id` | Study-assigned pseudonymous section identifier |
| `study_term_id` | Study-assigned pseudonymous term identifier |
| `study_enrollment_id` | Study-assigned pseudonymous enrollment identifier |
| `study_assignment_id` | Study-assigned pseudonymous assignment identifier |
| `study_submission_id` | Study-assigned pseudonymous submission identifier |
| `study_resource_id` | Study-assigned pseudonymous VLE / resource identifier |

All identifiers in this table should be opaque tokens (e.g., UUID or integer) generated
and retained by UNT's data-owning office. UNT retains the identity crosswalk; we receive
only the pseudonymous tokens (see §6).

### 1.2 Course and Section Characteristics

- Subject / discipline code (e.g., department prefix)
- Course level (undergraduate, graduate)
- Delivery modality (face-to-face, fully online, hybrid)
- Term / semester code (mapped to `study_term_id`)
- Course/section official start date and end date (year-month precision is sufficient;
  exact calendar date is acceptable if already pseudonymized relative to term start)

### 1.3 Enrollment Fields

- Participant role (student, observer — instructors excluded or flagged)
- Enrollment status at end of term (active, completed, withdrawn, inactive)
- Enrollment creation date (relative to term start, in days, is preferred over an
  absolute date)
- Last-activity date, if available (relative to term start, in days)

### 1.4 Course-Design Fields (Assignment / Module Metadata)

- Assignment type (quiz, assignment, discussion, external tool, etc.)
- Points possible
- Grading type (points, percent, pass/fail, not graded, etc.)
- Due date (offset in days from term start)
- Unlock date (offset in days from term start), if set
- Lock date (offset in days from term start), if set
- Module membership (which module, if any, the assignment belongs to)
- Module position (ordinal position of the assignment within its module)

### 1.5 Learning Activity

Timestamped or daily-aggregated interaction logs linking a pseudonymous student, course,
and resource. Preferred granularity is **daily aggregates** (date offset from term start,
interaction count, resource type), but raw timestamped records are acceptable and will
be aggregated on our side. Minimum fields:

- `study_student_id`
- `study_course_id`
- `study_resource_id`
- Resource type / asset type (page, file, quiz, discussion, external URL, etc.)
- Date or timestamp of interaction (absolute timestamp or day-offset from term start)
- Interaction count (if pre-aggregated)

### 1.6 Submission Fields

- `study_submission_id`, `study_assignment_id`, `study_student_id`
- Submission timestamp (day-offset from term start, or absolute timestamp)
- Attempt number
- Score received
- Points possible at time of submission
- Late indicator (boolean or categorical)
- Missing indicator (boolean)

### 1.7 Outcome Fields

- Official final course grade (letter grade or numeric, as recorded in the SIS)
- Pass / fail / withdrawal / completion outcome (categorical)

---

## 2. Optional Demographic Fields

The following fields are requested **only if** the Study's IRB protocol is amended to
include a fairness or subgroup analysis. Please withhold these fields unless we notify
you of IRB approval for that analysis component.

- Age band (e.g., 18–24, 25–34, 35+) — not exact date of birth
- Gender (as recorded in the SIS)
- Race/ethnicity (as recorded in the SIS, aggregated per federal categories)
- First-generation college student indicator
- Pell Grant recipient indicator

If any of these fields require a separate data-sharing agreement or additional IRB
coverage at UNT, please let us know and we will coordinate accordingly before requesting
them.

---

## 3. Fields We Explicitly Do Not Need

To minimize re-identification risk, please **exclude** the following from the extract:

- Student legal name, preferred name, or any name field
- Email addresses (student or instructor)
- Student ID numbers (university-assigned numeric or alphanumeric IDs)
- Canvas login IDs or usernames
- Exact dates of birth
- Physical addresses or ZIP codes
- IP addresses
- Device identifiers or browser user-agent strings
- Complete URLs (resource type and a pseudonymous resource ID are sufficient)
- Submission file attachments or submission text/body content
- Discussion post text or reply content
- Inbox or announcement message content

---

## 4. De-Identification and Data Governance

We ask that UNT's data-owning office (or its designee) perform the following steps
**before** the extract is transmitted to our research team:

1. **Generate pseudonymous identifiers.** Replace all direct identifiers (student IDs,
   Canvas user IDs, course SIS IDs, etc.) with opaque study tokens as described in §1.1.
2. **Retain the identity crosswalk.** UNT retains the mapping between study tokens and
   real identifiers. Our team receives only the de-identified extract and never the
   crosswalk.
3. **Apply date shifting or offsetting.** Where absolute calendar dates appear, express
   them as offsets (in days) from the term start date, or apply a consistent random shift
   within a term so that term-relative patterns are preserved but absolute dates are
   obscured.
4. **Review for small-cell suppression.** If any category cell in the extract contains
   fewer than [institution threshold, e.g., 5] students, please suppress or aggregate
   that cell before delivery.

The extract should be delivered via [secure transfer method to be agreed upon, e.g.,
UNT's secure file transfer portal or an encrypted email attachment], and should be
accompanied by a brief data dictionary confirming the fields included and any
transformations applied.

---

## 5. Question: Historical Canvas Activity Log Retention

One consideration relevant to our study timeline is the retention period for Canvas
activity logs. **Canvas Data 2 (CD2) web-log streams typically have a short rolling
retention window** (often 30–60 days), meaning that historical activity logs from prior
terms may no longer be available in the live CD2 pipeline.

Could you confirm whether UNT has:

1. Archived historical Canvas activity logs in an institutional data warehouse or
   analytics platform (e.g., an Apereo Learning Analytics Processor, a Snowflake or
   Redshift warehouse, or a campus-managed extract)?
2. A defined retention period for those archived logs, and the earliest term for which
   reliable activity-log data is available?

If historical logs are not available for all requested terms, we would appreciate
guidance on the earliest term from which complete activity records can be extracted.
This will inform our cohort selection and any limitations we report in the Study.

---

## 6. Closing

We appreciate your time and assistance with this request. The data extract described
above will be used solely for the purposes of the Study identified in our IRB protocol,
will not be linked to any other dataset without separate IRB approval and your office's
consent, and will be destroyed or returned according to the data retention terms agreed
upon at the time of transfer.

Please do not hesitate to contact us with any questions about the scope of this request
or the Study's data governance procedures. We are happy to schedule a brief call to
discuss any concerns before data preparation begins.

Sincerely,

---

**[PI Name]**
[Title]
[Department / College]
[Institution]
[Email Address]
[Phone Number]

**[Date]**

---

*This letter was prepared in support of the Study's IRB data request. File version
controlled at `docs/keitha_pearce_data_request_draft.md`.*
