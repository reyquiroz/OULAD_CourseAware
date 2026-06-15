libname stud 'C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD';

/*
code_module: An identification code for the specific course (e.g., AAA, BBB, CCC).
code_presentation: The identification code of the semester or presentation in which the course was offered (e.g., 2013B for a February start, or 2013J for an October start).
id_student: A unique, anonymized identification number assigned to each student.
id_site: An identification number representing the exact VLE material or resource the student interacted with (such as a PDF, forum page, or HTML content).
date: The day of the student's interaction with the material, measured as the number of days since the start of that module's presentation.
sum_click: The total number of times the student interacted with (clicked on) the specific VLE material on that particular day.*/

proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\studentVle.csv"
out=stdt_vle
dbms=csv
replace;
getnames =YES;
run;

data std_v1; 
set stdt_vle;
run;

proc print data=std_v1 (obs=2); run;

proc freq data=std_v1;
tables code_presentation;
run;


proc freq data=std_v1;
tables id_site;
run;


proc freq data=std_v1;
tables code_module;
run;

proc sql;
    create table duplicate_counts as
    select *, count(code_module) as Row_Count
    from std_v1
    group by code_module
    having count(*) > 1;
quit;


proc print data=duplicate_counts (obs=2); run;

/*1 Assessments*/
proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\assessments.csv"
out=assessments
dbms=csv
replace;
getnames =YES;
run;

/*Courses*/
proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\courses.csv"
out=courses
dbms=csv
replace;
getnames =YES;
run;

/*Student Assessment*/
proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\studentAssessment.csv"
out=std_asses
dbms=csv
replace;
getnames =YES;
run;

/*Student Info*/
proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\studentInfo.csv"
out=stdt_info
dbms=csv
replace;
getnames =YES;
run;

/*Student Registration*/
proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\studentRegistration.csv"
out=stdt_reg
dbms=csv
replace;
getnames =YES;
run;

/*VLE*/
proc import 
datafile = "C:\Users\rq0016\Documents\Personal 2024-2025\phd student success\New_approach\Data\Dataset_1_OULAD\vle.csv"
out=vle
dbms=csv
replace;
getnames =YES;
run;

/*Understanding the data model
StudentAssessment
    + Assessments
    + StudentInfo
    + Courses
    + StudentRegistration
    + (Aggregated StudentVle)
*/





proc contents data=assessments;
run;

data asses2 (drop = date weight);
set assessments;
Asses_date_N = input(date, best12.); 
weight_N  = input(weight, best12.);
run;

proc print data=asses2 (obs=2); run;

proc contents data=courses;
run;

data std_asses2 (drop = date_submitted is_banked score );
set std_asses;
date_submitted_N = input(date_submitted, best12.); 
is_banked_N  = input(is_banked, best12.);
score_N  = input(score, best12.);
run;

proc print data=std_asses2 (obs=2); run;

proc contents data=std_assess;
run;

data courses2 (drop = module_presentation_length);
set courses;
mod_pres_leng_N = input(module_presentation_length, best12.); 
run;

proc print data=courses2 (obs=2); run;

proc contents data=stdt_info;
run;

proc print data=stdt_info2 (obs=2); run;

data stdt_info2 (drop = num_of_prev_attempts studied_credits);
set stdt_info;
num_of_prev_attempts_N = input(num_of_prev_attempts, best12.); 
studied_credits_N  = input(studied_credits, best12.);
run;

proc contents data=stdt_reg;
run;

data stdt_reg2 (drop = date_registration date_unregistration );
set stdt_reg;
date_registration_N = input(date_registration, best12.); 
date_unregistration_N  = input(date_unregistration, best12.);
run;

proc print data=stdt_reg2 (obs=2); run;



proc contents data=stdt_vle2;
run;
data stdt_vle2 (drop = date sum_click);
set stdt_vle;
date_N = input(date, best12.); 
sum_click_N  = input(sum_click, best12.);
run;


proc print data=stdt_vle2 (obs=2); run;

proc contents data=vle;
run;

data vle2 (drop = week_from week_to);
set vle;
week_from_N = input(week_from, best12.); 
week_to_N  = input(week_to, best12.);
run;

proc print data=vle2 (obs=2); run;

/*Aggregate VLE to avoid duplicate explosion*/
proc sql;
    create table stdt_vle_agg as
    select 
        id_student,
        code_module,
        code_presentation,
        sum(sum_click_N) as total_clicks
    from stdt_vle2
    group by id_student, code_module, code_presentation;
quit;


proc print data=stdt_vle_agg (obs=2); run;

/*Master relational table*/


proc sql;
    create table master as
    select 
        /* Core identifiers */
        sa.id_student,
        sa.id_assessment,
        a.code_module,
        a.code_presentation,

        /* Assessment info */
        a.assessment_type,
        a.Asses_date_N as asses_date,
        a.weight_N,

        /* Student performance */
        sa.date_submitted_N,
        sa.score_N,
        sa.is_banked_N,

        /* Student demographics */
        si.gender,
        si.region,
        si.highest_education,
        si.imd_band,
        si.age_band,
        si.num_of_prev_attempts_N,
        si.studied_credits_N,
        si.disability,
        si.final_result,

        /* Course info */
        c.mod_pres_leng_N,

        /* Registration info */
        sr.date_registration_N,
        sr.date_unregistration_N,

        /* Engagement */
        v.total_clicks

    from std_asses2 as sa

    /* Join Assessments */
    left join asses2 as a
        on sa.id_assessment = a.id_assessment

    /* Join StudentInfo (composite key) */
    left join stdt_info2 as si
        on sa.id_student = si.id_student
        and a.code_module = si.code_module
        and a.code_presentation = si.code_presentation

    /* Join Courses */
    left join courses2 as c
        on si.code_module = c.code_module
        and si.code_presentation = c.code_presentation

    /* Join Registration */
    left join stdt_reg2 as sr
        on si.id_student = sr.id_student
        and si.code_module = sr.code_module
        and si.code_presentation = sr.code_presentation

    /* Join aggregated VLE */
    left join stdt_vle_agg as v
        on si.id_student = v.id_student
        and si.code_module = v.code_module
        and si.code_presentation = v.code_presentation
    ;
quit;

proc print data=stdt_info2  (obs=2); run;

proc freq data=stdt_info2;
tables final_result;
run;


