select constant_therapy.sessions.id, constant_therapy.sessions.patient_id, constant_therapy.sessions.completed_task_count, constant_therapy.sessions.task_type_id, constant_therapy.sessions.task_level, constant_therapy.sessions.accuracy, constant_therapy.sessions.is_baseline
from constant_therapy.sessions
where constant_therapy.sessions.accuracy is not null
order by patient_id asc;