--  table created in constant_therapy database as context_action_def
-- table not working
select sess.id, sess.patient_id, sess.task_type_id, sess.task_level, sess.completed_task_count, sess.accuracy, tp.domain_id, ps.condition_since, ps.birth_year, def.deficit_id,
DATE(sess.start_time) AS start_time, sess.start_time AS start_time_min, DATE(sess.end_time) AS end_time, sess.end_time as end_time_min
from constant_therapy.sessions sess
join (
select domain_id, task_type_id, task_level from constant_therapy.task_progression
) tp on tp.task_type_id = sess.task_type_id and tp.task_level = sess.task_level
join (
select user_id, id, condition_since, age_group, birth_year, usertype, created_date, subscription_status from ct_customer.customers
) ps on ps.user_id = sess.patient_id
join (
select customer_id, deficit_id from ct_customer.customers_to_deficits
) def on def.customer_id = ps.id
where (sess.accuracy is not null) and (ps.condition_since is not null) and (ps.birth_year is not null) and (ps.birth_year <> 0) and (ps.usertype = "patient");