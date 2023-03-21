-- only patients and those with time since onset are included in this query
select ct_customer.customers.id, ct_customer.customers.condition_since, ct_customer.customers.age_group, ct_customer.customers_to_deficits.deficit_id, ct_customer.customers_to_disorders.disorder_id,
	constant_therapy.sessions.completed_task_count, constant_therapy.sessions.task_type_id, constant_therapy.sessions.task_level, constant_therapy.sessions.accuracy, constant_therapy.sessions.is_baseline
from ct_customer.customers
inner join ct_customer.customers_to_deficits on ct_customer.customers.id = ct_customer.customers_to_deficits.customer_id
inner join ct_customer.customers_to_disorders on ct_customer.customers.id = ct_customer.customers_to_disorders.customer_id
inner join constant_therapy.sessions on ct_customer.customers.id = constant_therapy.sessions.patient_id
where (ct_customer.customers.usertype = "patient") and (ct_customer.customers.condition_since is not NULL) and (ct_customer.customers.subscription_status is not NULL) and (ct_customer.customers_to_disorders.disorder_id <> "9999")
and (ct_customer.customers_to_deficits.deficit_id <> "9999")
order by id asc;