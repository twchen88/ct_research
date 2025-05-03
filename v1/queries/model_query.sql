select sess.id, sess.patient_id, sess.start_time, sess.task_type_id, sess.task_level, sess.domain_ids, sess.domain_scores
from constant_therapy.sessions sess
join constant_therapy.users u
                on u.id = sess.patient_id
                and not u.is_demo
join ct_customer.customers c
                on c.user_id = u.id
				and (c.subscription_status IN ('flex', 'grandfathered', 'scholarship', 'expired', 'paying', 'veteran', 'active'))
join ct_customer.customers_to_disorders cds
                on cds.customer_id = c.id
join ct_customer.customers_to_deficits cdf
                on cdf.customer_id = c.id
where sess.parent_id is not NULL
	and sess.patient_id != 31647
    and sess.accuracy is not null
    and sess.domain_scores is not null
    and sess.domain_ids is not null
    and sess.type = 'SCHEDULED';