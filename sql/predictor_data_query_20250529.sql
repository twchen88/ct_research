-- Query to get data for the predictor, performs simple filtering
-- This query is based on constant_therapy.sessions, which is then joined with users, ct_customer.customers, ct_customer.customers_to_disorders, and ct_customer.customers_to_deficits
-- to ensure that the patient/session is valid and has the necessary information
-- The query filters out demo users, ensures the session has a parent_id, and checks for non-null values in accuracy, domain_scores, and domain_ids

-- Modified based on Claire Cordella's queries for her paper, sample code can be found at backup locations (Box drive or local backup)
SELECT sess.id, sess.patient_id, sess.start_time, sess.task_type_id, sess.task_level, sess.domain_ids, sess.domain_scores
FROM constant_therapy.sessions sess
JOIN constant_therapy.users u
                ON u.id = sess.patient_id
                AND NOT u.is_demo
JOIN ct_customer.customers c
                ON c.user_id = u.id
				AND (c.subscription_status IN ('flex', 'grandfathered', 'scholarship', 'expired', 'paying', 'veteran', 'active'))
JOIN ct_customer.customers_to_disorders cds
                ON cds.customer_id = c.id
JOIN ct_customer.customers_to_deficits cdf
                ON cdf.customer_id = c.id
WHERE sess.parent_id IS NOT NULL
	AND sess.patient_id != 31647
    AND sess.accuracy IS NOT NULL
    AND sess.domain_scores IS NOT NULL
    AND sess.domain_ids IS NOT NULL
    AND sess.type = 'SCHEDULED';